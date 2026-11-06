"""Strict model loading and epoch-boundary single-process/DDP training state."""
import json
import logging
import os
import random
import hashlib
from collections.abc import Mapping

import numpy as np
import torch
import torch.distributed as dist


def read_checkpoint(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:  # Older PyTorch used by the original project.
        return torch.load(path, map_location='cpu')


def clean_state_dict(checkpoint):
    state = checkpoint
    for name in ('model', 'state_dict'):
        if isinstance(state, Mapping) and name in state:
            state = state[name]
    if not isinstance(state, Mapping):
        raise ValueError('Checkpoint does not contain a model state dictionary')
    result = {}
    for key, value in state.items():
        while key.startswith('module.'):
            key = key[len('module.'):]
        if key in result:
            raise ValueError('Duplicate key after stripping module prefix: ' + key)
        if not isinstance(value, torch.Tensor):
            raise ValueError('Non-tensor model state: ' + key)
        result[key] = value
    return result


def _validate_state(model, state, allowed_missing=()):
    current = model.state_dict()
    missing = set(current) - set(state)
    unexpected = set(state) - set(current)
    mismatch = [k for k in set(state) & set(current) if state[k].shape != current[k].shape]
    if missing != set(allowed_missing) or unexpected or mismatch:
        raise ValueError('Checkpoint mismatch: missing={}, unexpected={}, shapes={}'.format(
            sorted(missing), sorted(unexpected), sorted(mismatch)))


def load_model_checkpoint(model, path, mode='eval'):
    if mode not in ('eval', 'resume', 'uad_init'):
        raise ValueError('Unknown checkpoint loading mode: ' + mode)
    checkpoint = read_checkpoint(path)
    state = clean_state_dict(checkpoint)
    metadata = checkpoint.get('metadata')
    sidecar = str(path) + '.meta.json'
    if metadata is None and os.path.isfile(sidecar):
        with open(sidecar, encoding='utf-8') as stream:
            metadata = json.load(stream)
    if mode == 'uad_init':
        if metadata is not None and metadata.get('type') != 'none':
            raise ValueError('uad_init must use a Baseline UAD checkpoint')
        roots = ('feature_aug.', 'local_bottleneck.', 'local_classifier.')
        allowed = {k for k in model.state_dict() if k.startswith(roots)}
    else:
        allowed = set()
        if metadata is not None and metadata != model.checkpoint_metadata:
            raise ValueError('Model/config metadata does not match checkpoint')
        if metadata is None:
            logging.getLogger('transreid').warning(
                'Legacy raw checkpoint has no metadata: strictly checking all keys/shapes, '
                'but non-tensor settings (e.g. GAMMA) cannot be verified.')
    _validate_state(model, state, allowed)
    result = model.load_state_dict(state, strict=not allowed)
    logging.getLogger('transreid').info(
        'Loaded %s (%s): %d tensors; initialized new tensors: %s',
        path, mode, len(state), sorted(result.missing_keys))
    return checkpoint


def write_model_metadata(path, model):
    with open(str(path) + '.meta.json', 'w', encoding='utf-8') as stream:
        json.dump(model.checkpoint_metadata, stream, indent=2)


def _plain(value):
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    return value


def training_signature(cfg):
    sections = {k: _plain(getattr(cfg, k)) for k in
                ('MODEL', 'INPUT', 'DATASETS', 'DATALOADER', 'SOLVER', 'TEST')}
    for k in ('PRETRAIN_CHOICE', 'PRETRAIN_PATH', 'DEVICE_ID'):
        sections['MODEL'].pop(k, None)
    for k in ('RESUME', 'LOG_PERIOD', 'CHECKPOINT_PERIOD'):
        sections['SOLVER'].pop(k, None)
    sections['TEST'].pop('WEIGHT', None)
    return sections


def _rng_state():
    state = {'python': random.getstate(), 'numpy': np.random.get_state(),
             'torch': torch.get_rng_state()}
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state(torch.cuda.current_device())
    return state


def _restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if 'cuda' in state:
        torch.cuda.set_rng_state(state['cuda'], torch.cuda.current_device())


def _loader_signature(loader):
    # PKM __iter__ updates its estimated length: do not compare len(loader).
    source = getattr(loader.dataset, 'dataset', None)
    manifest = json.dumps(_plain(source), ensure_ascii=True, sort_keys=True).encode('utf-8')
    return {'dataset_length': len(loader.dataset),
            'manifest_sha256': hashlib.sha256(manifest).hexdigest(),
            'batch_size': loader.batch_sampler.batch_size,
            'drop_last': loader.batch_sampler.drop_last,
            'num_workers': loader.num_workers,
            'persistent_workers': loader.persistent_workers}


def save_training_checkpoint(path, model, cfg, epoch, optimizer, optimizer_center,
                             scheduler, scaler, bank, center_criterion, loader):
    distributed = dist.is_available() and dist.is_initialized()
    world = dist.get_world_size() if distributed else 1
    rank = dist.get_rank() if distributed else 0
    # BN statistics may differ by rank between validation epochs.
    runtime = {'rng': _rng_state(), 'buffers': {
        k: v.detach().cpu().clone() for k, v in model.named_buffers()}}
    states = [None] * world
    if distributed:
        dist.all_gather_object(states, runtime)
    else:
        states[0] = runtime
    if rank == 0:
        checkpoint = {
            'format_version': 1, 'model': model.state_dict(),
            'metadata': model.checkpoint_metadata, 'config': cfg.dump(),
            'training_signature': training_signature(cfg),
            'loader_signature': _loader_signature(loader),
            'epoch': epoch, 'world_size': world, 'rank_states': states,
            'optimizer': optimizer.state_dict(),
            'optimizer_center': optimizer_center.state_dict(),
            'center_criterion': center_criterion.state_dict(),
            'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
            'prototype_bank': bank.state_dict() if bank is not None else None,
        }
        temp = str(path) + '.tmp'
        torch.save(checkpoint, temp)
        os.replace(temp, path)
    if distributed:
        dist.barrier()


def restore_training_checkpoint(path, model, cfg, optimizer, optimizer_center,
                                scheduler, scaler, bank, center_criterion, loader):
    checkpoint = read_checkpoint(path)
    required = {'format_version', 'model', 'metadata', 'epoch', 'world_size',
                'rank_states', 'optimizer', 'optimizer_center', 'center_criterion',
                'scheduler', 'scaler', 'prototype_bank', 'training_signature',
                'loader_signature'}
    if not required.issubset(checkpoint) or checkpoint['format_version'] != 1:
        raise ValueError('Resume requires a full *_resume.pth checkpoint, not model-only weights')
    distributed = dist.is_available() and dist.is_initialized()
    world = dist.get_world_size() if distributed else 1
    rank = dist.get_rank() if distributed else 0
    if checkpoint['world_size'] != world or len(checkpoint['rank_states']) != world:
        raise ValueError('Resume requires the same world size')
    if checkpoint['metadata'] != model.checkpoint_metadata:
        raise ValueError('Resume model metadata mismatch')
    if checkpoint['training_signature'] != training_signature(cfg):
        raise ValueError('Resume training configuration mismatch')
    if checkpoint['loader_signature'] != _loader_signature(loader) or loader.persistent_workers:
        raise ValueError('Resume loader mismatch or unsupported persistent worker RNG')
    if (bank is None) != (checkpoint['prototype_bank'] is None):
        raise ValueError('Resume prototype bank mismatch')
    state = clean_state_dict(checkpoint)
    _validate_state(model, state)
    model.load_state_dict(state, strict=True)
    optimizer.load_state_dict(checkpoint['optimizer'])
    optimizer_center.load_state_dict(checkpoint['optimizer_center'])
    center_criterion.load_state_dict(checkpoint['center_criterion'], strict=True)
    scheduler.load_state_dict(checkpoint['scheduler'])
    scaler.load_state_dict(checkpoint['scaler'])
    if bank is not None:
        bank.load_state_dict(checkpoint['prototype_bank'], strict=True)
    buffers = dict(model.named_buffers())
    for k, value in checkpoint['rank_states'][rank]['buffers'].items():
        buffers[k].copy_(value.to(buffers[k].device))
    _restore_rng(checkpoint['rank_states'][rank]['rng'])
    return checkpoint['epoch']
