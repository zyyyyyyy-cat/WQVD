"""Export matched UAD/A1 features once, then diagnose retrieval offline."""
import argparse
import csv
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from utils.a1_diagnostics import (write_csv, protocols, evaluate, local_statistics,
                                  compare_queries, transition_summary, FusionFeatures)

LOGGER = logging.getLogger('transreid')


def fingerprint(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return {'path': str(Path(path).resolve()), 'sha256': h.hexdigest()}


def configuration(path, kind, args):
    from config import cfg as defaults
    cfg = defaults.clone()
    cfg.merge_from_file(path)
    cfg.merge_from_list(args.opts)
    if args.data_root:
        cfg.DATASETS.ROOT_DIR = args.data_root
    if args.batch_size:
        cfg.TEST.IMS_PER_BATCH = args.batch_size
    if args.workers is not None:
        cfg.DATALOADER.NUM_WORKERS = args.workers
    if cfg.MODEL.FEATURE_AUG.TYPE != kind:
        raise ValueError('Expected {} config: {}'.format(kind, path))
    if cfg.TEST.NECK_FEAT != 'before' or cfg.TEST.FEAT_NORM != 'yes' or cfg.TEST.RE_RANKING or cfg.TEST.TOP_K_EVAL:
        raise ValueError('Diagnostics require NECK_FEAT=before, FEAT_NORM=yes, RE_RANKING=False, TOP_K_EVAL=0')
    cfg.MODEL.PRETRAIN_CHOICE = 'none'
    cfg.MODEL.DIST_TRAIN = False
    cfg.SOLVER.RESUME = ''
    cfg.freeze()
    return cfg


def audit_checkpoint(path):
    from utils.checkpoint import read_checkpoint
    ckpt = read_checkpoint(path)
    return {**fingerprint(path), 'epoch': ckpt.get('epoch'),
            'world_size': ckpt.get('world_size'),
            'training_signature': ckpt.get('training_signature'),
            'training_details_status': 'available' if 'training_signature' in ckpt else 'unknown_model_only_checkpoint'}


def export(args, cache):
    from datasets import make_dataloader
    from model import make_model
    from utils.checkpoint import load_model_checkpoint
    if cache.exists() and any(cache.iterdir()):
        raise ValueError('Export needs an empty cache directory; choose another --cache-dir: ' + str(cache))
    cfg0 = configuration(args.uad_config, 'none', args)
    cfg1 = configuration(args.a1_config, 'a1', args)
    for section in ('DATASETS', 'INPUT'):
        if dict(getattr(cfg0, section)) != dict(getattr(cfg1, section)):
            raise ValueError('UAD/A1 must use identical ' + section)
    if cfg0.MODEL.TRANSFORMER_TYPE != cfg1.MODEL.TRANSFORMER_TYPE or list(cfg0.MODEL.STRIDE_SIZE) != list(cfg1.MODEL.STRIDE_SIZE):
        raise ValueError('Backbone and stride must match')
    # A single validation loader supplies BOTH models; no separate ordering to drift.
    _, _, loaders, num_queries, classes, cameras, views = make_dataloader(cfg1)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; use --device cpu for a CPU environment')
    torch.manual_seed(cfg1.SOLVER.SEED)
    models = []
    for cfg, path in [(cfg0, args.uad_weight), (cfg1, args.a1_weight)]:
        model = make_model(cfg, classes, cameras, views)
        load_model_checkpoint(model, path, mode='eval')
        models.append(model.to(device).eval())
    count = sum(len(loader.dataset) for loader in loaders)
    nq = sum(num_queries)
    if nq == 0 or count == nq:
        raise ValueError('Nonempty query and gallery required')
    cache.mkdir(parents=True, exist_ok=True)
    arrays = {name: np.lib.format.open_memmap(cache / (name + '.npy'), mode='w+',
              dtype='float32', shape=(count, 768))
              for name in ('uad_cls', 'a1_cls', 'a1_local')}
    indices_store = weights_store = None
    stability = np.lib.format.open_memmap(cache / 'stability.npy', mode='w+', dtype='float32', shape=(count,)) if args.stability else None
    meta = [None] * count
    qoffset, goffset = 0, nq
    torch.backends.cudnn.benchmark = False
    with torch.inference_mode():
        for loader, query_count in zip(loaders, num_queries):
            offset = 0
            for images, pids, camids, _, mids, paths in loader:
                images = images.to(device)
                a = models[0].diagnostic_features(images)
                b = models[1].diagnostic_features(images)
                torch.testing.assert_close(b['fusion'], b['cls'] + cfg1.MODEL.FEATURE_AUG.GAMMA * b['local'], rtol=0, atol=0)
                if indices_store is None:
                    shape = (count, *b['indices'].shape[1:])
                    indices_store = np.lib.format.open_memmap(cache / 'indices.npy', mode='w+', dtype='int32', shape=shape)
                    weights_store = np.lib.format.open_memmap(cache / 'weights.npy', mode='w+', dtype='float32', shape=shape)
                target = np.array([qoffset + j if j < query_count else goffset + j - query_count
                                   for j in range(offset, offset + len(images))])
                for name, value in [('uad_cls', a['cls']), ('a1_cls', b['cls']), ('a1_local', b['local'])]:
                    if not torch.isfinite(value).all():
                        raise ValueError('Non-finite exported ' + name)
                    arrays[name][target] = value.cpu().float().numpy()
                indices_store[target] = b['indices'].cpu().numpy()
                weights_store[target] = b['weights'].cpu().float().numpy()
                if stability is not None:
                    # Brightness perturbation keeps patch coordinates unchanged.
                    mean = images.new_tensor(cfg1.INPUT.PIXEL_MEAN)[None, :, None, None]
                    std = images.new_tensor(cfg1.INPUT.PIXEL_STD)[None, :, None, None]
                    changed = ((images * std + mean) * args.brightness_factor).clamp(0, 1)
                    alt = models[1].diagnostic_features((changed - mean) / std)['indices']
                    overlap = (b['indices'].unsqueeze(-1) == alt.unsqueeze(-2)).any(-1).float().mean((1, 2))
                    stability[target] = overlap.cpu().numpy()
                for j, idx in enumerate(target):
                    mid = int(mids[j])
                    modality = {1: 'RGB', 2: 'NIR', 3: 'TIR'}[mid]
                    platform = 'G' if int(camids[j]) < 5 else 'A'
                    meta[idx] = {'sample_index': int(idx), 'split': 'query' if idx < nq else 'gallery',
                                 'pid': int(pids[j]), 'camera_id': int(camids[j]),
                                 'modality_id': mid, 'modality': modality,
                                 'platform': platform, 'domain': modality + '-' + platform,
                                 'path': str(paths[j])}
                offset += len(images)
                if offset % (10 * cfg1.TEST.IMS_PER_BATCH) == 0:
                    LOGGER.info('Exported %d/%d images in modality loader', offset, len(loader.dataset))
            if offset != len(loader.dataset):
                raise ValueError('Validation loader did not cover its dataset exactly')
            qoffset += query_count
            goffset += len(loader.dataset) - query_count
    if any(m is None for m in meta):
        raise ValueError('Incomplete export manifest')
    for array in [*arrays.values(), indices_store, weights_store, stability]:
        if array is not None:
            array.flush()
    write_csv(cache / 'samples.csv', meta)
    audit = {'format_version': 1, 'count': count, 'num_query': nq,
             'gamma': cfg1.MODEL.FEATURE_AUG.GAMMA,
             'uad': audit_checkpoint(args.uad_weight), 'a1': audit_checkpoint(args.a1_weight),
             'uad_config': cfg0.dump(), 'a1_config': cfg1.dump(),
             'samples_sha256': fingerprint(cache / 'samples.csv')['sha256'],
             'stability_brightness_factor': args.brightness_factor if args.stability else None,
             'inference': 'eval, float32, no autocast, raw pre-BN features',
             'ordering': 'all modality queries, then all modality galleries; shared loader'}
    # Completion marker written last: incomplete caches cannot be analyzed.
    (cache / 'manifest.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding='utf-8')
    LOGGER.info('Feature export complete: %s', cache)


def load_cache(cache):
    audit = json.loads((cache / 'manifest.json').read_text(encoding='utf-8'))
    if audit['format_version'] != 1 or fingerprint(cache / 'samples.csv')['sha256'] != audit['samples_sha256']:
        raise ValueError('Invalid cache version or modified sample manifest')
    with open(cache / 'samples.csv', encoding='utf-8-sig', newline='') as stream:
        meta = list(csv.DictReader(stream))
    for i, row in enumerate(meta):
        for k in ('sample_index', 'pid', 'camera_id', 'modality_id'):
            row[k] = int(row[k])
        if row['sample_index'] != i or row['split'] != ('query' if i < audit['num_query'] else 'gallery'):
            raise ValueError('Cache sample ordering mismatch')
    if len(meta) != audit['count']:
        raise ValueError('Cache sample count mismatch')
    arrays = {k: np.load(cache / (k + '.npy'), mmap_mode='r')
              for k in ('uad_cls', 'a1_cls', 'a1_local', 'indices', 'weights')}
    for name, value in arrays.items():
        if len(value) != len(meta):
            raise ValueError('Cache feature count mismatch: ' + name)
    shape = arrays['a1_cls'].shape
    if len(shape) != 2 or any(arrays[k].shape != shape or arrays[k].dtype != np.float32
                              for k in ('uad_cls', 'a1_cls', 'a1_local')):
        raise ValueError('Cache features must have equal NxD shapes and float32 dtype')
    if arrays['indices'].ndim != 3 or arrays['indices'].shape != arrays['weights'].shape:
        raise ValueError('Cache Top-K index/weight shapes mismatch')
    if not np.isfinite(audit['gamma']) or audit['gamma'] < 0:
        raise ValueError('Invalid cached fusion gamma')
    return audit, meta, arrays


def analyze(args, cache, output):
    audit, meta, arrays = load_cache(cache)
    stability = np.load(cache / 'stability.npy', mmap_mode='r') if audit['stability_brightness_factor'] is not None else None
    gamma = audit['gamma']
    local = []
    for start in range(0, len(meta), 1024):
        stop = start + 1024
        local.extend(local_statistics(arrays['a1_cls'][start:stop], arrays['a1_local'][start:stop],
                     arrays['indices'][start:stop], arrays['weights'][start:stop], gamma,
                     stability[start:stop] if stability is not None else None))
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'a1_local_diagnostics.csv', [{**m, **r} for m, r in zip(meta, local)])
    variants = {'UAD-CLS': arrays['uad_cls'], 'A1-CLS': arrays['a1_cls'],
                'A1-Fusion': FusionFeatures(arrays['a1_cls'], arrays['a1_local'], gamma),
                'A1-Local': arrays['a1_local']}
    for value in args.gammas:
        if value < 0 or not np.isfinite(value):
            raise ValueError('Sensitivity gammas must be finite and nonnegative')
        if value not in (0, gamma):
            variants['A1-Gamma-' + str(value)] = FusionFeatures(arrays['a1_cls'], arrays['a1_local'], value)
    summaries, transitions = [], []
    query_path = output / 'query_comparison.csv'
    with open(query_path, 'w', newline='', encoding='utf-8-sig') as stream:
        writer = None
        for protocol, qi, gi in protocols(meta, args.detailed_domains):
            LOGGER.info('Analyze %s: queries=%d gallery=%d', protocol, len(qi), len(gi))
            results, protocol_rows = {}, {}
            for name, features in variants.items():
                records, stats = evaluate(features, meta, qi, gi, args.chunk_size)
                results[name] = records
                row = {'feature': name, 'protocol': protocol, 'query_count': len(qi),
                       'gallery_count': len(gi), **stats}
                summaries.append(row)
                protocol_rows[name] = row
            for label, before, after in [('train', 'UAD-CLS', 'A1-CLS'), ('fusion', 'A1-CLS', 'A1-Fusion'), ('total', 'UAD-CLS', 'A1-Fusion')]:
                for field in ('mAP', 'mINP', 'rank1', 'rank5', 'rank10'):
                    if field in protocol_rows[before]:
                        protocol_rows[after][label + '_delta_' + field + '_pp'] = 100 * (protocol_rows[after][field] - protocol_rows[before][field])
            rows = compare_queries(protocol, results, meta, local)
            if rows:
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                writer.writerows(rows)
                transitions.extend(transition_summary(protocol, rows))
    write_csv(output / 'feature_comparison.csv', summaries)
    write_csv(output / 'rank1_transitions.csv', transitions)
    groups = []
    for split in ('query', 'gallery'):
        for mid in ('RGB', 'NIR', 'TIR'):
            selected = [r for m, r in zip(meta, local) if m['split'] == split and m['modality'] == mid]
            row = {'split': split, 'modality': mid, 'count': len(selected)}
            for field in ('rho', 'cos_cls_local', 'direction_delta', 'topk_overlap', 'topk_entropy', 'topk_entropy_normalized', 'topk_stability'):
                values = [r[field] for r in selected if r[field] is not None]
                for quantile in (0.05, 0.5, 0.95):
                    row[field + '_p' + str(int(100 * quantile))] = float(np.quantile(values, quantile)) if values else None
                row[field + '_mean'] = float(np.mean(values)) if values else None
            groups.append(row)
    write_csv(output / 'a1_local_summary.csv', groups)
    run = {'cache_manifest': audit, 'gamma_sensitivity': args.gammas,
           'detailed_domains': args.detailed_domains, 'chunk_size': args.chunk_size,
           'metric_units': 'metrics/AP/INP/fractions: 0..1; summary deltas: percentage points',
           'distance': 'squared Euclidean after float32 L2 normalization',
           'caveats': ['A1-CLS is not a retrained ablation.', 'Gamma sweep is inference-only sensitivity.',
                       'Training seed/updates cannot be recovered from model-only weights.',
                       'Independent checkpoint feature vectors are not compared by cosine.',
                       'No causal conclusions or statistical significance from one checkpoint pair.']}
    (output / 'analysis_manifest.json').write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding='utf-8')
    LOGGER.info('Analysis complete: %s', output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['all', 'export', 'analyze'], default='all')
    parser.add_argument('--uad-config', default=str(ROOT / 'configs/UAD.yml'))
    parser.add_argument('--a1-config', default=str(ROOT / 'configs/UAD_A1.yml'))
    parser.add_argument('--uad-weight')
    parser.add_argument('--a1-weight')
    parser.add_argument('--data-root')
    parser.add_argument('--output-dir', default='logs/A1_vs_UAD_analysis')
    parser.add_argument('--cache-dir')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--chunk-size', type=int, default=64)
    parser.add_argument('--gammas', nargs='*', type=float, default=[])
    parser.add_argument('--detailed-domains', action='store_true')
    parser.add_argument('--stability', action='store_true')
    parser.add_argument('--brightness-factor', type=float, default=1.05)
    parser.add_argument('--opts', nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if int(os.environ.get('WORLD_SIZE', '1')) > 1:
        parser.error('Use python on one process; do not torchrun this offline tool')
    if args.chunk_size <= 0 or (args.batch_size is not None and args.batch_size <= 0) or (args.workers is not None and args.workers < 0):
        parser.error('Invalid chunk/batch/worker size')
    if not np.isfinite(args.brightness_factor) or args.brightness_factor <= 0:
        parser.error('Brightness factor must be finite and positive')
    output = Path(args.output_dir)
    cache = Path(args.cache_dir) if args.cache_dir else output / 'features'
    if args.stage in ('all', 'export'):
        if not args.uad_weight or not args.a1_weight:
            parser.error('Export requires --uad-weight and --a1-weight')
        export(args, cache)
    if args.stage in ('all', 'analyze'):
        analyze(args, cache, output)


if __name__ == '__main__':
    main()
