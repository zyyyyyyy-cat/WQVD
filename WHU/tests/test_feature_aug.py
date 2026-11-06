"""CPU acceptance tests; no dataset or pretrained weights required."""
import copy
import gc
import importlib
import importlib.util
import random
import sys
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import cfg as defaults
from model import make_model
from model.backbones.vit_pytorch import TransReID
from model.modules import A1LocalEnhancement, A2FeatureModulation
from processor.processor import PCA, GPD, PrototypeBank
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from utils.checkpoint import (load_model_checkpoint, save_training_checkpoint,
                              restore_training_checkpoint, write_model_metadata)

torch.set_num_threads(2)


def configuration(kind='none'):
    cfg = defaults.clone()
    cfg.merge_from_file(str(ROOT / 'configs' / ('UAD.yml' if kind == 'none' else 'UAD_' + kind.upper() + '.yml')))
    cfg.MODEL.PRETRAIN_CHOICE = 'none'
    return cfg


def identity_loss(cfg, classes=3):
    module = importlib.import_module('loss.make_loss')
    center_cls = module.CenterLoss
    with patch.object(module, 'CenterLoss', side_effect=lambda **kw: center_cls(
            num_classes=kw['num_classes'], feat_dim=kw['feat_dim'], use_gpu=False)):
        criterion, center = module.make_loss(cfg, classes)
    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        criterion.id_criterion.use_gpu = False  # CPU testing only; original criterion.
    return criterion, center


def assert_gradient(parameter):
    assert parameter.grad is not None
    assert torch.isfinite(parameter.grad).all()
    assert parameter.grad.abs().sum() > 0


def cpu_scaler():
    # CPU GradScaler is newer than the production project's torch==2.2.2.
    if hasattr(torch.amp, 'GradScaler'):
        return torch.amp.GradScaler('cpu')
    return torch.cuda.amp.GradScaler(enabled=False)


@pytest.mark.parametrize('n', [1, 7, 128])
def test_a1_shapes_selection_and_detach(n):
    torch.manual_seed(10)
    module = A1LocalEnhancement()
    cls = torch.randn(4, 768, requires_grad=True)
    patches = torch.randn(4, n, 768, requires_grad=True)
    z, u, details = module(cls, patches, return_details=True)
    assert z.shape == u.shape == (4, 768)
    assert details['indices'].shape == (4, 2, min(16, n))
    assert details['top_scores'].dtype == torch.float32
    details['top_scores'].retain_grad()
    (u * torch.randn_like(u)).sum().backward()
    assert cls.grad is None  # No CLS value/query-conditioning gradient shortcut.
    assert_gradient(patches)
    assert_gradient(module.patch_proj.weight)
    assert_gradient(module.relation_attn.in_proj_weight)
    if n > 1:
        assert_gradient(details['top_scores'])
        assert_gradient(module.query_tokens)
        assert_gradient(module.query_proj.weight)
    torch.testing.assert_close(z, cls + 0.05 * u, rtol=0, atol=0)


def test_a1_amp_scores_stay_fp32():
    module = A1LocalEnhancement()
    with torch.autocast('cpu', dtype=torch.bfloat16):
        z, u, details = module(torch.randn(4, 768), torch.randn(4, 23, 768), True)
    assert details['top_scores'].dtype == details['weights'].dtype == torch.float32
    assert torch.isfinite(z).all()
    z.square().mean().backward()
    assert_gradient(module.query_proj.weight)


def test_a2_initialization_gradient_and_patch_dependency():
    torch.manual_seed(11)
    module = A2FeatureModulation()
    cls = torch.randn(8, 768, requires_grad=True)
    patches = torch.randn(8, 128, 768, requires_grad=True)
    z, details = module(cls, patches, True)
    mask, cond = details['mask'], details['condition']
    assert z.shape == mask.shape == (8, 768)
    assert cond.shape == (8, 4, 192)
    assert abs(mask.mean().item() - 0.5) < 0.03
    assert 0.005 < mask.std().item() < 0.08
    scale = 1 + 0.2 * (2 * mask - 1)
    assert scale.min() >= 0.8 and scale.max() <= 1.2
    assert (z - cls).norm() / cls.norm() < 0.06
    (mask * torch.randn_like(mask)).sum().backward()
    assert cls.grad is None
    for parameter in [patches, module.query_tokens, module.vad_self_attn.in_proj_weight,
                      module.vad_cross_attn.in_proj_weight, module.vad_ffn[-1].weight,
                      module.cls_proj.weight, module.mask_token,
                      module.afm_cross_attn.in_proj_weight, module.mask_head[-1].weight]:
        assert_gradient(parameter)
    with torch.no_grad():
        different = module(cls, patches + torch.randn_like(patches), True)[1]['mask']
    assert (different - mask).abs().max() > 1e-5


def test_token_taps_are_exact_one_pass():
    model = TransReID(img_size=(32, 32), patch_size=16, stride_size=(16, 16),
                     embed_dim=48, depth=12, num_heads=4, qkv_bias=True).eval()
    observed, count = {}, [0] * 12
    handles = []
    for i, block in enumerate(model.blocks):
        def record(_, args, output, i=i):
            count[i] += 1
            if i == 10:
                observed['l11'] = output[:, 1:]
        handles.append(block.register_forward_hook(record))
    images = torch.randn(2, 3, 32, 32)
    cls, patches = model(images, return_tokens='l11')
    assert count == [1] * 12
    torch.testing.assert_close(patches, observed['l11'], rtol=0, atol=0)
    assert patches.requires_grad
    for handle in handles:
        handle.remove()
    torch.testing.assert_close(model(images), cls, rtol=0, atol=0)
    normalized = []
    handle = model.norm.register_forward_hook(lambda _, args, out: normalized.append(out))
    cls12, p12 = model(images, return_tokens='l12')
    handle.remove()
    torch.testing.assert_close(p12, normalized[0][:, 1:], rtol=0, atol=0)
    torch.testing.assert_close(cls12, cls, rtol=0, atol=0)


def test_baseline_matches_pre_edit_snapshot():
    original = ROOT.parent / 'tmp' / 'uad_original'
    if not original.exists():
        pytest.skip('Pre-edit snapshot is only available in the implementation workspace')
    spec = importlib.util.spec_from_file_location('_original_vit', original / 'model/backbones/vit_pytorch.py')
    vit = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = vit
    spec.loader.exec_module(vit)
    old = types.ModuleType('_original_model')
    source = (original / 'model/make_model.py').read_text(encoding='utf-8').replace(
        'from .backbones.vit_pytorch import', 'from _original_vit import')
    exec(compile(source, str(original / 'model/make_model.py'), 'exec'), old.__dict__)
    cfg = configuration()
    torch.manual_seed(77)
    before = old.make_model(cfg, 3, 0, 0)
    torch.manual_seed(77)
    after = make_model(cfg, 3, 0, 0)
    assert set(before.state_dict()) == set(after.state_dict())
    for key, value in before.state_dict().items():
        torch.testing.assert_close(after.state_dict()[key], value, rtol=0, atol=0)
    images = [torch.randn(2, 3, 256, 128) for _ in range(3)]
    with torch.no_grad():
        before.eval()
        after.eval()
        for neck in ('before', 'after'):
            before.neck_feat = after.neck_feat = neck
            for mode in (1, 2, 3):
                torch.testing.assert_close(before(images[0], mode=mode), after(images[0], mode=mode), rtol=0, atol=0)
        before.train()
        after.train()
        torch.manual_seed(78)
        original_outputs = before(images)
        torch.manual_seed(78)
        current_outputs = after(images)
        for key, expected in zip(('cls_score', 'global_feat', 'bn_feat'), original_outputs):
            torch.testing.assert_close(current_outputs[key], expected, rtol=0, atol=0)
    del before, after
    gc.collect()


@pytest.mark.parametrize('kind', ['none', 'a1', 'a2'])
def test_full_vit_training_losses_optimizer_and_inference(kind):
    torch.manual_seed(12)
    cfg = configuration(kind)
    model = make_model(cfg, 3, 0, 0)
    criterion, center = identity_loss(cfg)
    optimizer, _ = make_optimizer(cfg, model, center)
    optimized = [id(p) for group in optimizer.param_groups for p in group['params']]
    assert len(optimized) == len(set(optimized))
    assert set(optimized) == {id(p) for p in model.parameters() if p.requires_grad}
    imgs = [torch.randn(4, 3, 256, 128) for _ in range(3)]
    labels = torch.tensor([0, 0, 1, 1])
    output = model(imgs, labels)
    assert output['global_feat'].shape == (12, 768)
    assert (output['local_score'] is not None) == (kind == 'a1')
    assert ('feature_aug' in model._modules) == (kind != 'none')
    loss, _, _ = criterion(output['cls_score'], output['global_feat'], labels.repeat(3))
    pca, ids, centers = PCA(output['global_feat'], labels, 3, group_size=2)
    bank = PrototypeBank(3, 768, device='cpu')
    gpd = GPD(output['global_feat'], labels, bank, ids, centers, 3)
    total = loss + 0.01 * pca + gpd
    if kind == 'a1':
        local_ce = criterion.id_criterion(output['local_score'], labels.repeat(3))
        torch.testing.assert_close(local_ce, F.cross_entropy(output['local_score'], labels.repeat(3)))
        total = total + 0.05 * local_ce
    assert torch.isfinite(total)
    total.backward()
    assert_gradient(model.base.blocks[10].attn.qkv.weight)
    assert_gradient(model.base.blocks[11].attn.qkv.weight)
    if kind != 'none':
        for name, parameter in model.feature_aug.named_parameters():
            assert_gradient(parameter)
    optimizer.step()
    bank.update(ids, centers, ddp=False)
    model.eval()
    with torch.no_grad():
        expected, _ = model._extract_feature(imgs[0])
        for mode in (1, 2, 3):
            torch.testing.assert_close(model(imgs[0], mode=mode), expected, rtol=0, atol=0)
        model.neck_feat = 'after'
        torch.testing.assert_close(model(imgs[0], mode=1), model.bottleneck(expected), rtol=0, atol=0)
    del model, optimizer, output, total
    gc.collect()


def test_main_and_local_identity_criterion_smoothing():
    cfg = configuration('a1')
    cfg.MODEL.IF_LABELSMOOTH = 'on'
    criterion, _ = identity_loss(cfg)
    logits = torch.randn(12, 3)
    labels = torch.tensor([0, 0, 1, 1]).repeat(3)
    main = criterion(logits, torch.randn(12, 768), labels)[1]
    torch.testing.assert_close(main, criterion.id_criterion(logits, labels), rtol=0, atol=0)


class SmallModel(nn.Module):
    def __init__(self, aug=False):
        super().__init__()
        self.base = nn.Linear(8, 8)
        self.bottleneck = nn.BatchNorm1d(8)
        self.classifier = nn.Linear(8, 3)
        if aug:
            self.feature_aug = nn.Linear(8, 8)
        self.checkpoint_metadata = {'version': 1, 'type': 'a1' if aug else 'none'}

    def forward(self, x):
        x = self.base(x)
        if hasattr(self, 'feature_aug'):
            x = x + 0.05 * self.feature_aug(x)
        return self.classifier(self.bottleneck(x))


def test_checkpoint_migration_and_rejections(tmp_path):
    base, augmented = SmallModel(), SmallModel(True)
    path = tmp_path / 'base.pth'
    torch.save({'state_dict': {'module.' + k: v for k, v in base.state_dict().items()}}, path)
    load_model_checkpoint(augmented, path, 'uad_init')
    for key, value in base.state_dict().items():
        torch.testing.assert_close(augmented.state_dict()[key], value, rtol=0, atol=0)
    with pytest.raises(ValueError):
        load_model_checkpoint(augmented, path)
    state = base.state_dict()
    state.pop('base.weight')
    torch.save(state, path)
    with pytest.raises(ValueError):
        load_model_checkpoint(augmented, path, 'uad_init')
    state = augmented.state_dict()
    state['classifier.weight'] = torch.randn(4, 8)
    torch.save(state, path)
    with pytest.raises(ValueError):
        load_model_checkpoint(augmented, path)
    path = tmp_path / 'aug.pth'
    torch.save(augmented.state_dict(), path)
    write_model_metadata(path, augmented)
    load_model_checkpoint(augmented, path)
    augmented.checkpoint_metadata['gamma'] = 0.1
    with pytest.raises(ValueError):
        load_model_checkpoint(augmented, path)
    augmented.checkpoint_metadata.pop('gamma')
    with pytest.raises(ValueError):
        load_model_checkpoint(base, path)


def test_imagenet_interpolation_and_missing_keys(tmp_path):
    source = TransReID(img_size=(32, 32), patch_size=16, stride_size=(16, 16),
                      embed_dim=48, depth=12, num_heads=4, qkv_bias=True)
    target = TransReID(img_size=(48, 32), patch_size=16, stride_size=(16, 16),
                      embed_dim=48, depth=12, num_heads=4, qkv_bias=True)
    state = {k: v for k, v in source.state_dict().items() if not k.startswith('fc.')}
    state['head.weight'] = torch.randn(1000, 48)
    path = tmp_path / 'imagenet.pth'
    torch.save({'model': state}, path)
    target.load_param(path)
    torch.testing.assert_close(target.blocks[11].attn.qkv.weight, source.blocks[11].attn.qkv.weight)
    assert target.pos_embed.shape == (1, 7, 48)
    state.pop('blocks.11.norm1.weight')
    torch.save(state, path)
    with pytest.raises(ValueError):
        target.load_param(path)


def test_full_resume_reproduces_next_update(tmp_path):
    torch.manual_seed(123)
    random.seed(123)
    np.random.seed(123)
    cfg = configuration('a1')
    model = SmallModel(True)
    center = nn.Linear(8, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.008, momentum=0.9)
    optimizer_center = torch.optim.SGD(center.parameters(), lr=0.5)
    scheduler = create_scheduler(cfg, optimizer)
    scaler = cpu_scaler()
    bank = PrototypeBank(3, 8, device='cpu')
    loader = DataLoader(TensorDataset(torch.zeros(8, 8)), batch_size=4)

    def step():
        optimizer.zero_grad()
        x = torch.randn(4, 8) + random.random() + float(np.random.rand())
        loss = F.cross_entropy(model(x), torch.tensor([0, 0, 1, 1]))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        bank.update(torch.tensor([0, 1]), torch.randn(2, 8), ddp=False)
        return loss.detach().clone()

    scheduler.step(1)
    step()
    path = tmp_path / 'resume.pth'
    save_training_checkpoint(path, model, cfg, 1, optimizer, optimizer_center,
                             scheduler, scaler, bank, center, loader)
    scheduler.step(2)
    expected_loss = step()
    expected_state = copy.deepcopy(model.state_dict())
    expected_bank = bank.prototypes.clone()
    expected_rng = (random.random(), np.random.rand(), torch.rand(1))
    assert restore_training_checkpoint(path, model, cfg, optimizer, optimizer_center,
                                       scheduler, scaler, bank, center, loader) == 1
    scheduler.step(2)
    torch.testing.assert_close(step(), expected_loss, rtol=0, atol=0)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, expected_state[key], rtol=0, atol=0)
    torch.testing.assert_close(bank.prototypes, expected_bank, rtol=0, atol=0)
    assert random.random() == expected_rng[0]
    assert np.random.rand() == expected_rng[1]
    torch.testing.assert_close(torch.rand(1), expected_rng[2], rtol=0, atol=0)
    cfg.MODEL.FEATURE_AUG.GAMMA = 0.1
    with pytest.raises(ValueError):
        restore_training_checkpoint(path, model, cfg, optimizer, optimizer_center,
                                    scheduler, scaler, bank, center, loader)
