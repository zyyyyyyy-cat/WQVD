"""Synthetic protocol, ranking, cache, and forward compatibility checks."""
import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.a1_diagnostics import (evaluate, query_metrics, local_statistics,
                                  compare_queries, transition_summary, protocols,
                                  FusionFeatures, write_csv)
from utils.metrics import compute_indices_chunked, eval_func
from model.make_model import build_transformer
from config import cfg as defaults

spec = importlib.util.spec_from_file_location('analyze_a1', ROOT / 'tools/analyze_a1_vs_uad.py')
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)
torch.set_num_threads(2)


def sample(i, pid, cam, mid=1, split='gallery'):
    modality = {1: 'RGB', 2: 'NIR', 3: 'TIR'}[mid]
    platform = 'G' if cam < 5 else 'A'
    return dict(sample_index=i, pid=pid, camera_id=cam, modality_id=mid,
                split=split, modality=modality, platform=platform,
                domain=modality + '-' + platform, path='image_' + str(i))


def test_same_camera_negative_exclusion_and_invalid_query():
    # Nearest negative is same camera: it MUST be removed, unlike Market-1501.
    gpids, gcams = np.array([2, 1, 3, 1]), np.array([0, 1, 2, 0])
    distances = np.array([0.01, 0.1, 0.2, 0.3])
    r = query_metrics(distances, np.arange(4), 1, 0, gpids, gcams, np.arange(4) + 10)
    assert r['rank1_correct'] == 1 and r['AP'] == r['INP'] == 1
    assert r['valid_positive_count'] == 1
    assert r['nearest_negative_index'] == 12
    assert r['margin'] == pytest.approx(0.1)
    invalid = query_metrics(distances, np.arange(4), 99, 0, gpids, gcams, np.arange(4))
    assert not invalid['valid'] and invalid['AP'] is None
    assert set(invalid) == set(r)


def test_official_metrics_parity_and_chunk_invariance():
    rng = np.random.default_rng(42)
    features = rng.normal(size=(18, 8)).astype('float32')
    meta = [sample(i, i % 3, 0, split='query') for i in range(6)]
    meta += [sample(i, i % 3, 1 + i % 2) for i in range(6, 18)]
    qi, gi = np.arange(6), np.arange(6, 18)
    records, summary = evaluate(features, meta, qi, gi, 2)
    records2, summary2 = evaluate(features, meta, qi, gi, 4)
    assert summary == summary2
    assert [r['first_pos_rank'] for r in records.values()] == [r['first_pos_rank'] for r in records2.values()]
    f = torch.nn.functional.normalize(torch.from_numpy(features), dim=1)
    order = compute_indices_chunked(f[qi], f[gi], q_chunk_size=2)
    cmc, ap, inp = eval_func(order, np.array([m['pid'] for m in meta[:6]]),
                           np.array([m['pid'] for m in meta[6:]]),
                           np.zeros(6), np.array([m['camera_id'] for m in meta[6:]]),
                           max_rank=10, return_minp=True)
    assert summary['mAP'] == pytest.approx(ap)
    assert summary['mINP'] == pytest.approx(inp)
    assert summary['rank1'] == pytest.approx(cmc[0])
    assert summary['rank5'] == pytest.approx(cmc[4])
    assert summary['valid_positive_count'] == 24


def test_rank1_down_ap_up_and_missing_negative():
    pids, cams = np.array([1, 1, 2, 2, 2]), np.ones(5)
    before_d = np.array([0.1, 0.9, 0.2, 0.3, 0.4])
    after_d = np.array([0.2, 0.3, 0.1, 0.4, 0.5])
    a = query_metrics(before_d, np.argsort(before_d), 1, 0, pids, cams, np.arange(5))
    b = query_metrics(after_d, np.argsort(after_d), 1, 0, pids, cams, np.arange(5))
    assert a['AP'] == pytest.approx(0.7) and b['AP'] == pytest.approx(7 / 12)
    # More positives allow overall AP to rise even if rank1 falls.
    pids = np.array([1, 1, 1, 2, 2, 2, 2, 2])
    a = query_metrics(np.arange(8), np.array([0, 3, 4, 5, 6, 7, 1, 2]), 1, 0, pids, np.ones(8), np.arange(8))
    b = query_metrics(np.arange(8), np.array([3, 0, 1, 2, 4, 5, 6, 7]), 1, 0, pids, np.ones(8), np.arange(8))
    meta = [sample(0, 1, 0, split='query')]
    local = [dict(rho=.05, cos_cls_local=.2, direction_delta=.03,
                  topk_overlap=.5, topk_entropy=1., topk_stability=None)]
    rows = compare_queries('TIR->TIR', {'UAD-CLS': {0: a}, 'A1-CLS': {0: a}, 'A1-Fusion': {0: b}}, meta, local)
    assert rows[0]['fusion_transition'] == '1->0'
    assert rows[0]['fusion_rank1_down_AP_up']
    transition = [r for r in transition_summary('TIR->TIR', rows)
                  if r['comparison'] == 'fusion' and r['query_group'] == 'ALL' and r['transition'] == '1->0'][0]
    assert transition['rank1_down_AP_up_count'] == 1
    only_pos = query_metrics(np.array([.1]), np.array([0]), 1, 0, np.array([1]), np.array([1]), np.array([0]))
    assert only_pos['d_neg_min'] is None and only_pos['margin'] is None


def test_residual_direction_overlap_and_entropy():
    cls = np.array([[1., 0.], [1., 0.]], dtype='float32')
    local = np.array([[2., 0.], [0., 2.]], dtype='float32')
    idx = np.array([[[0, 1], [1, 2]], [[0, 1], [0, 1]]])
    weights = np.full((2, 2, 2), .5, dtype='float32')
    rows = local_statistics(cls, local, idx, weights, .05)
    assert rows[0]['rho'] == pytest.approx(.1)
    assert rows[0]['direction_delta'] == 0
    assert rows[1]['direction_delta'] > 0
    assert rows[0]['topk_overlap'] == .5 and rows[1]['topk_overlap'] == 1
    assert rows[0]['topk_entropy_normalized'] == pytest.approx(1)
    np.testing.assert_allclose(FusionFeatures(cls, local, .05)[[1]], (cls + .05 * local)[[1]])


class FakeBackbone(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.proj = nn.Linear(4, 768)

    def forward(self, x, return_tokens=None):
        cls = self.proj(x)
        return (cls, cls[:, None].expand(-1, 20, -1)) if return_tokens else cls


def test_diagnostic_interface_preserves_forward_state():
    cfg = defaults.clone()
    cfg.merge_from_file(str(ROOT / 'configs/UAD_A1.yml'))
    cfg.MODEL.PRETRAIN_CHOICE = 'none'
    model = build_transformer(3, 0, 0, cfg, {'vit_base_in': FakeBackbone}).eval()
    x = torch.randn(2, 4)
    saved = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        output = model.diagnostic_features(x)
        torch.testing.assert_close(output['fusion'], model(x, mode=1), rtol=0, atol=0)
        torch.testing.assert_close(output['fusion'], output['cls'] + .05 * output['local'], rtol=0, atol=0)
    for k, v in saved.items():
        torch.testing.assert_close(model.state_dict()[k], v, rtol=0, atol=0)
    model.train()
    with pytest.raises(RuntimeError):
        model.diagnostic_features(x)


def test_offline_cache_reports_and_integrity(tmp_path):
    cache = tmp_path / 'features'
    cache.mkdir()
    meta = [sample(0, 99, 0, split='query'), sample(1, 1, 0, split='query'),
            sample(2, 1, 1), sample(3, 2, 1)]
    write_csv(cache / 'samples.csv', meta)
    for name in ('uad_cls', 'a1_cls', 'a1_local'):
        np.save(cache / (name + '.npy'), np.random.default_rng(3).normal(size=(4, 8)).astype('float32'))
    np.save(cache / 'indices.npy', np.tile(np.array([[[0, 1], [1, 2]]]), (4, 1, 1)))
    np.save(cache / 'weights.npy', np.full((4, 2, 2), .5, dtype='float32'))
    audit = dict(format_version=1, count=4, num_query=2, gamma=.05,
                 stability_brightness_factor=None,
                 samples_sha256=tool.fingerprint(cache / 'samples.csv')['sha256'])
    (cache / 'manifest.json').write_text(json.dumps(audit), encoding='utf-8')
    args = argparse.Namespace(gammas=[0, .025, .05, .1], detailed_domains=True, chunk_size=2)
    tool.analyze(args, cache, tmp_path / 'reports')
    for name in ('feature_comparison', 'query_comparison', 'rank1_transitions', 'a1_local_diagnostics', 'a1_local_summary'):
        with open(tmp_path / 'reports' / (name + '.csv'), encoding='utf-8-sig') as stream:
            assert list(csv.DictReader(stream))
    with open(tmp_path / 'reports/query_comparison.csv', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]['UAD-CLS_valid'] == 'False'  # Invalid first query must not break CSV schema.
    audit['samples_sha256'] = 'tampered'
    (cache / 'manifest.json').write_text(json.dumps(audit), encoding='utf-8')
    with pytest.raises(ValueError, match='modified sample'):
        tool.load_cache(cache)


def test_export_shared_loader_query_gallery_order_and_stability(tmp_path):
    """Exercise export->reload->analyze with real branch weights, mocked images."""
    configs = []
    for kind in ('none', 'a1'):
        cfg = defaults.clone()
        cfg.merge_from_file(str(ROOT / 'configs' / ('UAD.yml' if kind == 'none' else 'UAD_A1.yml')))
        cfg.MODEL.PRETRAIN_CHOICE = 'none'
        configs.append(cfg)
    def make(cfg, num_class, camera_num, view_num):
        return build_transformer(num_class, camera_num, view_num, cfg, {'vit_base_in': FakeBackbone})
    paths = []
    for i, cfg in enumerate(configs):
        model = make(cfg, 3, 0, 0)
        path = tmp_path / ('checkpoint_' + str(i) + '.pth')
        torch.save({'model': model.state_dict(), 'metadata': model.checkpoint_metadata}, path)
        paths.append(str(path))
    # Input needs image channels for the brightness probe; adapt fake backbone.
    def forward(self, x, return_tokens=None):
        cls = self.proj(x.flatten(1)[:, :4])
        return (cls, cls[:, None].expand(-1, 20, -1)) if return_tokens else cls
    class Loader:
        dataset = list(range(3))
        def __init__(self, mid):
            self.mid = mid
        def __iter__(self):
            images = torch.arange(36).reshape(3, 3, 2, 2).float() / 50
            yield images[:2], (1, 1), (0, 1), None, (self.mid, self.mid), ('q' + str(self.mid), 'g1' + str(self.mid))
            yield images[2:], (2,), (1,), None, (self.mid,), ('g2' + str(self.mid),)
    args = argparse.Namespace(uad_config=str(ROOT / 'configs/UAD.yml'),
        a1_config=str(ROOT / 'configs/UAD_A1.yml'), opts=[], data_root=None,
        batch_size=2, workers=0, device='cpu', uad_weight=paths[0], a1_weight=paths[1],
        stability=True, brightness_factor=1.05, gammas=[], detailed_domains=False, chunk_size=2)
    cache = tmp_path / 'cache'
    with patch('datasets.make_dataloader', return_value=(None, None, [Loader(m) for m in (1, 2, 3)], [1, 1, 1], 3, 0, 0)), \
         patch('model.make_model', side_effect=make), patch.object(FakeBackbone, 'forward', forward):
        tool.export(args, cache)
    audit, meta, arrays = tool.load_cache(cache)
    assert [m['path'] for m in meta] == ['q1', 'q2', 'q3', 'g11', 'g21', 'g12', 'g22', 'g13', 'g23']
    assert [m['modality_id'] for m in meta] == [1, 2, 3, 1, 1, 2, 2, 3, 3]
    assert len(np.load(cache / 'stability.npy')) == 9
    assert audit['uad']['sha256'] != audit['a1']['sha256']
    assert audit['uad']['training_details_status'] == 'unknown_model_only_checkpoint'
    np.testing.assert_allclose(arrays['a1_cls'][0], arrays['a1_cls'][1])
    tool.analyze(args, cache, tmp_path / 'results')
    with pytest.raises(ValueError, match='empty cache'):
        tool.export(args, cache)
