"""Synthetic branch stability probe; explicitly not a WHU-MARS accuracy experiment."""
import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.modules import A1LocalEnhancement, A2FeatureModulation
from processor.processor import PCA, GPD, PrototypeBank
from loss.wrt_loss import TripletLoss_WRT


def probe(kind, steps, device):
    torch.manual_seed(1234)
    module = (A1LocalEnhancement() if kind == 'a1' else A2FeatureModulation()).to(device)
    neck = nn.BatchNorm1d(768).to(device)
    head = nn.Linear(768, 3, bias=False).to(device)
    local_neck = nn.BatchNorm1d(768).to(device) if kind == 'a1' else None
    local_head = nn.Linear(768, 3, bias=False).to(device) if kind == 'a1' else None
    parameters = list(module.parameters()) + list(neck.parameters()) + list(head.parameters())
    if kind == 'a1':
        parameters += list(local_neck.parameters()) + list(local_head.parameters())
    optimizer = torch.optim.SGD(parameters, lr=0.008, momentum=0.9, weight_decay=1e-4)
    bank = PrototypeBank(3, 768, device=device)
    target = torch.tensor([0, 0, 1, 1], device=device)
    target_rep = target.repeat(3)
    wrt = TripletLoss_WRT()
    records = []
    for step in range(steps):
        optimizer.zero_grad()
        # Random tokens simulate original LN12 CLS/P12; raw P11 for A1.
        cls = F.layer_norm(torch.randn(12, 768, device=device), (768,))
        patches = torch.randn(12, 128, 768, device=device)
        if kind == 'a2':
            patches = F.layer_norm(patches, (768,))
            z, details = module(cls, patches, True)
        else:
            z, local, details = module(cls, patches, True)
        loss = F.cross_entropy(head(neck(z)), target_rep) + wrt(z, target_rep)[0]
        pca, ids, centers = PCA(z, target, 3, group_size=2)
        loss = loss + 0.01 * pca + GPD(z, target, bank, ids, centers, 3)
        if kind == 'a1':
            loss = loss + 0.05 * F.cross_entropy(local_head(local_neck(local)), target_rep)
        loss.backward()
        gradient_norms = {n: p.grad.norm().item() for n, p in module.named_parameters()}
        assert all(torch.isfinite(p.grad).all() for p in module.parameters())
        assert all(v > 0 for v in gradient_norms.values())
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        bank.update(ids, centers, ddp=False)
        assert torch.isfinite(loss)
        record = {'step': step + 1, 'loss': loss.item(), 'cls_norm': cls.norm(dim=1).mean().item(),
                  'z_norm': z.norm(dim=1).mean().item(),
                  'relative_perturbation': ((z - cls).norm() / cls.norm()).item(),
                  'gradient_norms': gradient_norms}
        if kind == 'a2':
            mask = details['mask'].detach()
            record.update(mask_mean=mask.mean().item(), mask_std=mask.std().item(),
                          mask_min=mask.min().item(), mask_max=mask.max().item())
        if step in (0, steps - 1) or (step + 1) % 10 == 0:
            records.append(record)
            print(kind, {k: round(v, 6) if isinstance(v, float) else v
                         for k, v in record.items() if k != 'gradient_norms'}, flush=True)
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', default='logs/feature_aug_synthetic_check.json')
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error('--steps must be positive')
    torch.set_num_threads(2)
    result = {kind: probe(kind, args.steps, args.device) for kind in ('a1', 'a2')}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'synthetic_only': True, 'device': args.device,
                               'steps_per_branch': args.steps, 'results': result}, indent=2))
    print('PASS: finite losses/features and nonzero finite gradients in all branch parameters')
