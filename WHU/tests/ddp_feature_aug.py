"""Two-rank CPU/Gloo acceptance; CUDA/NCCL training entry point is unchanged."""
import argparse
import gc
import hashlib
import json
import random
import socket
import sys
import tempfile
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_feature_aug import (configuration, identity_loss, SmallModel, assert_gradient,
                              cpu_scaler,
                              make_model, make_optimizer, create_scheduler,
                              PCA, GPD, PrototypeBank, save_training_checkpoint,
                              restore_training_checkpoint)


def worker(rank, world, port, output_dir):
    torch.set_num_threads(2)
    dist.init_process_group('gloo', init_method='tcp://127.0.0.1:{}'.format(port),
                            rank=rank, world_size=world, timeout=timedelta(seconds=120))
    results = {}
    try:
        for kind in ('none', 'a1', 'a2'):
            torch.manual_seed(200 + rank)
            cfg = configuration(kind)
            cfg.INPUT.SIZE_TRAIN = [32, 32]  # Actual 12-block/768-dim backbone, small images.
            model = make_model(cfg, 3, 0, 0)
            ddp = DistributedDataParallel(model, find_unused_parameters=True)
            criterion, center = identity_loss(cfg)
            optimizer, _ = make_optimizer(cfg, model, center)
            bank = PrototypeBank(3, 768, device='cpu')
            labels = torch.tensor([0, 0, 1, 1])
            losses = []
            for step in range(2):
                optimizer.zero_grad()
                imgs = [torch.randn(4, 3, 32, 32) for _ in range(3)]
                out = ddp(imgs)
                loss = criterion(out['cls_score'], out['global_feat'], labels.repeat(3))[0]
                pca, ids, centers = PCA(out['global_feat'], labels, 3, 2)
                loss = loss + 0.01 * pca + GPD(out['global_feat'], labels, bank, ids, centers, 3)
                if kind == 'a1':
                    loss = loss + 0.05 * criterion.id_criterion(out['local_score'], labels.repeat(3))
                loss.backward()
                if kind != 'none':
                    for parameter in model.feature_aug.parameters():
                        assert_gradient(parameter)
                optimizer.step()
                bank.update(ids, centers, ddp=True)
                losses.append(loss.item())
            digest = hashlib.sha256()
            for parameter in model.parameters():
                digest.update(parameter.detach().numpy().tobytes())
            hashes = [None] * world
            dist.all_gather_object(hashes, digest.hexdigest())
            assert len(set(hashes)) == 1, 'DDP parameters diverged'
            banks = [torch.empty_like(bank.prototypes) for _ in range(world)]
            dist.all_gather(banks, bank.prototypes)
            for value in banks:
                torch.testing.assert_close(value, bank.prototypes, rtol=0, atol=0)
            # Unequal inference counts through the unwrapped module, as in processor.
            model.eval()
            for _ in range(rank + 1):
                with torch.no_grad():
                    assert model(imgs[0], mode=1).shape == (4, 768)
            dist.barrier()
            results[kind] = {'losses': losses, 'parameters_synchronized': True,
                             'prototypes_synchronized': True, 'uneven_eval': True}
            del ddp, model, optimizer, out, loss
            gc.collect()

        cfg = configuration('a1')
        cfg.MODEL.DIST_TRAIN = True
        small = SmallModel(True)
        wrapped = DistributedDataParallel(small)
        center = torch.nn.Linear(8, 3)
        optimizer = torch.optim.SGD(small.parameters(), lr=0.008, momentum=0.9)
        center_optimizer = torch.optim.SGD(center.parameters(), lr=0.5)
        scheduler = create_scheduler(cfg, optimizer)
        scaler = cpu_scaler()
        bank = PrototypeBank(3, 8, device='cpu')
        loader = DataLoader(TensorDataset(torch.zeros(8, 8)), batch_size=4)

        def update():
            optimizer.zero_grad()
            loss = F.cross_entropy(wrapped(torch.randn(4, 8)), torch.tensor([0, 0, 1, 1]))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            bank.update(torch.tensor([0, 1]), torch.randn(2, 8), ddp=True)
            return loss.detach().clone()

        update()
        small.bottleneck.running_mean.fill_(rank + 0.25)
        saved_buffer = small.bottleneck.running_mean.clone()
        random.seed(400 + rank)
        np.random.seed(400 + rank)
        path = Path(output_dir) / 'ddp_resume.pth'
        save_training_checkpoint(path, small, cfg, 1, optimizer, center_optimizer,
                                 scheduler, scaler, bank, center, loader)
        expected_loss = update()
        expected = {k: v.clone() for k, v in small.state_dict().items()}
        expected_rng = (random.random(), float(np.random.rand()), torch.rand(1))
        restore_training_checkpoint(path, small, cfg, optimizer, center_optimizer,
                                    scheduler, scaler, bank, center, loader)
        torch.testing.assert_close(small.bottleneck.running_mean, saved_buffer, rtol=0, atol=0)
        torch.testing.assert_close(update(), expected_loss, rtol=0, atol=0)
        for k, v in small.state_dict().items():
            torch.testing.assert_close(v, expected[k], rtol=0, atol=0)
        assert random.random() == expected_rng[0]
        assert float(np.random.rand()) == expected_rng[1]
        torch.testing.assert_close(torch.rand(1), expected_rng[2], rtol=0, atol=0)
        results['resume'] = {'next_update_exact': True, 'rank_rng_exact': True,
                             'rank_buffers_exact': True}
        (Path(output_dir) / 'rank_{}.json'.format(rank)).write_text(json.dumps(results, indent=2))
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    with tempfile.TemporaryDirectory() as temp:
        output_dir = args.output or temp
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        mp.spawn(worker, args=(2, port, output_dir), nprocs=2, join=True)
        print('PASS: 2-rank DDP Baseline/A1/A2, prototype synchronization, uneven eval, full resume')
