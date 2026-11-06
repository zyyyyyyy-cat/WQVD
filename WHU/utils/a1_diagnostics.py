"""Offline paired retrieval diagnostics with the existing WHU evaluator."""
import csv
import itertools

import numpy as np
import torch
from torch.nn import functional as F

from utils.metrics import (MODALITY_NAMES, eval_func, iter_euclidean_distance_chunks,
                           _domain_protocol_definitions, _platform_mask)


class FusionFeatures:
    """Compute fusion only for requested rows; do not duplicate the whole cache."""
    def __init__(self, cls, local, gamma):
        self.cls, self.local, self.gamma = cls, local, gamma

    def __getitem__(self, index):
        return self.cls[index] + self.gamma * self.local[index]


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError('No rows to write: ' + str(path))
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, 'w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def protocols(meta, detailed=False):
    q = np.array([i for i, m in enumerate(meta) if m['split'] == 'query'])
    g = np.array([i for i, m in enumerate(meta) if m['split'] == 'gallery'])
    mids = np.array([m['modality_id'] for m in meta])
    cams = np.array([m['camera_id'] for m in meta])
    yield 'overall', q, g
    for qm, gm in itertools.product(sorted(set(mids[q])), sorted(set(mids[g]))):
        yield MODALITY_NAMES[qm] + '->' + MODALITY_NAMES[gm], q[mids[q] == qm], g[mids[g] == gm]
    for mid in sorted(set(mids[q])):
        yield MODALITY_NAMES[mid] + '->ALL', q[mids[q] == mid], g
    for qp, gp in [('A', 'G'), ('G', 'A')]:
        yield qp + '->' + gp, q[_platform_mask(cams[q], qp)], g[_platform_mask(cams[g], gp)]
    if detailed:
        for _, qn, gn, qm, qp, gm, gp in _domain_protocol_definitions():
            yield qn + '->' + gn, q[(mids[q] == qm) & _platform_mask(cams[q], qp)], g[(mids[g] == gm) & _platform_mask(cams[g], gp)]


def query_metrics(distances, order, pid, camera, gpids, gcams, gallery_ids):
    """All same-camera gallery items removed, including negative identities."""
    kept = order[gcams[order] != camera]
    positives = np.flatnonzero(gpids[kept] == pid)
    base = {'valid': bool(len(positives)), 'valid_positive_count': len(positives),
            'valid_gallery_count': len(kept), **dict.fromkeys((
                'rank1_correct', 'AP', 'INP', 'first_pos_rank', 'last_pos_rank',
                'mean_pos_rank', 'd_pos_min', 'd_pos_mean', 'd_pos_max',
                'd_neg_min', 'margin', 'distance_tie', 'nearest_positive_index',
                'nearest_negative_index', 'top1_gallery_index'))}
    if not len(positives):
        return {**base, 'status': 'no_positive_after_same_camera_exclusion'}
    # Reuse official AP/INP and same-camera exclusion, rather than Market-1501.
    cmc, ap, inp = eval_func(order[None, :], np.array([pid]), gpids,
                            np.array([camera]), gcams, max_rank=1, return_minp=True)
    pos = kept[gpids[kept] == pid]
    neg = kept[gpids[kept] != pid]
    dpos = float(distances[pos].min())
    dneg = float(distances[neg].min()) if len(neg) else None
    return {**base, 'status': 'ok', 'rank1_correct': int(cmc[0]), 'AP': float(ap),
            'INP': float(inp), 'first_pos_rank': int(positives[0] + 1),
            'last_pos_rank': int(positives[-1] + 1),
            'mean_pos_rank': float((positives + 1).mean()),
            'd_pos_min': dpos, 'd_pos_mean': float(distances[pos].mean()),
            'd_pos_max': float(distances[pos].max()), 'd_neg_min': dneg,
            'margin': dneg - dpos if dneg is not None else None,
            'distance_tie': bool(dneg is not None and dpos == dneg),
            'nearest_positive_index': int(gallery_ids[pos[0]]),
            'nearest_negative_index': int(gallery_ids[neg[0]]) if len(neg) else None,
            'top1_gallery_index': int(gallery_ids[kept[0]])}


def evaluate(features, meta, qi, gi, chunk_size=64):
    if chunk_size <= 0:
        raise ValueError('chunk_size must be positive')
    if not len(qi) or not len(gi):
        return {}, {'status': 'empty_query_or_gallery', 'valid_query_count': 0,
                    'valid_positive_count': 0}
    gm = [meta[i] for i in gi]
    gpids = np.array([m['pid'] for m in gm])
    gcams = np.array([m['camera_id'] for m in gm])
    # Exactly the float32 L2 normalization and squared Euclidean distance in WHU.
    qf = F.normalize(torch.as_tensor(np.asarray(features[qi]).copy()), dim=1)
    gf = F.normalize(torch.as_tensor(np.asarray(features[gi]).copy()), dim=1)
    if not torch.isfinite(qf).all() or not torch.isfinite(gf).all():
        raise ValueError('Features contain non-finite values')
    records = {}
    offset = 0
    for chunk in iter_euclidean_distance_chunks(qf, gf, chunk_size):
        # Match compute_indices_chunked, including its tie ordering.
        orders = torch.argsort(chunk, dim=1).numpy()
        distances = chunk.numpy()
        for row, order in zip(distances, orders):
            idx = int(qi[offset])
            records[idx] = query_metrics(row, order, meta[idx]['pid'],
                                         meta[idx]['camera_id'], gpids, gcams, gi)
            offset += 1
    valid = [r for r in records.values() if r['valid']]
    summary = {'status': 'ok' if valid else 'no_valid_query',
               'valid_query_count': len(valid),
               'valid_positive_count': sum(r['valid_positive_count'] for r in valid)}
    if valid:
        summary.update(mAP=float(np.mean([r['AP'] for r in valid])),
                       mINP=float(np.mean([r['INP'] for r in valid])))
        for k in (1, 5, 10):
            summary['rank' + str(k)] = float(np.mean([r['first_pos_rank'] <= k for r in valid]))
    return records, summary


def local_statistics(cls, local, indices, weights, gamma, stability=None):
    c, u = torch.as_tensor(cls.copy()), torch.as_tensor(local.copy())
    z = c + gamma * u
    cn, un = c.norm(dim=-1), u.norm(dim=-1)
    rho = gamma * un / cn.clamp_min(1e-12)
    cosine = F.cosine_similarity(c, u, dim=-1)
    delta = (F.normalize(z, dim=-1) - F.normalize(c, dim=-1)).norm(dim=-1)
    pairs = list(itertools.combinations(range(indices.shape[1]), 2))
    rows = []
    for i in range(len(c)):
        overlap = [len(set(indices[i, a]) & set(indices[i, b])) / indices.shape[-1]
                   for a, b in pairs]
        entropy = -(weights[i] * np.log(np.maximum(weights[i], 1e-30))).sum(-1)
        rows.append({'cls_norm': float(cn[i]), 'local_norm': float(un[i]),
                     'rho': float(rho[i]), 'cos_cls_local': float(cosine[i]),
                     'direction_delta': float(delta[i]),
                     'topk_overlap': float(np.mean(overlap)) if pairs else None,
                     'topk_entropy': float(entropy.mean()),
                     'topk_entropy_normalized': float(entropy.mean() / np.log(indices.shape[-1]))
                     if indices.shape[-1] > 1 else 0.0,
                     'topk_stability': float(stability[i]) if stability is not None else None})
    return rows


COMPARISONS = [('train', 'UAD-CLS', 'A1-CLS'),
               ('fusion', 'A1-CLS', 'A1-Fusion'),
               ('total', 'UAD-CLS', 'A1-Fusion')]
DELTA_FIELDS = ('AP', 'INP', 'first_pos_rank', 'last_pos_rank', 'mean_pos_rank',
                'd_pos_min', 'd_pos_mean', 'd_neg_min', 'margin')


def compare_queries(protocol, results, meta, local_rows):
    rows = []
    for idx in results['UAD-CLS']:
        row = {'protocol': protocol, **meta[idx], **local_rows[idx]}
        for name, records in results.items():
            row.update({name + '_' + key: value for key, value in records[idx].items()})
        for label, before, after in COMPARISONS:
            a, b = results[before][idx], results[after][idx]
            row.update({label + '_delta_' + field: None for field in DELTA_FIELDS})
            row[label + '_rank1_down_AP_up'] = False
            if a['valid'] != b['valid']:
                raise ValueError('Feature variants have different valid query sets')
            if not a['valid']:
                row[label + '_transition'] = 'invalid'
                continue
            row[label + '_transition'] = '{}->{}'.format(a['rank1_correct'], b['rank1_correct'])
            for field in DELTA_FIELDS:
                row[label + '_delta_' + field] = (b[field] - a[field]
                    if a[field] is not None and b[field] is not None else None)
            row[label + '_rank1_down_AP_up'] = bool(a['rank1_correct'] and not b['rank1_correct'] and b['AP'] > a['AP'])
        rows.append(row)
    return rows


def transition_summary(protocol, rows):
    output = []
    groups = [('ALL', rows)]
    for field in ('modality', 'platform', 'domain'):
        for value in sorted(set(r[field] for r in rows)):
            groups.append((field + ':' + value, [r for r in rows if r[field] == value]))
    for label, _, _ in COMPARISONS:
        for group, members in groups:
            valid = [r for r in members if r[label + '_transition'] != 'invalid']
            for transition in ('0->1', '1->0', '1->1', '0->0'):
                subset = [r for r in valid if r[label + '_transition'] == transition]
                result = {'protocol': protocol, 'comparison': label, 'query_group': group,
                          'transition': transition, 'count': len(subset),
                          'valid_query_count': len(valid), 'invalid_query_count': len(members) - len(valid),
                          'fraction': len(subset) / len(valid) if valid else None,
                          'rank1_down_AP_up_count': sum(r[label + '_rank1_down_AP_up'] for r in subset)}
                for field in DELTA_FIELDS:
                    values = [r[label + '_delta_' + field] for r in subset
                              if r[label + '_delta_' + field] is not None]
                    result['mean_delta_' + field] = float(np.mean(values)) if values else None
                for field in ('rho', 'cos_cls_local', 'direction_delta', 'topk_overlap', 'topk_entropy', 'topk_stability'):
                    values = [r[field] for r in subset if r[field] is not None]
                    result['mean_' + field] = float(np.mean(values)) if values else None
                output.append(result)
    return output
