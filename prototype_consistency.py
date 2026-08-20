"""Prototype consistency analysis for PIP-Net on GMDB.

For the frequent-test patients of ONE syndrome (--syndrome, default Cornelia de Lange), we:
  - classify each with the head and report how many are correctly predicted (top-1), and
  - build a per-image DECISION signature = top-k prototypes by contribution g_d * relu(W[c,d])
    to the true syndrome c, then report which prototypes OVERLAP across patients (appear in
    more than one patient's top-k), plus the mean pairwise Jaccard / raw overlap and a random
    null (expected overlap of random k-subsets of P prototypes).

Ground truth = the image's parent folder name (matches the classifier class order).

Run: python prototype_consistency.py --model_dir <run_dir>
"""
import os
import sys
import csv
import argparse
from collections import defaultdict, Counter
from itertools import combinations

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gmdb_common import (
    load_pipnet, build_transform, build_image_index, resolve_official,
    DEFAULT_FREQ_ROOT, DEFAULT_META_DIR, DEFAULT_MODEL_DIR, CSV_FREQ_TEST)
from util.data import load_landmarks, landmark_path_for_image, NUM_LANDMARKS


@torch.no_grad()
def presence_vectors(net, records, transform, device, masked, lm_dir, lm_size, batch, clamp):
    """{path: prototype-presence vector g (P,)} for each unique image."""
    paths = sorted({r['path'] for r in records})
    out = {}
    for i in tqdm(range(0, len(paths), batch), desc='encode %d imgs' % len(paths), ncols=0):
        chunk = paths[i:i + batch]
        imgs = torch.stack([transform(Image.open(p).convert('RGB')) for p in chunk]).to(device)
        lms = None
        if masked:
            lms = torch.stack([torch.from_numpy(load_landmarks(
                landmark_path_for_image(p, lm_dir), lm_size, NUM_LANDMARKS)) for p in chunk]).to(device)
        _, pooled, _ = net(imgs, inference=clamp, lms=lms)
        for p, g in zip(chunk, pooled.float().cpu()):
            out[p] = g
    return out


def topk_set(scores, k):
    """Indices of the top-k POSITIVE scores (set); fewer than k if fewer are positive."""
    order = torch.argsort(scores, descending=True)
    return {int(d) for d in order[:k] if scores[int(d)] > 0}


def jaccard(a, b):
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def random_null(k, P, n=20000, seed=0):
    """Expected raw overlap and Jaccard of two random k-subsets of P prototypes."""
    rng = np.random.default_rng(seed)
    ov, ja = [], []
    for _ in range(n):
        a = set(rng.choice(P, k, replace=False).tolist())
        b = set(rng.choice(P, k, replace=False).tolist())
        inter = len(a & b)
        ov.append(inter)
        ja.append(inter / len(a | b))
    return float(np.mean(ov)), float(np.mean(ja))


def syndrome_stats(records, gmap, W, c, k, bias):
    """For one syndrome's records return (n, n_correct, mean_pairwise_jaccard, mean_pairwise_overlap).
    Decision signature per patient = top-k prototypes by contribution g_d * relu(W[c,d]) to the true
    class c; overlap is averaged over all within-syndrome patient pairs."""
    n_correct, dec_sets = 0, []
    for r in records:
        g = gmap[r['path']]
        logits = W @ g + (bias if bias is not None else 0.)
        n_correct += int(torch.argmax(logits)) == c
        dec_sets.append(topk_set(g * W[c], k))
    js = [jaccard(a, b) for a, b in combinations(dec_sets, 2)]
    ov = [len(a & b) for a, b in combinations(dec_sets, 2)]
    m = lambda x: float(np.mean(x)) if x else float('nan')
    return len(records), n_correct, m(js), m(ov)


def main():
    p = argparse.ArgumentParser(description='Decision-prototype consistency across patients of a '
                                            'syndrome: one syndrome (--syndrome) or the top-N '
                                            'frequent syndromes by image count (--top_syndromes).')
    p.add_argument('--model_dir', default=DEFAULT_MODEL_DIR)
    p.add_argument('--checkpoint', default='net_trained')
    p.add_argument('--freq_root', default=DEFAULT_FREQ_ROOT)
    p.add_argument('--meta_dir', default=DEFAULT_META_DIR)
    p.add_argument('--train_dir', default=None, help='Default: <freq_root>/gmdb_train (class order).')
    p.add_argument('--landmarks_dir', default=None, help='Default: the run\'s training landmarks_dir.')
    p.add_argument('--output_dir', default=None, help='Default: <model_dir>/prototype_consistency.')
    p.add_argument('--topk', type=int, default=10, help='Prototypes per image signature.')
    p.add_argument('--syndrome', default='Cornelia de Lange syndrome',
                   help='Syndrome (frequent-test folder name) to analyze (single-syndrome mode).')
    p.add_argument('--top_syndromes', type=int, default=0,
                   help='If >0, analyze the top-N frequent-test syndromes by image count and print '
                        'AVERAGE accuracy and AVERAGE decision-prototype overlap (overrides --syndrome).')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--no_clamp', action='store_true', help='Do not clamp presence (<0.1 -> 0).')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    device = torch.device(args.device)
    net, margs = load_pipnet(args.model_dir, args.checkpoint, device)
    transform = build_transform(margs.net, margs.image_size)
    masked = getattr(margs, 'masked_pool', False)
    lm_dir = args.landmarks_dir or getattr(margs, 'landmarks_dir', None)
    lm_size = getattr(margs, 'landmark_img_size', 112)
    clamp = not args.no_clamp
    k = args.topk
    train_dir = args.train_dir or os.path.join(args.freq_root, 'gmdb_train')
    output_dir = args.output_dir or os.path.join(args.model_dir, 'prototype_consistency')
    classes = sorted(os.listdir(train_dir))                          # classifier class order
    cls_idx = {name: i for i, name in enumerate(classes)}
    W = torch.relu(net._classification.weight).cpu()                 # (C, P) effective weights
    P = W.shape[1]

    # region name per prototype (round-robin), if masked pooling
    region_of = None
    if masked and hasattr(net._pool, 'proto2region'):
        from pipnet.train import REGION_LANDMARKS
        names = list(REGION_LANDMARKS.keys())
        p2r = net._pool.proto2region.cpu()
        region_of = lambda d: names[int(p2r[d])]

    # ---- frequent-test images ----
    import pandas as pd
    freq_index = build_image_index(args.freq_root)
    recs = resolve_official(pd.read_csv(os.path.join(args.meta_dir, CSV_FREQ_TEST)), freq_index)
    by_syn = defaultdict(list)
    for r in recs:
        by_syn[os.path.basename(os.path.dirname(r['path']))].append(r)
    bias = net._classification.bias.detach().cpu() if net._classification.bias is not None else None

    # ===== MULTI-SYNDROME MODE: top-N by image count -> average accuracy & decision overlap =====
    if args.top_syndromes > 0:
        top_syn = [s for s, _ in sorted(by_syn.items(), key=lambda kv: -len(kv[1]))[:args.top_syndromes]
                   if s in cls_idx]
        sel = [r for s in top_syn for r in by_syn[s]]
        print('[data] top-%d frequent syndromes by image count | %d images | P=%d, k=%d'
              % (args.top_syndromes, len(sel), P, k), flush=True)
        gmap = presence_vectors(net, sel, transform, device, masked, lm_dir, lm_size, args.batch_size, clamp)
        null_ov, null_jac = random_null(k, P)
        rows = []
        print('\n  %-45s %5s %5s %6s %7s %s' % ('syndrome', 'n', 'corr', 'acc', 'mean_J', 'mean_ov'),
              flush=True)
        for s in top_syn:
            n, nc, mj, mo = syndrome_stats(by_syn[s], gmap, W, cls_idx[s], k, bias)
            rows.append((s, n, nc, nc / n, mj, mo))
            print('  %-45s %5d %5d %6.3f %7.3f %.2f/%d' % (s[:45], n, nc, nc / n, mj, mo, k), flush=True)
        accs = [r[3] for r in rows]
        tot_n, tot_c = sum(r[1] for r in rows), sum(r[2] for r in rows)
        avg_j, avg_o = float(np.nanmean([r[4] for r in rows])), float(np.nanmean([r[5] for r in rows]))
        print('\n[averages over %d syndromes]' % len(rows), flush=True)
        print('  average accuracy      : %.3f (macro over syndromes)   %.3f (micro over %d images)'
              % (float(np.mean(accs)), tot_c / tot_n, tot_n), flush=True)
        print('  average decision overlap: Jaccard=%.3f   raw-overlap=%.2f/%d   (random null J=%.3f)'
              % (avg_j, avg_o, k, null_jac), flush=True)
        os.makedirs(output_dir, exist_ok=True)
        out_csv = os.path.join(output_dir, 'decision_overlap_top%d.csv' % args.top_syndromes)
        with open(out_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['syndrome', 'n_images', 'n_correct', 'top1_acc',
                        'decision_within_jaccard', 'decision_within_overlap', 'top_k', 'P'])
            for s, n, nc, acc, mj, mo in rows:
                w.writerow([s, n, nc, '%.4f' % acc, '%.4f' % mj, '%.4f' % mo, k, P])
            w.writerow([])
            w.writerow(['AVERAGE', 'macro_acc', '%.4f' % float(np.mean(accs)),
                        'micro_acc', '%.4f' % (tot_c / tot_n),
                        'avg_jaccard', '%.4f' % avg_j, 'avg_overlap', '%.4f' % avg_o])
        print('\n[results] saved to %s' % out_csv, flush=True)
        return

    # ===== SINGLE-SYNDROME MODE =====
    target = args.syndrome
    sel = [r for r in recs if os.path.basename(os.path.dirname(r['path'])) == target]
    if not sel:
        avail = sorted({os.path.basename(os.path.dirname(r['path'])) for r in recs})
        raise SystemExit('[error] syndrome %r not in frequent test. Examples: %s'
                         % (target, ', '.join(a[:40] for a in avail[:8])))
    if target not in cls_idx:
        raise SystemExit('[error] %r is not a classifier class (train folder).' % target)
    c = cls_idx[target]
    print('[data] syndrome: %s | %d frequent-test images | P=%d prototypes, k=%d'
          % (target, len(sel), P, k), flush=True)

    gmap = presence_vectors(net, sel, transform, device, masked, lm_dir, lm_size, args.batch_size, clamp)
    b = net._classification.bias.detach().cpu() if net._classification.bias is not None else None

    # ---- classify + decision signature per patient ----
    n = len(sel)
    n_correct = 0
    dec_sets = []                                  # top-k prototypes by contribution g_d * relu(W[c,d])
    for r in sel:
        g = gmap[r['path']]
        logits = W @ g + (b if b is not None else 0.)
        n_correct += int(torch.argmax(logits)) == c
        dec_sets.append(topk_set(g * W[c], k))
    print('\n[prediction] %s: %d/%d correctly predicted (top-1 acc = %.3f)'
          % (target, n_correct, n, n_correct / n), flush=True)

    # ---- DECISION consistency: overlap of top-k contribution prototypes across patients ----
    within_j = [jaccard(a, bb) for a, bb in combinations(dec_sets, 2)]
    within_o = [len(a & bb) for a, bb in combinations(dec_sets, 2)]
    null_ov, null_jac = random_null(k, P)
    mean = lambda x: float(np.mean(x)) if x else float('nan')
    print('[decision consistency] pairs=%d  mean Jaccard=%.3f  mean raw-overlap=%.2f/%d  '
          '(random null J=%.3f)' % (len(within_j), mean(within_j), mean(within_o), k, null_jac), flush=True)

    # overlapping prototypes = those in >1 patient's top-k decision signature, by appearance count
    cnt = Counter(d for s in dec_sets for d in s)
    overlap = [(d, m) for d, m in cnt.most_common() if m > 1]
    print('\n[overlapping decision prototypes] (in top-%d contribution for >1 of %d patients):' % (k, n),
          flush=True)
    print('  %-8s %-14s %-10s %s' % ('proto', 'region', 'patients', 'fraction'), flush=True)
    for d, m in overlap:
        print('  p%-7d %-14s %-10s %.3f' % (d, region_of(d) if region_of else '-',
              '%d/%d' % (m, n), m / n), flush=True)
    if not overlap:
        print('  (no prototype appeared in more than one patient)', flush=True)

    # ---- save ----
    os.makedirs(output_dir, exist_ok=True)
    fname = 'decision_overlap_%s.csv' % ''.join(ch if ch.isalnum() else '_' for ch in target)[:50]
    with open(os.path.join(output_dir, fname), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['syndrome', 'n_images', 'n_correct', 'top1_acc',
                    'decision_within_jaccard', 'decision_within_overlap', 'top_k', 'P'])
        w.writerow([target, n, n_correct, '%.4f' % (n_correct / n),
                    '%.4f' % mean(within_j), '%.4f' % mean(within_o), k, P])
        w.writerow([])
        w.writerow(['overlapping_prototype', 'region', 'n_patients', 'fraction'])
        for d, m in overlap:
            w.writerow([d, region_of(d) if region_of else '', m, '%.4f' % (m / n)])
    print('\n[results] saved to %s' % os.path.join(output_dir, fname), flush=True)


if __name__ == '__main__':
    main()
