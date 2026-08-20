"""Prototype purity: region-masked vs vanilla PIP-Net (Metric 1 = top-K exemplar region purity).

For each prototype we take its top-K most-activating image patches (the peak of its activation
map in the K images where it fires hardest), label each patch by the facial region its peak
falls in (nearest landmark within --min_dist; else 'none'), and define

    purity(d) = fraction of the K patches that fall in the prototype's majority region.

Purity is measured on the RAW prototype activation maps (proto_features), identically for both
models -- masked pooling only restricts POOLING, not the activation map, so scoring the raw-map
peak is the fair comparison (otherwise the masked model would be pure by construction). We report
mean purity over USED prototypes (nonzero classification weight, active in >= K images) for each
model, a region-prevalence null (majority fraction of K labels drawn from the empirical region
distribution, controlling for unequal region sizes), and a Mann-Whitney U test between models.

Run: python prototype_purity.py   (defaults compare the given masked vs vanilla convnext runs)
"""
import os
import sys
import csv
import argparse

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gmdb_common import (
    load_pipnet, build_transform, build_image_index, resolve_official,
    DEFAULT_FREQ_ROOT, DEFAULT_META_DIR, CSV_FREQ_TEST)
from util.data import load_landmarks, landmark_path_for_image, NUM_LANDMARKS
from pipnet.train import REGION_LANDMARKS

MASKED_DIR  = '/data2/masino_lab/abamini/pipnet/runs/run_pipnet_20260630_075413_convnext_tiny_26_True/'
VANILLA_DIR = '/data2/masino_lab/abamini/pipnet/runs/run_pipnet_20260630_003311_convnext_tiny_26_False/'
DEFAULT_LANDMARKS = '/data2/masino_lab/abamini/facial_data/Final_data/gmdb_landmarks_all/'
REGION_NAMES = list(REGION_LANDMARKS.keys()) + ['none']   # index -1 -> 'none'


def label_regions(peaks, lms, region_idx, radius):
    """Region index per peak (nearest region landmark within `radius`, else -1='none').
    peaks (B,P,2) normalized (row,col); lms (B,L,2) normalized, -1 = missing."""
    B, P, _ = peaks.shape
    R = len(region_idx)
    mind = torch.full((B, P, R), float('inf'), device=peaks.device)
    for g, idx in enumerate(region_idx):
        idx_t = torch.tensor(idx, device=peaks.device, dtype=torch.long)
        pts = lms[:, idx_t, :]                                    # (B, kg, 2)
        valid = (pts >= 0).all(dim=-1)                            # (B, kg)
        d2 = ((peaks[:, :, None, :] - pts[:, None, :, :]) ** 2).sum(dim=-1)   # (B, P, kg)
        d2 = d2.masked_fill(~valid[:, None, :], float('inf'))
        mind[:, :, g] = d2.min(dim=2).values.sqrt()
    region = mind.argmin(dim=2)                                   # (B, P)
    return torch.where(mind.min(dim=2).values <= radius, region, torch.full_like(region, -1))


@torch.no_grad()
def encode_peaks(net, paths, lms_map, transform, device, region_idx, radius, batch):
    """Return peak_val (N,P) and peak_region (N,P, int; -1=none) for images in `paths` order."""
    pv_all, pr_all = [], []
    for i in tqdm(range(0, len(paths), batch), desc='peaks %d imgs' % len(paths), ncols=0):
        chunk = paths[i:i + batch]
        imgs = torch.stack([transform(Image.open(p).convert('RGB')) for p in chunk]).to(device)
        lms = torch.stack([lms_map[p] for p in chunk]).to(device)         # (B, L, 2)
        pf, _, _ = net(imgs)                                              # (B, P, H, W)
        B, P, H, W = pf.shape
        flat = pf.flatten(2)                                             # (B, P, H*W)
        pv, idx = flat.max(dim=2)                                        # (B, P)
        h = (idx // W).float() / max(H - 1, 1)
        w = (idx % W).float() / max(W - 1, 1)
        region = label_regions(torch.stack([h, w], dim=2), lms, region_idx, radius)
        pv_all.append(pv.cpu().numpy())
        pr_all.append(region.cpu().numpy())
    return np.concatenate(pv_all, 0), np.concatenate(pr_all, 0)


def purities(peak_val, peak_region, used, K, thresh):
    """purity per used prototype (majority-region fraction of its top-K patches) + the top-K
    region labels (for the null). Skips prototypes whose K-th strongest peak is < thresh."""
    N, P = peak_val.shape
    out, regs = {}, {}
    for d in np.where(used)[0]:
        order = np.argsort(peak_val[:, d])[::-1][:K]
        if len(order) < K or peak_val[order[-1], d] < thresh:
            continue
        r = peak_region[order, d]
        out[d] = np.bincount(r + 1).max() / K                # +1 so -1('none') is a valid bin
        regs[d] = r
    return out, regs


def null_purity(regs, K, seed=0, reps=3000):
    """Chance majority fraction: draw K region labels i.i.d. from the empirical pool (controls
    for region prevalence / unequal region sizes)."""
    pool = np.concatenate(list(regs.values())) if regs else np.array([])
    if pool.size == 0:
        return float('nan')
    rng = np.random.default_rng(seed)
    return float(np.mean([np.bincount(rng.choice(pool, K) + 1).max() / K for _ in range(reps)]))


def analyze(model_dir, checkpoint, label, paths, lms_map, transform, device, region_idx,
            radius, K, thresh, batch):
    net, _ = load_pipnet(model_dir, checkpoint, device)
    W = torch.relu(net._classification.weight).detach().cpu().numpy()    # (C, P)
    used = W.sum(0) > 0                                                  # prototypes with class weight
    pv, pr = encode_peaks(net, paths, lms_map, transform, device, region_idx, radius, batch)
    pur, regs = purities(pv, pr, used, K, thresh)
    vals = np.array(list(pur.values()))
    null = null_purity(regs, K)
    print('\n[%s] used-with-weight=%d | scored prototypes (active>=%d imgs)=%d' % (
        label, int(used.sum()), K, len(vals)), flush=True)
    print('  mean purity=%.3f  std=%.3f  median=%.3f  frac(pure>0.5)=%.3f  |  null=%.3f' % (
        np.mean(vals), np.std(vals), np.median(vals), np.mean(vals > 0.5), null), flush=True)
    return dict(label=label, purity=pur, vals=vals, null=null, n_scored=len(vals),
                n_used=int(used.sum()))


def main():
    p = argparse.ArgumentParser(description='Prototype region purity: masked vs vanilla PIP-Net.')
    p.add_argument('--model_masked', default=MASKED_DIR)
    p.add_argument('--model_vanilla', default=VANILLA_DIR)
    p.add_argument('--checkpoint', default='net_trained')
    p.add_argument('--freq_root', default=DEFAULT_FREQ_ROOT)
    p.add_argument('--meta_dir', default=DEFAULT_META_DIR)
    p.add_argument('--landmarks_dir', default=None, help='Default: masked run\'s landmarks_dir or gmdb_landmarks_all.')
    p.add_argument('--topk_patches', type=int, default=15, help='K exemplar patches per prototype.')
    p.add_argument('--min_activation', type=float, default=0.1,
                   help='A prototype is scored only if its K-th strongest peak >= this.')
    p.add_argument('--min_dist', type=float, default=None,
                   help='Max distance (normalized) from a peak to a region landmark to count as in-region; '
                        'else the patch is labelled "none". Default: masked run\'s mask_radius.')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--output_dir', default=None, help='Default: <model_masked>/prototype_purity.')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    device = torch.device(args.device)
    # config from the masked run (transform, landmarks, radius); both runs share backbone/data
    _, m_margs = load_pipnet(args.model_masked, args.checkpoint, device)
    transform = build_transform(m_margs.net, m_margs.image_size)
    lm_dir = args.landmarks_dir or getattr(m_margs, 'landmarks_dir', None) or DEFAULT_LANDMARKS
    lm_size = getattr(m_margs, 'landmark_img_size', 112)
    radius = args.min_dist if args.min_dist is not None else getattr(m_margs, 'mask_radius', 0.13)
    region_idx = list(REGION_LANDMARKS.values())
    K, thresh = args.topk_patches, args.min_activation
    output_dir = args.output_dir or os.path.join(args.model_masked, 'prototype_purity')

    # frequent-test images + precomputed landmarks (shared by both models)
    import pandas as pd
    freq_index = build_image_index(args.freq_root)
    recs = resolve_official(pd.read_csv(os.path.join(args.meta_dir, CSV_FREQ_TEST)), freq_index)
    paths = sorted({r['path'] for r in recs})
    lms_map = {p: torch.from_numpy(load_landmarks(landmark_path_for_image(p, lm_dir), lm_size, NUM_LANDMARKS))
               for p in paths}
    print('[data] %d frequent-test images | K=%d, min_activation=%.2f, region_radius=%.3f' % (
        len(paths), K, thresh, radius), flush=True)

    r_mask = analyze(args.model_masked, args.checkpoint, 'MASKED', paths, lms_map, transform,
                     device, region_idx, radius, K, thresh, args.batch_size)
    r_van = analyze(args.model_vanilla, args.checkpoint, 'VANILLA', paths, lms_map, transform,
                    device, region_idx, radius, K, thresh, args.batch_size)

    # comparison
    print('\n' + '=' * 70 + '\n[comparison] top-K exemplar region purity (higher = more anatomically pure)',
          flush=True)
    print('  MASKED  mean=%.3f (null %.3f, +%.3f over null)  n=%d' % (
        np.mean(r_mask['vals']), r_mask['null'], np.mean(r_mask['vals']) - r_mask['null'], r_mask['n_scored']),
        flush=True)
    print('  VANILLA mean=%.3f (null %.3f, +%.3f over null)  n=%d' % (
        np.mean(r_van['vals']), r_van['null'], np.mean(r_van['vals']) - r_van['null'], r_van['n_scored']),
        flush=True)
    try:
        from scipy.stats import mannwhitneyu
        u, pv = mannwhitneyu(r_mask['vals'], r_van['vals'], alternative='two-sided')
        print('  Mann-Whitney U: p=%.3e  (%s)' % (
            pv, 'masked significantly different' if pv < 0.05 else 'no significant difference'), flush=True)
    except Exception as e:
        print('  (scipy Mann-Whitney skipped: %s)' % repr(e)[:80], flush=True)

    # save per-prototype purity for both models
    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, 'prototype_purity.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model', 'prototype', 'purity', 'top_k', 'region_radius'])
        for r in (r_mask, r_van):
            for d, val in sorted(r['purity'].items()):
                w.writerow([r['label'], d, '%.4f' % val, K, '%.3f' % radius])
    with open(os.path.join(output_dir, 'prototype_purity_summary.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model', 'n_used_with_weight', 'n_scored', 'mean_purity', 'std', 'median',
                    'frac_pure_gt0.5', 'null_purity', 'top_k'])
        for r in (r_mask, r_van):
            v = r['vals']
            w.writerow([r['label'], r['n_used'], r['n_scored'], '%.4f' % np.mean(v), '%.4f' % np.std(v),
                        '%.4f' % np.median(v), '%.4f' % np.mean(v > 0.5), '%.4f' % r['null'], K])
    print('\n[results] saved to %s' % output_dir, flush=True)


if __name__ == '__main__':
    main()
