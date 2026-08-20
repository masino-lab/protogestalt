"""Syndrome prediction for PIP-Net on GMDB, using the RIGHT tool for each split:

  GMDB-FREQUENT (seen)   -> the trained PIP-Net CLASSIFIER head. Every frequent syndrome has
                            an output unit (the model was trained on gmdb_train's 275 classes),
                            so we read the prediction straight off argmax/top-k of the logits
                            o_c = <relu(W[c,:]), g>  (g = prototype-presence vector). This is the
                            model's actual capability on seen classes.
  GMDB-RARE (unseen)     -> the GestaltMatcher retrieval (kNN) protocol from gallery_validation.
                            The head has NO unit for rare syndromes, so we rank each rare test
                            embedding by cosine against a rare gallery (10-fold CV).

Both are reported as class-mean top-k accuracy  mA_k = (1/C) sum_c A_{k,c}. NOTE the two blocks
use DIFFERENT mechanisms (classifier vs retrieval) -- comparable in form, not directly to each
other.

Ground truth for the classifier: a frequent-test image lives at <freq_root>/gmdb_test/<syndrome>/
<id>.jpg, so its true syndrome is its PARENT FOLDER NAME -- which is exactly one of the
classifier's class names (sorted(gmdb_train)). This avoids the frequent-CSV label scheme, which
need not match the ImageFolder class order the head was trained in.

Run:  python syndrome_prediction.py --model_dir <run_dir>
"""
import os
import sys
import argparse

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gmdb_common import (
    load_pipnet, build_transform, build_image_index, resolve_official, embed,
    DEFAULT_FREQ_ROOT, DEFAULT_RARE_ROOT, DEFAULT_META_DIR, DEFAULT_MODEL_DIR,
    CSV_FREQ_TEST, CSV_RARE_GALLERY, CSV_RARE_TEST)
from util.data import load_landmarks, landmark_path_for_image, NUM_LANDMARKS


# --- frequent: classifier head ---------------------------------------------
@torch.no_grad()
def classify(net, records, transform, device, masked, landmarks_dir, landmark_img_size,
             batch_size=32, clamp=True):
    """Run the trained classifier over each unique image path -> {path: logits (C,)}."""
    paths = sorted({r['path'] for r in records})
    out = {}
    for i in tqdm(range(0, len(paths), batch_size), desc='classify %d imgs' % len(paths), ncols=0):
        chunk = paths[i:i + batch_size]
        imgs = torch.stack([transform(Image.open(p).convert('RGB')) for p in chunk]).to(device)
        lms = None
        if masked:
            lms = torch.stack([torch.from_numpy(load_landmarks(
                landmark_path_for_image(p, landmarks_dir), landmark_img_size, NUM_LANDMARKS))
                for p in chunk]).to(device)
        _, _, logits = net(imgs, inference=clamp, lms=lms)
        for p, o in zip(chunk, logits.float().cpu()):
            out[p] = o
    return out


def classifier_scores(out_map, records, classes, topks):
    """Micro (per-image) top-k of the classifier. Ground truth = parent folder name;
    prediction = classes[argsort(logits)]. Returns {k: micro} and #test classes."""
    maxk = max(topks)
    truth = [os.path.basename(os.path.dirname(r['path'])) for r in records]
    hits = {k: [] for k in topks}
    for r, t in zip(records, truth):
        top_names = [classes[int(j)] for j in torch.argsort(out_map[r['path']], descending=True)[:maxk]]
        for k in topks:
            hits[k].append(t in top_names[:k])
    res = {k: float(np.mean(hits[k])) for k in topks}
    return res, len(set(truth))


# --- rare: retrieval, MICRO (per-image) top-k ------------------------------
def retrieval_micro(gallery, test, emb, topks, device, mode='knn', temp=0.1):
    """Per-image (micro) top-k retrieval. mode='knn' = image-kNN; 'mean'/'max'/'softmax' =
    class-aggregated (NCM). A probe's own gallery copy is excluded. Returns {k: micro}."""
    g = torch.stack([emb[r['path']] for r in gallery]).to(device)
    g_lab = np.array([r['label'] for r in gallery])
    g_id = torch.as_tensor([r['image_id'] for r in gallery], device=device)
    t_lab = np.array([r['label'] for r in test])
    t_id = torch.as_tensor([r['image_id'] for r in test], device=device)
    t = torch.stack([emb[r['path']] for r in test]).to(device)
    maxk = max(topks)
    sims = t @ g.t()                                              # (Nq, Ng)
    own = t_id[:, None] == g_id[None, :]
    if mode == 'knn':
        pred = g_lab[sims.masked_fill(own, -1e9).topk(min(maxk, g.shape[0]), dim=1)
                     .indices.cpu().numpy()]
    else:
        classes = np.unique(g_lab)
        valid = ~own
        cs = torch.full((sims.shape[0], len(classes)), float('-inf'), device=device)
        for ci, c in enumerate(classes):
            cols = torch.as_tensor(np.where(g_lab == c)[0], device=device)
            sub, vm = sims[:, cols], valid[:, cols]
            if mode == 'max':
                cs[:, ci] = sub.masked_fill(~vm, float('-inf')).amax(1)
            elif mode == 'softmax':
                w = torch.softmax((sub / temp).masked_fill(~vm, float('-inf')), 1)
                cs[:, ci] = (w * sub.masked_fill(~vm, 0.)).sum(1)
            else:  # mean = nearest-class-mean
                cnt, s = vm.sum(1), (sub * vm).sum(1)
                cs[:, ci] = torch.where(cnt > 0, s / cnt.clamp(min=1),
                                        torch.full_like(s, float('-inf')))
        cs = torch.nan_to_num(cs, nan=float('-inf'))
        pred = classes[cs.topk(min(maxk, cs.shape[1]), dim=1).indices.cpu().numpy()]
    hits = (pred == t_lab[:, None])
    return {k: float(hits[:, :min(k, pred.shape[1])].any(1).mean()) for k in topks}


# --- rare: GestaltMatcher retrieval (reuses gallery_validation.score) -------
def rare_retrieval(net, rare_index, meta_dir, transform, device, masked, landmarks_dir,
                   landmark_img_size, topks, batch_size, clamp, class_aggregate='none'):
    """10-fold-averaged retrieval mA_k on GMDB-rare (per gallery_validation)."""
    import pandas as pd
    rg = pd.read_csv(os.path.join(meta_dir, CSV_RARE_GALLERY))
    rt = pd.read_csv(os.path.join(meta_dir, CSV_RARE_TEST))
    folds = sorted(int(s) for s in rg['split'].unique())
    gallery = {k: resolve_official(rg[rg['split'] == k], rare_index) for k in folds}
    test = {k: resolve_official(rt[rt['split'] == k], rare_index) for k in folds}
    all_recs = [r for k in folds for r in gallery[k] + test[k]]
    emb = embed(net, all_recs, transform, device, masked,
                [landmarks_dir] if landmarks_dir else None,
                landmark_img_size, batch_size, clamp)

    def fold_avg(mode):
        fs = [retrieval_micro(gallery[k], test[k], emb, topks, device, mode=mode)
              for k in folds if test[k]]
        return {k: {'micro': float(np.mean([f[k] for f in fs])),
                    'std':   float(np.std([f[k] for f in fs]))} for k in topks}, len(fs)

    knn, nfolds = fold_avg('knn')
    agg = fold_avg(class_aggregate)[0] if class_aggregate and class_aggregate != 'none' else None
    return knn, agg, nfolds


# --- driver ----------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description='GMDB syndrome prediction: classifier for frequent '
                                            '(seen), GestaltMatcher retrieval for rare (unseen).')
    p.add_argument('--model_dir', default=DEFAULT_MODEL_DIR)
    p.add_argument('--checkpoint', default='net_trained')
    p.add_argument('--freq_root', default=DEFAULT_FREQ_ROOT)
    p.add_argument('--rare_root', default=DEFAULT_RARE_ROOT)
    p.add_argument('--meta_dir', default=DEFAULT_META_DIR)
    p.add_argument('--train_dir', default=None,
                   help='ImageFolder whose sorted class names are the classifier classes. '
                        'Default: <freq_root>/gmdb_train.')
    p.add_argument('--landmarks_dir', default=None, help='Default: the run\'s training landmarks_dir.')
    p.add_argument('--output_dir', default=None, help='Default: <model_dir>/syndrome_prediction.')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--no_clamp', action='store_true',
                   help='Use the un-clamped presence vector (default clamps <0.1, matching training).')
    p.add_argument('--topk', default='1,5,10')
    p.add_argument('--class_aggregate', choices=['none', 'mean', 'max', 'softmax'], default='mean',
                   help='Also report class-aggregated (NCM) retrieval for rare.')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    device = torch.device(args.device)
    net, margs = load_pipnet(args.model_dir, args.checkpoint, device)
    transform = build_transform(margs.net, margs.image_size)
    masked = getattr(margs, 'masked_pool', False)
    lm_dir = args.landmarks_dir or getattr(margs, 'landmarks_dir', None)
    lm_size = getattr(margs, 'landmark_img_size', 112)
    clamp = not args.no_clamp
    topks = tuple(int(x) for x in args.topk.split(','))
    train_dir = args.train_dir or os.path.join(args.freq_root, 'gmdb_train')
    output_dir = args.output_dir or os.path.join(args.model_dir, 'syndrome_prediction')

    # classifier classes = ImageFolder order (sorted folder names)
    classes = sorted(os.listdir(train_dir))
    n_out = net._classification.weight.shape[0]
    if len(classes) != n_out:
        print('[warn] %d train folders but classifier has %d outputs -- class alignment may be off.'
              % (len(classes), n_out), flush=True)

    # ---- FREQUENT: classifier ----
    freq_index = build_image_index(args.freq_root)
    import pandas as pd
    ft = pd.read_csv(os.path.join(args.meta_dir, CSV_FREQ_TEST))
    freq_test = resolve_official(ft, freq_index)
    print('[frequent] %d test images | %d classifier classes' % (len(freq_test), len(classes)),
          flush=True)
    out_map = classify(net, freq_test, transform, device, masked, lm_dir, lm_size,
                       args.batch_size, clamp)
    freq_res, freq_C = classifier_scores(out_map, freq_test, classes, topks)

    # ---- RARE: retrieval ----
    rare_index = build_image_index(args.rare_root)
    rare_knn, rare_agg, nfolds = rare_retrieval(
        net, rare_index, args.meta_dir, transform, device, masked, lm_dir, lm_size,
        topks, args.batch_size, clamp, args.class_aggregate)

    # ---- report (micro = per-image top-k accuracy) ----
    lines = ['GMDB syndrome prediction (micro = per-image top-k accuracy)', '']
    lines.append('===== FREQUENT (seen) — PIP-Net CLASSIFIER head =====')
    lines.append('C=%d n=%d : %s' % (freq_C, len(freq_test),
        '  '.join('top%d=%.4f (%.2f%%)' % (k, freq_res[k], 100 * freq_res[k]) for k in topks)))
    lines.append('')
    lines.append('===== RARE (unseen, %d-fold) — GestaltMatcher retrieval (kNN) =====' % nfolds)
    lines.append('image-kNN : %s' % '  '.join('top%d=%.4f±%.4f (%.2f±%.2f%%)' % (
        k, rare_knn[k]['micro'], rare_knn[k]['std'],
        100 * rare_knn[k]['micro'], 100 * rare_knn[k]['std']) for k in topks))
    if rare_agg is not None:
        lines.append('class-agg (%s) : %s' % (args.class_aggregate, '  '.join(
            'top%d=%.4f±%.4f (%.2f±%.2f%%)' % (k, rare_agg[k]['micro'], rare_agg[k]['std'],
             100 * rare_agg[k]['micro'], 100 * rare_agg[k]['std']) for k in topks)))
    print('\n' + '\n'.join(lines), flush=True)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'syndrome_prediction_results.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n[results] saved to %s' % output_dir, flush=True)


if __name__ == '__main__':
    main()
