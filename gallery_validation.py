"""
Gallery-based (retrieval) validation for PIP-Net on GMDB — GestaltMatcher protocol
(Hsieh 2022; Hustinx 2023).

The model encodes each image into its masked-pooled prototype-presence vector.
A labelled gallery of these vectors is built; each test vector is ranked against
the gallery by cosine similarity and classified by the disorder of its nearest
neighbour. We report class-mean top-k accuracy

    mA_k = (1/C) * sum_c A_{k,c}
  Frequent (seen) : official single split — gallery = gmdb_frequent_gallery_images,
                    test = gmdb_frequent_test_images CSVs. Patient-disjoint by the
                    original GMDB split (no train/test leak).
  Rare (unseen)   : official 10-fold CV (gmdb_rare_{gallery,test}_images CSVs,
                    `split` column 0-9); mA_k averaged over folds.

Both sets are built from the original GMDB metadata CSVs (GestaltMatcher protocol);
image_ids are resolved to on-disk crops under the respective root.

Run standalone:   python gallery_validation.py --model_dir <run_dir>
Or in pipeline:   main.py --gallery_validation
"""

import os
import sys
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gmdb_common import (
    DEFAULT_FREQ_ROOT, DEFAULT_RARE_ROOT, DEFAULT_META_DIR, DEFAULT_MODEL_DIR,
    CSV_FREQ_GALLERY, CSV_FREQ_TEST, CSV_RARE_GALLERY, CSV_RARE_TEST,
    REDUCED_FREQ_SYNDROMES, REDUCED_RARE_SYNDROMES,
    load_pipnet, build_transform, build_image_index, resolve_official,
    top_syndromes_by_image_count, embed, _image_id_from_stem)


# --- retrieval + class-mean accuracy --------------------------------------
def score(gallery, test, emb, topks, device, csls=0):
    """Cosine-rank `test` against `gallery`; return per-k {macro} and #classes.

    csls>0: rank by CSLS (2*sim - r_S) with r_S the mean of each gallery item's top-`csls`
    similarities to the queries — corrects kNN hubness (a few gallery vectors being
    everyone's neighbour). r_S is constant per gallery item, so it only re-ranks columns."""
    g = torch.stack([emb[r['path']] for r in gallery]).to(device)
    t = torch.stack([emb[r['path']] for r in test]).to(device)
    g_lab = np.array([r['label'] for r in gallery])
    g_id = np.array([r['image_id'] for r in gallery])
    t_id = np.array([r['image_id'] for r in test])
    t_lab = np.array([r['label'] for r in test])
    maxk = max(topks)

    sims = t @ g.t()                                              # (Nq, Ng)
    own = torch.as_tensor(t_id[:, None] == g_id[None, :], device=device)  # probe's own copy
    sims = sims.masked_fill(own, -1e9)
    if csls > 0:
        k = min(csls, sims.shape[0])
        r_s = sims.topk(k, dim=0).values.clamp(min=-1.).mean(dim=0, keepdim=True)  # (1, Ng)
        rank = (2 * sims - r_s).masked_fill(own, -1e9)
    else:
        rank = sims
    retrieved = g_lab[rank.topk(min(maxk, g.shape[0]), dim=1).indices.cpu().numpy()]

    hits = (retrieved == t_lab[:, None])
    classes = np.unique(t_lab)
    res = {}
    for k in topks:
        correct = hits[:, :min(k, retrieved.shape[1])].any(axis=1)
        macro = np.mean([correct[t_lab == c].mean() for c in classes])
        res[k] = {'macro': float(macro)}
    return res, len(classes)


# --- class-aggregated (nearest-class-mean) retrieval ----------------------
def score_class_aggregate(gallery, test, emb, topks, device, mode='mean', temp=0.1):
    """Rank *syndromes* (not individual images) for each probe.

    Image-kNN (`score`) lets a syndrome with many gallery images grab more of the
    near-neighbour slots, biasing top-k toward populous syndromes. Here each syndrome
    gets ONE score — an aggregate of the probe's cosine similarities to that
    syndrome's gallery images — and syndromes are ranked by it, removing the
    gallery-abundance bias.

      mode='mean'    : average similarity = cosine to the class centroid
                       (nearest-class-mean). Count-neutral; the recommended default.
      mode='max'     : similarity to the single nearest image of the class
                       (per-class NN); the class-level analogue of image-kNN top-1.
      mode='softmax' : temperature-weighted smooth-max in [max, mean]
                       (temp->0 -> max, temp->inf -> mean).

    A probe's own gallery copy is excluded per (probe, image). Macro is over test
    classes, matching `score`. Returns per-k {macro} and #test classes."""
    g = torch.stack([emb[r['path']] for r in gallery]).to(device)
    t_all = torch.stack([emb[r['path']] for r in test]).to(device)
    g_lab = np.array([r['label'] for r in gallery])
    g_id = torch.as_tensor([r['image_id'] for r in gallery], device=device)
    t_lab = np.array([r['label'] for r in test])
    t_ids_all = torch.as_tensor([r['image_id'] for r in test], device=device)
    gal_classes = np.unique(g_lab)
    cls_cols = [torch.as_tensor(np.where(g_lab == c)[0], device=device) for c in gal_classes]
    maxk = max(topks)

    pred = []
    for i in range(0, len(test), 512):
        pe = t_all[i:i + 512]
        sims = pe @ g.t()                                          # (B, N)
        pid = t_ids_all[i:i + 512]
        valid_all = pid[:, None] != g_id[None, :]                  # (B, N) drop own copy
        B = sims.shape[0]
        cscores = torch.full((B, len(gal_classes)), float('-inf'), device=device)
        for ci, cols in enumerate(cls_cols):
            sub = sims[:, cols]                                    # (B, n_c)
            vm = valid_all[:, cols]
            if mode == 'max':
                cscores[:, ci] = sub.masked_fill(~vm, float('-inf')).amax(dim=1)
            elif mode == 'softmax':
                wts = torch.softmax((sub / temp).masked_fill(~vm, float('-inf')), dim=1)
                cscores[:, ci] = (wts * sub.masked_fill(~vm, 0.)).sum(dim=1)
            else:  # mean (nearest-class-mean)
                cnt = vm.sum(dim=1)
                s = (sub * vm).sum(dim=1)
                cscores[:, ci] = torch.where(cnt > 0, s / cnt.clamp(min=1),
                                             torch.full_like(s, float('-inf')))
        cscores = torch.nan_to_num(cscores, nan=float('-inf'))     # all-invalid rows
        top = cscores.topk(min(maxk, cscores.shape[1]), dim=1).indices.cpu().numpy()
        pred.append(gal_classes[top])                             # (B, kk) class labels
    pred = np.concatenate(pred, axis=0)

    hits = (pred == t_lab[:, None])
    classes = np.unique(t_lab)
    res = {}
    for k in topks:
        correct = hits[:, :min(k, pred.shape[1])].any(axis=1)
        macro = np.mean([correct[t_lab == c].mean() for c in classes])
        res[k] = {'macro': float(macro)}
    return res, len(classes)


# --- NCE projection head ---------------------------------------------------
class ProjectionHead(nn.Module):
    """Maps a prototype-presence vector (P,) to an L2-normalized embedding (out_dim,).
    `hidden`=0 gives a plain linear metric; >0 adds one nonlinear hidden layer so the
    head can model prototype interactions a linear/diagonal metric cannot."""
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        if hidden > 0:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True), ## remove hidden layer, like logistic regression  
                nn.Linear(hidden, out_dim))
        else:
            self.net = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)


def train_nce_head(X, y, device, *, hidden, out_dim, iters, lr, temperature,
                   batch_classes=32, per_class=4, weight_decay=1e-4, seed=0):
    """Train a projection head by supervised contrastive (InfoNCE-with-labels) on frozen
    embeddings X (N,P) with labels y, so same-syndrome patients are close under cosine.

    Trained ONLY on the frequent gallery (passed in by the caller); the Rare split is
    never seen, so its numbers stay a true open-set test. PK batches (`batch_classes`
    syndromes x `per_class` images) guarantee positives per anchor; singleton syndromes
    are skipped for sampling but the head still applies to them at eval time."""
    rng = np.random.default_rng(seed)
    X = X.to(device)
    by_class = defaultdict(list)
    for i, c in enumerate(y):
        by_class[int(c)].append(i)
    head = ProjectionHead(X.shape[1], hidden, out_dim).to(device)
    classes = [c for c, idx in by_class.items() if len(idx) >= 2]
    if len(classes) < 2:
        print('[nce] <2 multi-image syndromes; head left at init.', flush=True)
        return head.eval()
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    head.train()
    for it in range(iters):
        cs = rng.choice(classes, size=min(batch_classes, len(classes)), replace=False)
        idx = []
        for c in cs:
            pool = by_class[int(c)]
            sel = rng.choice(pool, size=min(per_class, len(pool)), replace=len(pool) < per_class)
            idx.extend(int(s) for s in sel)
        idx = np.asarray(idx)
        z = head(X[idx])                                   # (B, out_dim), L2-normalized
        yb = torch.as_tensor(y[idx], device=device)
        B = z.shape[0]
        sim = (z @ z.t()) / temperature
        eye = torch.eye(B, dtype=torch.bool, device=device)
        sim = sim.masked_fill(eye, -1e9)                   # exclude self
        logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        pos = (yb[:, None] == yb[None, :]) & ~eye          # same-syndrome positives
        pc = pos.sum(1)
        valid = pc > 0
        loss = (-(logp * pos).sum(1)[valid] / pc[valid].clamp(min=1)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it == 0 or (it + 1) % 50 == 0:
            print('[nce] iter %d/%d  loss=%.4f' % (it + 1, iters, loss.item()), flush=True)
    return head.eval()


@torch.no_grad()
def apply_head(emb, head, device, batch=4096):
    """Re-embed every stored presence vector through the trained head -> new {path: tensor}."""
    paths = list(emb)
    X = torch.stack([emb[p] for p in paths])
    out = {}
    for i in range(0, len(paths), batch):
        z = head(X[i:i + batch].to(device)).cpu()
        for p, e in zip(paths[i:i + batch], z):
            out[p] = e
    return out


# --- external (Hustinx 2023) embeddings ------------------------------------
def load_hustinx_embeddings(csv_path):
    """Aggregate Hustinx (2023) GestaltMatcher encodings into one L2-normalized 512-d vector
    per image_id, for an apples-to-apples retrieval comparison.

    The ';'-delimited CSV has 12 rows per image (3 models x flip x gray); each
    'representations' cell is a JSON list. We L2-normalize every row, average them, and
    renormalize (the ensemble + flip/gray mean direction). Returns {image_id:int -> tensor}."""
    import csv as _csv
    _csv.field_size_limit(1 << 24)
    sums, counts = {}, defaultdict(int)
    with open(csv_path) as fh:
        r = _csv.reader(fh, delimiter=';')
        header = next(r)
        ci_name, ci_rep = header.index('img_name'), header.index('representations')
        n = 0
        for row in r:
            iid = _image_id_from_stem(os.path.splitext(row[ci_name])[0])
            if iid is None:
                continue
            iid = int(iid)
            v = np.fromstring(row[ci_rep].strip()[1:-1], sep=',', dtype=np.float32)
            v = v / (np.linalg.norm(v) + 1e-12)
            sums[iid] = v if iid not in sums else sums[iid] + v
            counts[iid] += 1
            n += 1
            if n % 20000 == 0:
                print('[hustinx] parsed %d rows...' % n, flush=True)
    out = {}
    for iid, s in sums.items():
        v = s / counts[iid]
        out[iid] = torch.from_numpy((v / (np.linalg.norm(v) + 1e-12)).astype(np.float32))
    print('[hustinx] %d images aggregated (%d rows)' % (len(out), n), flush=True)
    return out


# --- driver ----------------------------------------------------------------
def run_validation(net, device, *, net_name, image_size, masked, landmarks_dir,
                   landmark_img_size, freq_root, rare_root, meta_dir, output_dir,
                   batch_size=32, clamp=False, topks=(1, 5), reduced_syndromes=False,
                   class_aggregate='none', agg_temp=0.1, tta=(), csls=0,
                   nce_head=False, nce_dim=128, nce_hidden=512, nce_iters=300,
                   nce_lr=1e-3, nce_temp=0.1, hustinx_path=None):
    import pandas as pd
    transform = build_transform(net_name, image_size)

    # Build an image_id->path index once per root (frequent ImageFolder vs flat aligned dir).
    freq_index = build_image_index(freq_root)
    rare_index = build_image_index(rare_root)

    fg = pd.read_csv(os.path.join(meta_dir, CSV_FREQ_GALLERY))
    ft = pd.read_csv(os.path.join(meta_dir, CSV_FREQ_TEST))
    freq_gallery = resolve_official(fg, freq_index)
    freq_test = resolve_official(ft, freq_index)
    rg = pd.read_csv(os.path.join(meta_dir, CSV_RARE_GALLERY))
    rt = pd.read_csv(os.path.join(meta_dir, CSV_RARE_TEST))
    folds = sorted(int(s) for s in rg['split'].unique())
    rare_gallery = {k: resolve_official(rg[rg['split'] == k], rare_index) for k in folds}
    rare_test = {k: resolve_official(rt[rt['split'] == k], rare_index) for k in folds}

    # Optionally restrict accuracy to the most-populated syndromes (top-204 frequent,
    # top-245 rare by distinct image count). The same syndrome set is applied to every
    # rare fold's gallery and test so folds stay comparable.
    if reduced_syndromes:
        freq_keep = top_syndromes_by_image_count(freq_gallery + freq_test, REDUCED_FREQ_SYNDROMES)
        freq_gallery = [r for r in freq_gallery if r['label'] in freq_keep]
        freq_test = [r for r in freq_test if r['label'] in freq_keep]
        rare_keep = top_syndromes_by_image_count(
            [r for k in folds for r in rare_gallery[k] + rare_test[k]], REDUCED_RARE_SYNDROMES)
        rare_gallery = {k: [r for r in rare_gallery[k] if r['label'] in rare_keep] for k in folds}
        rare_test = {k: [r for r in rare_test[k] if r['label'] in rare_keep] for k in folds}
        print('[reduced_syndromes] frequent -> %d syndromes, rare -> %d syndromes' % (
            len(freq_keep), len(rare_keep)), flush=True)

    print('[data] frequent: %d gallery / %d test | rare: %d folds' % (
        len(freq_gallery), len(freq_test), len(folds)), flush=True)

    all_recs = freq_gallery + freq_test + [r for k in folds for r in rare_gallery[k] + rare_test[k]]
    if hustinx_path:
        # Compare against Hustinx (2023) image embeddings on the SAME records: map by image_id
        # and drop records not covered by the encodings. Force top-1/5/10.
        topks = tuple(sorted(set(topks) | {1, 5, 10}))
        hmap = load_hustinx_embeddings(hustinx_path)
        cov = lambda recs: [r for r in recs if r['image_id'] in hmap]
        before = len(all_recs)
        freq_gallery, freq_test = cov(freq_gallery), cov(freq_test)
        rare_gallery = {k: cov(rare_gallery[k]) for k in folds}
        rare_test = {k: cov(rare_test[k]) for k in folds}
        all_recs = freq_gallery + freq_test + [r for k in folds for r in rare_gallery[k] + rare_test[k]]
        print('[hustinx] records covered by encodings: %d/%d (frequent %d gal / %d test)' % (
            len(all_recs), before, len(freq_gallery), len(freq_test)), flush=True)
        emb = {r['path']: hmap[r['image_id']] for r in all_recs}
        emb_source = 'Hustinx 2023 embeddings'
    else:
        if tta:
            print('[tta] averaging views: %s' % ', '.join(['orig'] + list(tta)), flush=True)
        emb = embed(net, all_recs, transform, device, masked, landmarks_dir, landmark_img_size,
                    batch_size, clamp, tta=tta)
        emb_source = 'raw prototype presence'

    def score_all(score_fn):
        """Frequent single-split + Rare 10-fold-averaged retrieval for a scorer
        `score_fn(gallery, test) -> (res, C)`."""
        fs, fC = score_fn(freq_gallery, freq_test)
        folds_sc = [score_fn(rare_gallery[k], rare_test[k])[0]
                    for k in folds if rare_test[k]]
        rr = {k: {'macro': float(np.mean([f[k]['macro'] for f in folds_sc])),
                  'std':   float(np.std([f[k]['macro'] for f in folds_sc]))} for k in topks}
        return fs, fC, rr, len(folds_sc)

    def report(tag, fs, fC, rr, nfolds):
        lines = ['', '===== %s (mA_k as fraction & %%) =====' % tag,
                 'Frequent (seen)  C=%d n=%d : %s' % (fC, len(freq_test),
                   '  '.join('mA%d=%.4f (%.2f%%)' % (
                       k, fs[k]['macro'], 100 * fs[k]['macro']) for k in topks)),
                 'Rare (unseen, %d-fold)     : %s' % (nfolds,
                   '  '.join('mA%d=%.4f±%.4f (%.2f±%.2f%%)' % (
                       k, rr[k]['macro'], rr[k]['std'], 100 * rr[k]['macro'], 100 * rr[k]['std'])
                       for k in topks))]
        print('\n'.join(lines), flush=True)
        return lines

    results, txt = {}, ['GestaltMatcher-protocol retrieval validation (mA_k = class-mean top-k acc)']
    suffix = ' +csls=%d' % csls if csls > 0 else ''

    # Baseline: image-kNN cosine (optionally CSLS-ranked) on the chosen embeddings.
    base = score_all(lambda gal, tst: score(gal, tst, emb, topks, device, csls=csls))
    txt += report('Image-kNN cosine (%s)%s' % (emb_source, suffix), *base)
    results['frequent'], results['rare'] = base[0], base[2]

    # Optional: class-aggregated (nearest-class-mean) ranking — one score per syndrome,
    # removing the gallery-abundance bias of image-kNN.
    if class_aggregate and class_aggregate != 'none':
        agg = score_all(lambda gal, tst: score_class_aggregate(
            gal, tst, emb, topks, device, mode=class_aggregate, temp=agg_temp))
        txt += report('Class-aggregated (%s, one score/syndrome)%s' % (class_aggregate, suffix), *agg)
        results['frequent_agg'], results['rare_agg'] = agg[0], agg[2]

    # Optional: supervised-contrastive (NCE) projection head, trained on the frequent
    # gallery only, then applied to ALL embeddings and re-scored. Frequent measures the
    # learned metric on seen syndromes; Rare measures whether it transfers to unseen ones.
    if nce_head:
        path2lab = {r['path']: r['label'] for r in freq_gallery}
        paths = sorted(path2lab)
        X = torch.stack([emb[p] for p in paths])
        y = np.array([path2lab[p] for p in paths])
        print('\n[nce] training projection head (P=%d -> hidden=%d -> dim=%d) on %d '
              'frequent-gallery images / %d syndromes' % (
                  X.shape[1], nce_hidden, nce_dim, len(paths), len(set(y.tolist()))), flush=True)
        head = train_nce_head(X, y, device, hidden=nce_hidden, out_dim=nce_dim,
                              iters=nce_iters, lr=nce_lr, temperature=nce_temp)
        emb_h = apply_head(emb, head, device)
        nce = score_all(lambda gal, tst: score(gal, tst, emb_h, topks, device, csls=csls))
        txt += report('NCE projection head (trained on frequent gallery)%s' % suffix, *nce)
        results['frequent_nce'], results['rare_nce'] = nce[0], nce[2]
        os.makedirs(output_dir, exist_ok=True)
        torch.save(head.state_dict(), os.path.join(output_dir, 'nce_head.pt'))

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'gallery_validation_results.txt'), 'w') as f:
        f.write('\n'.join(txt) + '\n')
    print('\n[results] saved to %s' % output_dir, flush=True)
    return results


def gallery_validation_from_net(net, args, device):
    """Pipeline hook: evaluate the freshly trained in-memory net (main.py)."""
    return run_validation(
        net, device, net_name=args.net, image_size=args.image_size,
        masked=getattr(args, 'masked_pool', False),
        landmarks_dir=getattr(args, 'landmarks_dir', None),
        landmark_img_size=getattr(args, 'landmark_img_size', 112),
        freq_root=DEFAULT_FREQ_ROOT, rare_root=DEFAULT_RARE_ROOT, meta_dir=DEFAULT_META_DIR,
        output_dir=os.path.join(args.log_dir, 'gallery_validation'),
        batch_size=getattr(args, 'batch_size', 32),
        reduced_syndromes=getattr(args, 'reduced_syndromes', False),
        class_aggregate=getattr(args, 'class_aggregate', 'none'),
        agg_temp=getattr(args, 'agg_temp', 0.1),
        tta=tuple(getattr(args, 'tta', ()) or ()),
        csls=getattr(args, 'csls', 0),
        nce_head=getattr(args, 'nce_head', False),
        nce_dim=getattr(args, 'nce_dim', 128), nce_hidden=getattr(args, 'nce_hidden', 512),
        nce_iters=getattr(args, 'nce_iters', 300), nce_lr=getattr(args, 'nce_lr', 1e-3),
        nce_temp=getattr(args, 'nce_temp', 0.1))


def main():
    p = argparse.ArgumentParser(description='GestaltMatcher-protocol retrieval validation for PIP-Net.')
    p.add_argument('--model_dir', default=DEFAULT_MODEL_DIR)
    p.add_argument('--checkpoint', default='net_trained')
    p.add_argument('--freq_root', default=DEFAULT_FREQ_ROOT)
    p.add_argument('--rare_root', default=DEFAULT_RARE_ROOT)
    p.add_argument('--meta_dir', default=DEFAULT_META_DIR)
    p.add_argument('--landmarks_dir', default=None, help='Default: the run\'s training landmarks_dir.')
    p.add_argument('--output_dir', default=None, help='Default: <model_dir>/gallery_validation.')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--clamp', action='store_true', help='Use the test-time clamped presence vector.')
    p.add_argument('--topk', default='1,5,10,30')
    p.add_argument('--reduced_syndromes', action='store_true',
                   help='Restrict accuracy to the most-populated syndromes: top-%d frequent and '
                        'top-%d rare by distinct image count.' % (REDUCED_FREQ_SYNDROMES, REDUCED_RARE_SYNDROMES))
    p.add_argument('--class_aggregate', choices=['none', 'mean', 'max', 'softmax'], default='mean',
                   help='Also report class-aggregated (nearest-class-mean) ranking: one score per '
                        'syndrome, removing image-kNN abundance bias. mean=cosine-to-centroid, '
                        'max=per-class NN, softmax=temp-weighted smooth-max.')
    p.add_argument('--agg_temp', type=float, default=0.1, help='Temperature for --class_aggregate softmax.')
    p.add_argument('--tta', default='', help='Comma-separated test-time-augmentation views to average '
                   'with the original, from {hflip,gray}. E.g. --tta hflip,gray. Empty = no TTA.')
    p.add_argument('--csls', type=int, default=0,
                   help='CSLS hubness correction: rank by 2*sim - r_S, r_S = mean of each gallery '
                        'item\'s top-N similarities to queries. 0 = plain cosine. Try 5-10.')
    p.add_argument('--nce_head', action='store_true',
                   help='Train a supervised-contrastive projection head on the frequent gallery and '
                        'report retrieval through it (alongside the raw-cosine baseline).')
    p.add_argument('--nce_dim', type=int, default=128, help='Output embedding dim of the NCE head.')
    p.add_argument('--nce_hidden', type=int, default=512,
                   help='Hidden width of the NCE head (0 = linear head, no nonlinearity).')
    p.add_argument('--nce_iters', type=int, default=300, help='SupCon training iterations.')
    p.add_argument('--nce_lr', type=float, default=1e-3, help='Adam LR for the NCE head.')
    p.add_argument('--nce_temp', type=float, default=0.1, help='SupCon temperature.')
    p.add_argument('--hustinx_encodings', default=None,
                   help='Path to Hustinx (2023) all_encodings.csv. When set, retrieval uses those '
                        'image embeddings (aggregated per image_id) instead of PIP-Net, on the same '
                        'gallery/test records — for a like-for-like comparison. Forces top-1/5/10. '
                        'No model is loaded.')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    device = torch.device(args.device)
    if args.hustinx_encodings:
        net = None  # external embeddings; no PIP-Net needed
        margs = argparse.Namespace(net='resnet50', image_size=112, masked_pool=False,
                                   landmarks_dir=None, landmark_img_size=112)
    else:
        net, margs = load_pipnet(args.model_dir, args.checkpoint, device)
    run_validation(
        net, device, net_name=margs.net, image_size=margs.image_size,
        masked=getattr(margs, 'masked_pool', False),
        landmarks_dir=args.landmarks_dir or getattr(margs, 'landmarks_dir', None),
        landmark_img_size=getattr(margs, 'landmark_img_size', 112),
        freq_root=args.freq_root, rare_root=args.rare_root, meta_dir=args.meta_dir,
        output_dir=args.output_dir or os.path.join(args.model_dir, 'gallery_validation'),
        batch_size=args.batch_size, clamp=args.clamp,
        topks=tuple(int(x) for x in args.topk.split(',')),
        reduced_syndromes=args.reduced_syndromes,
        class_aggregate=args.class_aggregate, agg_temp=args.agg_temp,
        tta=tuple(v.strip() for v in args.tta.split(',') if v.strip()),
        csls=args.csls,
        nce_head=args.nce_head, nce_dim=args.nce_dim, nce_hidden=args.nce_hidden,
        nce_iters=args.nce_iters, nce_lr=args.nce_lr, nce_temp=args.nce_temp,
        hustinx_path=args.hustinx_encodings)


if __name__ == '__main__':
    main()
