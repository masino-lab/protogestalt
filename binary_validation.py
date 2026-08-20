"""
Binary validation for PIP-Net on GMDB: NORMAL (non-syndromic faces) vs RARE/abnormal
(GMDB syndromic faces), using the SAME trained prototype-presence vectors as
gallery_validation.py (masked-pooled, L2-normalized; we reuse its `embed`).

A logistic-regression probe is trained on the prototype vectors to separate normal from
rare. This is run across THREE normal datasets, and the results for all three are written
to the SAME output files:

  utkface               : /data2/masino_lab/abamini/facial_data/UTKFace_align  (+ _landmarks)
  utkface_case_controls : /data2/masino_lab/abamini/facial_data/UTKFace_align_case_controls
                          (+ _case_controls_landmarks) -- age/sex/ethnicity-matched to the
                          GMDB test set, so demographics are controlled.
  casia_webface         : /data2/masino_lab/abamini/facial_data/casia_webface_align (+ _landmarks)

Positive / syndromic (label 1): GMDB-FREQUENT TRAIN supplies the TRAINING positives
(--n_train_pos, default 1000). GMDB-frequent TEST and GMDB-RARE are NEVER used for training,
so evaluation is patient-disjoint from training by the original GMDB split. The TEST positives
are ALL of GMDB-frequent test PLUS all of GMDB-rare (gallery + test). Normal (label 0): images
from the dataset above (identity-disjoint splits; CASIA identity = filename prefix, UTKFace
= per-image).

Training normals come from utkface and casia_webface ONLY: one LR per training dataset, each on
1000 GMDB-frequent-TRAIN positives + 1000 of that dataset's normals (balanced, 2000 images).
Each model is then tested on (all freq test + all rare) positives + normals:

  utkface               -> utkface-trained model   (normals = its identity-disjoint remainder)
  casia_webface         -> casia-trained model     (normals = its identity-disjoint remainder)
  utkface_case_controls -> utkface-trained model   (it is a SUBSET of utkface, so its images are
                           held out of the utkface training pool -- otherwise ~4% of this test
                           set would have been trained on)

The usual classification metrics are saved (balanced test where enough normals are available);
the `trained_on` column records which model produced each row. Then, for utkface and
casia_webface only, we sweep the number of normal images in the TEST set (model fixed) and plot
how AUROC/AUPRC change as test imbalance grows.

Caveat (documented on purpose): "rare vs normal" can mix syndrome signal with dataset/domain
signal (different crops/capture/demographics). The utkface_case_controls set is the
demographics-controlled comparison; a large drop from casia/utkface -> case_controls would
indicate how much AUROC was driven by demographic/domain separability rather than morphology.

Run:  python binary_validation.py --model_dir <run_dir> --increments 1,2,4,8,16
"""

import os
import sys
import csv
import random
import argparse
from collections import defaultdict

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                             precision_score, recall_score, f1_score, confusion_matrix)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gmdb_common import (
    load_pipnet, build_transform, build_image_index, resolve_official, embed,
    DEFAULT_FREQ_ROOT, DEFAULT_RARE_ROOT, DEFAULT_META_DIR, DEFAULT_MODEL_DIR,
    CSV_RARE_GALLERY, CSV_RARE_TEST, CSV_FREQ_TEST, IMG_EXTS)

# GMDB frequent TRAIN split -> supplies the training positives (patient-disjoint from
# frequent test / rare, which are used only for evaluation).
CSV_GMDB_TRAIN = 'gmdb_train_images_v1.1.0.csv'

# name -> config. 'id' picks how a normal identity is derived (for identity-disjoint splits):
#   'prefix' = filename up to first '_' (CASIA); 'image' = the whole file (UTKFace, ~unique).
# 'sweep' = run the grow-normal-test-images sweep for this dataset.
# 'train_with' = which dataset's normals train the model used to TEST this dataset. Only
# utkface and casia_webface supply training normals; utkface_case_controls is a SUBSET of
# utkface, so it is tested with the utkface-trained model (and its images are held out of
# that model's training pool to avoid leakage).
DATASETS = {
    'utkface': dict(
        img='/data2/masino_lab/abamini/facial_data/UTKFace_align',
        lmk='/data2/masino_lab/abamini/facial_data/UTKFace_align_landmarks',
        id='image', sweep=True, train_with='utkface'),
    'utkface_case_controls': dict(
        img='/data2/masino_lab/abamini/facial_data/UTKFace_align_case_controls',
        lmk='/data2/masino_lab/abamini/facial_data/UTKFace_align_case_controls_landmarks',
        id='image', sweep=False, train_with='utkface'),
    'casia_webface': dict(
        img='/data2/masino_lab/abamini/facial_data/casia_webface_align',
        lmk='/data2/masino_lab/abamini/facial_data/casia_webface_align_landmarks',
        id='prefix', sweep=True, train_with='casia_webface'),
}


# --- data collection -------------------------------------------------------
def collect_normal(root, pool, seed, id_mode):
    """Normal records {path, identity, label=0}. Identities are shuffled and taken until
    `pool` images are gathered, so the working set is capped before the (expensive)
    embedding step. id_mode='prefix' -> identity = filename before first '_' (CASIA, many
    images per identity); id_mode='image' -> identity = filename (UTKFace, ~one per person)."""
    by_id = defaultdict(list)
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in IMG_EXTS:
                ident = fn.split('_')[0] if id_mode == 'prefix' else fn
                by_id[ident].append(os.path.join(dirpath, fn))
    ids = list(by_id)
    random.Random(seed).shuffle(ids)
    recs = []
    for ident in ids:
        for p in by_id[ident]:
            recs.append({'path': p, 'identity': ident, 'label': 0})
        if len(recs) >= pool:
            break
    return recs


def image_basenames(root):
    """Set of image filenames under `root` (used to detect subset overlap between datasets)."""
    out = set()
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in IMG_EXTS:
                out.add(fn)
    return out


def collect_gmdb_csv(csv_name, index, meta_dir):
    """Positive records {path, image_id, label=1} from one GMDB CSV, deduped by image_id.
    Used for the frequent TRAIN split (training positives) and the frequent TEST split
    (evaluation positives)."""
    import pandas as pd
    df = pd.read_csv(os.path.join(meta_dir, csv_name))
    recs = resolve_official(df, index)
    by_id = {}
    for r in recs:
        by_id.setdefault(r['image_id'], r)
    pos = list(by_id.values())
    for r in pos:
        r['label'] = 1
    return pos


def collect_rare(rare_index, meta_dir):
    """Positive records {path, image_id, label=1} from GMDB-RARE (gallery + test union,
    deduped). These are used for TESTING ONLY -- never for training."""
    import pandas as pd
    recs = []
    for csv_name in (CSV_RARE_GALLERY, CSV_RARE_TEST):
        df = pd.read_csv(os.path.join(meta_dir, csv_name))
        recs += resolve_official(df, rare_index)
    by_id = {}
    for r in recs:
        by_id.setdefault(r['image_id'], r)
    rare = list(by_id.values())
    for r in rare:
        r['label'] = 1
    return rare


# --- splitting -------------------------------------------------------------
def take_group_count(records, n_target, seed):
    """Identity-disjoint split by COUNT: consume whole identities until >= n_target images,
    then trim to EXACTLY n_target (so the training set is the requested size, e.g. a balanced
    1000). Surplus images of the last identity are dropped rather than moved to `rest` --
    keeping the identity out of the test pool matters more than reusing a few images. `rest`
    therefore contains only identities never seen in training."""
    groups = defaultdict(list)
    for r in records:
        groups[r['identity']].append(r)
    names = list(groups)
    random.Random(seed).shuffle(names)
    taken, rest = [], []
    for g in names:
        (taken if len(taken) < n_target else rest).extend(groups[g])
    return taken[:n_target], rest


def to_matrix(records, emb):
    """Stack the cached embeddings of `records` into a float32 (N, P) array."""
    return np.stack([emb[r['path']].numpy() for r in records]).astype(np.float32)


# --- metrics ---------------------------------------------------------------
def classification_metrics(y_true, scores, thr=0.5):
    """Usual binary metrics at `thr` (positive class = rare/1), plus threshold-free AUROC/AUPRC."""
    y_pred = (scores >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) else float('nan')
    return {
        'accuracy':  accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall':    recall_score(y_true, y_pred, zero_division=0),   # sensitivity
        'specificity': spec,
        'f1':        f1_score(y_true, y_pred, zero_division=0),
        'auroc':     roc_auc_score(y_true, scores),
        'auprc':     average_precision_score(y_true, scores),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
    }


# --- sweep -----------------------------------------------------------------
def sweep_test_normals(clf, emb, rare_test, normal_test_pool, mults, seeds, base):
    """Train fixed; grow #normal TEST images (rare-test positives fixed). Returns lists of
    (n_normal_test, prevalence, (auroc_mean,std), (auprc_mean,std)). base = balanced test size
    (= len(rare_test)); m=1 is the class-balanced test point. AUROC should stay ~flat
    (prevalence-invariant) while AUPRC falls as negatives grow."""
    X_rare_test = to_matrix(rare_test, emb)
    s_rare = clf.predict_proba(X_rare_test)[:, 1]
    rows = []
    for m in mults:
        n_te = min(base * m, len(normal_test_pool))
        aus, aps = [], []
        for s in range(seeds):
            te_norm = random.Random(2000 + s).sample(normal_test_pool, n_te)
            s_norm = clf.predict_proba(to_matrix(te_norm, emb))[:, 1]
            sc = np.concatenate([s_rare, s_norm])
            yte = np.array([1] * len(rare_test) + [0] * n_te)
            aus.append(roc_auc_score(yte, sc))
            aps.append(average_precision_score(yte, sc))
        prev = len(rare_test) / (len(rare_test) + n_te)
        rows.append((n_te, prev,
                     (float(np.mean(aus)), float(np.std(aus))),
                     (float(np.mean(aps)), float(np.std(aps)))))
        print('  n_normal_test=%-6d (rare prevalence=%.3f) AUROC=%.4f+/-%.4f  AUPRC=%.4f+/-%.4f'
              % (n_te, prev, rows[-1][2][0], rows[-1][2][1], rows[-1][3][0], rows[-1][3][1]),
              flush=True)
        if n_te == len(normal_test_pool):
            break
    return rows


# --- dataset-identity confound probe ---------------------------------------
def dataset_identity_probe(models, emb, seed, balanced, n_test=1000):
    """Confound check: can a logistic regression tell UTKFace from CASIA-WebFace using the SAME
    prototype embeddings the screening model uses? Trains on each dataset's (identity-disjoint)
    training normals -- utkface=0, casia_webface=1 -- and tests on their held-out pools. A HIGH
    AUROC means dataset identity is highly separable in prototype space, so the normal-vs-rare
    ('screening') signal may partly reflect dataset/domain differences rather than syndromic
    morphology; an AUROC near 0.5 argues the screening signal is not driven by dataset identity."""
    if 'utkface' not in models or 'casia_webface' not in models:
        print('\n[dataset probe] needs both utkface and casia_webface trained; skipping.', flush=True)
        return None
    utk, cas = models['utkface'], models['casia_webface']
    X_tr = np.concatenate([to_matrix(utk['norm_tr'], emb), to_matrix(cas['norm_tr'], emb)])
    y_tr = np.array([0] * len(utk['norm_tr']) + [1] * len(cas['norm_tr']))
    clf = LogisticRegression(max_iter=2000,
                             class_weight='balanced' if balanced else None).fit(X_tr, y_tr)
    nu, nc = min(n_test, len(utk['norm_pool'])), min(n_test, len(cas['norm_pool']))
    te_u = random.Random(seed).sample(utk['norm_pool'], nu)
    te_c = random.Random(seed + 1).sample(cas['norm_pool'], nc)
    X_te = np.concatenate([to_matrix(te_u, emb), to_matrix(te_c, emb)])
    y_te = np.array([0] * nu + [1] * nc)
    scores = clf.predict_proba(X_te)[:, 1]
    auroc = roc_auc_score(y_te, scores)
    acc = accuracy_score(y_te, (scores >= 0.5).astype(int))
    print('\n' + '=' * 70, flush=True)
    print('[dataset identity probe] UTKFace(0) vs CASIA-WebFace(1) on prototype embeddings', flush=True)
    print('  train: %d UTKFace + %d CASIA | test: %d UTKFace + %d CASIA'
          % (len(utk['norm_tr']), len(cas['norm_tr']), nu, nc), flush=True)
    print('  AUROC=%.4f  accuracy=%.4f' % (auroc, acc), flush=True)
    verdict = ('HIGH separability -> dataset identity is a likely confound in the screening signal'
               if auroc > 0.8 else
               'low separability -> dataset identity is unlikely to dominate the screening signal')
    print('  => %s' % verdict, flush=True)
    return dict(auroc=float(auroc), accuracy=float(acc),
                n_train=len(utk['norm_tr']) + len(cas['norm_tr']), n_test=nu + nc)


# --- output ----------------------------------------------------------------
METRIC_FIELDS = ['dataset', 'trained_on', 'n_train_pos', 'n_train_normal',
                 'n_test_pos', 'n_test_pos_freq', 'n_test_pos_rare', 'n_test_normal',
                 'accuracy', 'precision', 'recall', 'specificity', 'f1', 'auroc', 'auprc',
                 'tn', 'fp', 'fn', 'tp']


def write_metrics_csv(rows, path):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in METRIC_FIELDS})


def write_sweep_csv(sweeps, path):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['dataset', 'n_normal_test', 'rare_prevalence',
                    'auroc_mean', 'auroc_std', 'auprc_mean', 'auprc_std'])
        for ds, rows in sweeps.items():
            for n_te, prev, (am, asd), (pm, psd) in rows:
                w.writerow([ds, n_te, '%.4f' % prev, am, asd, pm, psd])


def plot_sweep_metric(sweeps, path, metric):
    """One figure for ONE metric ('AUROC' or 'AUPRC'), all datasets overlaid, vs the PERCENTAGE
    of non-syndromic images in the test set. AUROC and AUPRC get separate files."""
    idx = 2 if metric == 'AUROC' else 3
    marker = 'o' if metric == 'AUROC' else 's'
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for ds, rows in sweeps.items():
        # rows are (n_normal_test, rare_prevalence, ...); non-syndromic % = (1 - prevalence)*100
        xs = [(1.0 - r[1]) * 100.0 for r in rows]
        ax.errorbar(xs, [r[idx][0] for r in rows], yerr=[r[idx][1] for r in rows],
                    marker=marker, capsize=3, linewidth=2, label=ds)
    ax.set_xlabel('Percentage of Non-syndromic images (Test Set)', fontsize=13, fontweight='bold')
    ax.set_ylabel('%s (positive class = rare)' % metric, fontsize=13, fontweight='bold')
    ax.set_title('%s vs %% Non-syndromic images' % metric, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.02)
    ax.tick_params(labelsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- driver ----------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description='NORMAL-vs-RARE binary validation for PIP-Net '
                                            'across 3 normal datasets (results in shared files).')
    p.add_argument('--model_dir', default=DEFAULT_MODEL_DIR)
    p.add_argument('--checkpoint', default='net_trained')
    p.add_argument('--rare_root', default=DEFAULT_RARE_ROOT)
    p.add_argument('--freq_root', default=DEFAULT_FREQ_ROOT)
    p.add_argument('--meta_dir', default=DEFAULT_META_DIR)
    p.add_argument('--output_dir', default=None, help='Default: <model_dir>/binary_validation.')
    p.add_argument('--datasets', default=','.join(DATASETS),
                   help='Comma-separated subset of: ' + ', '.join(DATASETS))
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--clamp', action='store_true', help='Use the test-time clamped presence vector.')
    p.add_argument('--tta', default='', help='TTA views to average, from {hflip,gray}.')
    p.add_argument('--normal_pool', type=int, default=35000,
                   help='Max normal images to load/embed per dataset (capped before embedding).')
    p.add_argument('--n_train_pos', type=int, default=200,
                   help='Number of GMDB-frequent-TRAIN images used as TRAINING positives.')
    p.add_argument('--n_train_normal', type=int, default=200,
                   help='Number of normal images used as TRAINING negatives (balanced by default).')
    p.add_argument('--increments', default='1,2,4,8,16,32',
                   help='Sweep multipliers on #test positives (m=1 = balanced test).')
    p.add_argument('--seeds', type=int, default=3, help='Random test-normal draws per increment.')
    p.add_argument('--balanced', action='store_true',
                   help="Use class_weight='balanced' in logistic regression.")
    p.add_argument('--seed', type=int, default=5, help='Seed for data collection / splits.')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    device = torch.device(args.device)
    net, margs = load_pipnet(args.model_dir, args.checkpoint, device)
    transform = build_transform(margs.net, margs.image_size)
    masked = getattr(margs, 'masked_pool', False)
    landmark_img_size = getattr(margs, 'landmark_img_size', 112)
    gmdb_lmk = getattr(margs, 'landmarks_dir', None)
    tta = tuple(v.strip() for v in args.tta.split(',') if v.strip())
    output_dir = args.output_dir or os.path.join(args.model_dir, 'binary_validation')
    os.makedirs(output_dir, exist_ok=True)
    mults = [int(x) for x in args.increments.split(',')]
    names = [n.strip() for n in args.datasets.split(',') if n.strip()]
    for n in names:
        if n not in DATASETS:
            raise SystemExit('[error] unknown dataset %r; choose from %s' % (n, ', '.join(DATASETS)))

    # Positive sets + embeddings are shared across datasets (positives use GMDB landmarks), so
    # embed once. GMDB-frequent TRAIN supplies the training positives; ALL of frequent TEST plus
    # ALL of GMDB-rare are the test positives (patient-disjoint from training by the GMDB split).
    freq_index = build_image_index(args.freq_root)
    rare_index = build_image_index(args.rare_root)
    freq_train_pool = collect_gmdb_csv(CSV_GMDB_TRAIN, freq_index, args.meta_dir)
    freq_test = collect_gmdb_csv(CSV_FREQ_TEST, freq_index, args.meta_dir)
    rare = collect_rare(rare_index, args.meta_dir)
    # sample the training positives BEFORE embedding so we only encode what we use
    n_pos = min(args.n_train_pos, len(freq_train_pool))
    pos_train = random.Random(args.seed).sample(freq_train_pool, n_pos)
    pos_test = freq_test + rare
    print('[data] freq-train pool=%d -> train positives=%d | test positives=%d (%d freq-test + %d rare)'
          % (len(freq_train_pool), len(pos_train), len(pos_test), len(freq_test), len(rare)),
          flush=True)
    if tta:
        print('[tta] averaging views: %s' % ', '.join(['orig'] + list(tta)), flush=True)
    emb = embed(net, pos_train + pos_test, transform, device, masked,
                [gmdb_lmk] if gmdb_lmk else None,
                landmark_img_size, args.batch_size, args.clamp, tta=tta)

    X_pos_train = to_matrix(pos_train, emb)
    X_pos_test = to_matrix(pos_test, emb)

    # --- train one model per TRAINING dataset (utkface, casia_webface) -------------------
    # utkface_case_controls supplies no training normals: it is a subset of utkface and is
    # tested with the utkface-trained model.
    train_names = []
    for n in names:
        tw = DATASETS[n]['train_with']
        if tw not in train_names:
            train_names.append(tw)

    models = {}
    for tname in train_names:
        tcfg = DATASETS[tname]
        print('\n' + '=' * 70 + '\n[train model] normals from %s  (%s)' % (tname, tcfg['img']),
              flush=True)
        normal = collect_normal(tcfg['img'], args.normal_pool, args.seed, tcfg['id'])
        # Hold out images of any dataset that is a SUBSET of this one and will be tested with
        # this model (case_controls c utkface): otherwise ~4% of that test set would be trained on.
        subs = [d for d in names if d != tname and DATASETS[d]['train_with'] == tname]
        excl = set()
        for d in subs:
            excl |= image_basenames(DATASETS[d]['img'])
        if excl:
            before = len(normal)
            normal = [r for r in normal if os.path.basename(r['path']) not in excl]
            print('[train model] held out %d images also in %s (no test leakage)'
                  % (before - len(normal), ', '.join(subs)), flush=True)
        emb.update(embed(net, normal, transform, device, masked,
                         [tcfg['lmk']], landmark_img_size, args.batch_size, args.clamp, tta=tta))
        norm_tr, norm_pool = take_group_count(normal, args.n_train_normal, args.seed)
        X_tr = np.concatenate([X_pos_train, to_matrix(norm_tr, emb)])
        y_tr = np.array([1] * len(pos_train) + [0] * len(norm_tr))
        clf = LogisticRegression(max_iter=2000,
                                 class_weight='balanced' if args.balanced else None).fit(X_tr, y_tr)
        models[tname] = dict(clf=clf, norm_tr=norm_tr, norm_pool=norm_pool)
        print('[train model] pos=%d + normal=%d | held-out normal pool=%d'
              % (len(pos_train), len(norm_tr), len(norm_pool)), flush=True)

    # --- evaluate each test dataset with its assigned model -----------------------------
    metric_rows, sweeps = [], {}
    for name in names:
        cfg = DATASETS[name]
        tname = cfg['train_with']
        model = models[tname]
        clf, norm_tr = model['clf'], model['norm_tr']
        print('\n' + '=' * 70 + '\n[dataset] %s  (%s)\n[model] trained on %s normals'
              % (name, cfg['img'], tname), flush=True)
        if name == tname:
            norm_pool = model['norm_pool']       # identity-disjoint remainder of the train set
        else:
            normal = collect_normal(cfg['img'], args.normal_pool, args.seed, cfg['id'])
            emb.update(embed(net, normal, transform, device, masked,
                             [cfg['lmk']], landmark_img_size, args.batch_size, args.clamp, tta=tta))
            norm_pool = normal                   # disjoint by construction (held out above)
        print('[test] normal pool=%d' % len(norm_pool), flush=True)

        # balanced test where possible: #normal test = min(#pos test, available normal pool)
        n_norm_test = min(len(pos_test), len(norm_pool))
        norm_te = random.Random(args.seed).sample(norm_pool, n_norm_test)
        X_te = np.concatenate([X_pos_test, to_matrix(norm_te, emb)])
        y_te = np.array([1] * len(pos_test) + [0] * n_norm_test)
        s_te = clf.predict_proba(X_te)[:, 1]
        m = classification_metrics(y_te, s_te)
        m.update(dataset=name, trained_on=tname,
                 n_train_pos=len(pos_train), n_train_normal=len(norm_tr),
                 n_test_pos=len(pos_test), n_test_pos_freq=len(freq_test), n_test_pos_rare=len(rare),
                 n_test_normal=n_norm_test)
        metric_rows.append(m)
        print('[metrics] acc=%.4f prec=%.4f rec=%.4f spec=%.4f f1=%.4f AUROC=%.4f AUPRC=%.4f'
              % (m['accuracy'], m['precision'], m['recall'], m['specificity'], m['f1'],
                 m['auroc'], m['auprc']), flush=True)

        # sweep: grow #normal TEST images (train fixed), for flagged datasets only
        if cfg['sweep']:
            print('[sweep] grow #normal TEST images; train fixed (pos=%d + normal=%d)'
                  % (len(pos_train), len(norm_tr)), flush=True)
            sweeps[name] = sweep_test_normals(clf, emb, pos_test, norm_pool, mults, args.seeds,
                                              base=len(pos_test))

    # dataset-identity confound probe (UTKFace vs CASIA on the same prototype embeddings)
    probe = dataset_identity_probe(models, emb, args.seed, args.balanced, n_test=args.n_train_normal)

    # write shared outputs
    mpath = os.path.join(output_dir, 'binary_validation_metrics.csv')
    write_metrics_csv(metric_rows, mpath)
    print('\n[results] classification metrics -> %s' % mpath, flush=True)
    if probe is not None:
        ppath = os.path.join(output_dir, 'dataset_identity_probe.csv')
        with open(ppath, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['task', 'auroc', 'accuracy', 'n_train', 'n_test'])
            w.writerow(['utkface_vs_casia', '%.4f' % probe['auroc'], '%.4f' % probe['accuracy'],
                        probe['n_train'], probe['n_test']])
        print('[results] dataset-identity probe -> %s' % ppath, flush=True)
    if sweeps:
        spath = os.path.join(output_dir, 'binary_validation_test_normals_sweep.csv')
        roc_path = os.path.join(output_dir, 'binary_validation_test_normals_auroc.png')
        prc_path = os.path.join(output_dir, 'binary_validation_test_normals_auprc.png')
        write_sweep_csv(sweeps, spath)
        plot_sweep_metric(sweeps, roc_path, 'AUROC')
        plot_sweep_metric(sweeps, prc_path, 'AUPRC')
        print('[results] test-normals sweep -> %s\n           AUROC plot -> %s\n'
              '           AUPRC plot -> %s' % (spath, roc_path, prc_path), flush=True)


if __name__ == '__main__':
    main()
