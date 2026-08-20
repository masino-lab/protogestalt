"""Shared GMDB + PIP-Net I/O helpers used by all evaluation/analysis scripts.

This module is the single source of the pieces that every tool needs — model loading, the
no-augmentation transform, image-id resolution, and the prototype-presence embedding — so the
scripts (gallery_validation, syndrome_prediction, binary_validation, explain_prediction,
prototype_consistency, prototype_purity) can each run standalone without importing one another.
"""
import os
import sys
import pickle
from collections import defaultdict

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipnet.pipnet import get_network, PIPNet
from util.data import _get_normalization, landmark_path_for_image, load_landmarks, NUM_LANDMARKS

# --- defaults (override on the CLI) ----------------------------------------
# Frequent images live in the GMDB ImageFolder splits: <FREQ_ROOT>/gmdb_{train,test,val}/<syndrome>/<id>.jpg
DEFAULT_FREQ_ROOT = '/data2/masino_lab/abamini/facial_data/Final_data'
# Rare images are extracted from the whole GMDB crops dataset (flat dir of <id>_aligned.jpg).
DEFAULT_RARE_ROOT = '/data2/masino_lab/resources/gmdb/gmdb_crops'
DEFAULT_META_DIR  = '/home/abamini/GMDB_data/gmdb_metadata'
DEFAULT_MODEL_DIR = '/data2/masino_lab/abamini/pipnet/past_runs/run_16_all_gmdb_images_fixed_protoypes_data_fixed/'
CSV_FREQ_GALLERY  = 'gmdb_frequent_gallery_images_v1.1.0.csv'  # frequent (seen), single split
CSV_FREQ_TEST     = 'gmdb_frequent_test_images_v1.1.0.csv'
CSV_RARE_GALLERY  = 'gmdb_rare_gallery_images_v1.1.0.csv'   # has the 10-fold `split` column
CSV_RARE_TEST     = 'gmdb_rare_test_images_v1.1.0.csv'
IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')

# --reduced_syndromes restricts accuracy to the most-populated syndromes:
# the top-204 frequent and top-245 rare syndromes by distinct image count.
REDUCED_FREQ_SYNDROMES = 204
REDUCED_RARE_SYNDROMES = 245


def load_pipnet(model_dir, checkpoint, device):
    """Rebuild PIP-Net from a run's saved args and load its weights."""
    with open(os.path.join(model_dir, 'metadata', 'args.pickle'), 'rb') as f:
        margs = pickle.load(f)
    margs.disable_pretrained = True  # weights come from the checkpoint; skip the backbone download

    ckpt = torch.load(os.path.join(model_dir, 'checkpoints', checkpoint), map_location=device)
    sd = ckpt.get('model_state_dict', ckpt)
    sd = {k[len('module.'):] if k.startswith('module.') else k: v for k, v in sd.items()}

    num_classes = sd['_classification.weight'].shape[0]
    feats, add_on, pool, clf, num_prototypes = get_network(num_classes, margs)
    net = PIPNet(num_classes, num_prototypes, feats, margs, add_on, pool, clf)
    net.load_state_dict(sd, strict=False)
    net.to(device).eval()
    print('[load] %s: %d classes, %d prototypes, masked_pool=%s r=%s' % (
        checkpoint, num_classes, num_prototypes,
        getattr(margs, 'masked_pool', False), getattr(margs, 'mask_radius', None)), flush=True)
    return net, margs


def build_transform(net_name, image_size):
    """No-augmentation, geometry-preserving transform (keeps landmarks valid)."""
    mean, std = _get_normalization(net_name)
    return transforms.Compose([transforms.Resize((image_size, image_size)),
                               transforms.ToTensor(), transforms.Normalize(mean, std)])


# --- collecting images -----------------------------------------------------
def _image_id_from_stem(stem):
    """Image id from a filename stem. Frequent crops are '<id>'; the aligned dataset
    uses '<id>_aligned_aligned'. Both reduce to the leading numeric token."""
    core = stem.split('_')[0]
    return core if core.isdigit() else None


def build_image_index(root):
    """image_id (str) -> on-disk path, scanning `root` recursively. Handles both the
    frequent ImageFolder layout (<root>/gmdb_{train,test,val}/<syndrome>/<id>.jpg) and
    the flat aligned dataset (<root>/<id>_aligned_aligned.jpg) from which rare images
    are extracted. First match wins (GMDB image_ids are globally unique)."""
    index = {}
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() not in IMG_EXTS:
                continue
            key = _image_id_from_stem(os.path.splitext(fn)[0])
            if key is not None:
                index.setdefault(key, os.path.join(dirpath, fn))
    return index


def resolve_official(df, index):
    """Map official-CSV rows (image_id, label) to on-disk records via a prebuilt
    image_id->path index; skip rows whose image is not on disk."""
    records = []
    for r in df.itertuples(index=False):
        path = index.get(str(int(r.image_id)))
        if path is not None:
            records.append({'path': path, 'image_id': int(r.image_id), 'label': int(r.label)})
    return records


def top_syndromes_by_image_count(records, n):
    """Set of the `n` syndromes (labels) with the most *distinct* images across `records`.

    Counts unique image_ids per label, not rows: the rare 10-fold CSVs repeat each
    image across folds, so row counts would overcount. Ties are broken by label id so
    the selection is deterministic across runs."""
    imgs = defaultdict(set)
    for r in records:
        imgs[r['label']].add(r['image_id'])
    ranked = sorted(imgs.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    return {lbl for lbl, _ in ranked[:n]}


# --- embedding -------------------------------------------------------------
def _hflip_landmarks(lms):
    """Mirror normalized landmark columns (col -> 1-col) for a horizontally flipped
    image so region-masked pooling stays anatomically aligned. Missing (-1) entries
    are left untouched. lms: (B, L, 2) as (row, col)."""
    out = lms.clone()
    valid = lms[..., 1] >= 0
    out[..., 1] = torch.where(valid, 1.0 - lms[..., 1], lms[..., 1])
    return out


@torch.no_grad()
def embed(net, records, transform, device, masked, landmarks_dir, landmark_img_size,
          batch_size=32, clamp=False, tta=()):
    """L2-normalized prototype-presence embedding per unique image path -> {path: tensor}.

    Test-time augmentation: with `tta` a subset of {'hflip','gray'}, the embedding is the
    L2-normalized mean of the original view and each requested view. 'hflip' mirrors the
    image and its landmark columns (so masked pooling stays aligned); 'gray' converts to
    grayscale (3 channels, geometry/landmarks unchanged)."""
    views = ['orig'] + [v for v in ('hflip', 'gray') if v in tta]
    to_gray = transforms.Grayscale(num_output_channels=3)
    paths = sorted({r['path'] for r in records})
    out = {}
    desc = 'embed %d imgs%s' % (len(paths), '' if len(views) == 1 else ' x%d views' % len(views))
    for i in tqdm(range(0, len(paths), batch_size), desc=desc, ncols=0):
        chunk = paths[i:i + batch_size]
        pil = [Image.open(p).convert('RGB') for p in chunk]
        base_lms = None
        if masked:
            base_lms = torch.stack([torch.from_numpy(load_landmarks(
                landmark_path_for_image(p, landmarks_dir), landmark_img_size, NUM_LANDMARKS))
                for p in chunk]).to(device)
        acc = None
        for v in views:
            if v == 'hflip':
                imgs = torch.stack([transform(im.transpose(Image.FLIP_LEFT_RIGHT)) for im in pil])
                lms = _hflip_landmarks(base_lms) if masked else None
            elif v == 'gray':
                imgs = torch.stack([transform(to_gray(im)) for im in pil])
                lms = base_lms
            else:
                imgs = torch.stack([transform(im) for im in pil])
                lms = base_lms
            _, pooled, _ = net(imgs.to(device), inference=clamp, lms=lms)
            e = F.normalize(pooled.float(), dim=1)
            acc = e if acc is None else acc + e
        acc = F.normalize(acc / len(views), dim=1).cpu()   # mean of unit views, renormalized
        for p, e in zip(chunk, acc):
            out[p] = e
    return out
