"""Local explanation for ONE image with a trained PIP-Net: classify it and visualize the
prototypes that drove the decision.

For the predicted class c, each prototype d contributes  contrib_d = g_d * w_{c,d}  (its
presence times its non-negative class weight) — the same quantity PIP-Net sums to score the
class. We rank prototypes by contrib_d and, for the top ones, show WHERE each fires (the
max-activation patch / box / heatmap) and its presence g_d and weight w_{c,d}.

Usage:
  python explain_prediction.py --model_dir <run_dir> --image <img.jpg>
"""
import os
import sys
import argparse
import textwrap

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gmdb_common import load_pipnet, build_transform
from util.data import load_landmarks, landmark_path_for_image, NUM_LANDMARKS
from util.vis_pipnet import get_img_coordinates

DEFAULT_MODEL = '/data2/masino_lab/abamini/pipnet/past_runs/run_16_all_gmdb_images_fixed_protoypes_data_fixed'
DEFAULT_IMAGE = '/data2/masino_lab/abamini/pipnet/past_runs/run_16_all_gmdb_images_fixed_protoypes_data_fixed/test_explanation_images/cornelia_de_lange_02.jpg'
TRAIN_DIR     = '/data2/masino_lab/abamini/facial_data/Final_data/gmdb_train'   # class-name order


def resolve_landmarks(image_path, landmarks_dir, img_size):
    """Return a (1, L, 2) landmark tensor for masked pooling, or None if unavailable.
    GMDB landmark files are '<id>_aligned_landmarks.txt' (id = leading numeric token), which
    differs from the doubly-aligned image stem, so we try the id-based name first."""
    if landmarks_dir is None:
        return None
    iid = os.path.basename(image_path).split('_')[0]
    cand = os.path.join(landmarks_dir, '%s_aligned_landmarks.txt' % iid)
    path = cand if os.path.exists(cand) else landmark_path_for_image(image_path, landmarks_dir)
    if not os.path.exists(path):
        print('[warn] no landmark file for %s -> masked pooling will fall back to global.' % image_path)
        return None
    lm = load_landmarks(path, img_size, NUM_LANDMARKS)
    return torch.from_numpy(lm).unsqueeze(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_dir', default=DEFAULT_MODEL)
    p.add_argument('--checkpoint', default='net_trained')
    p.add_argument('--image', default=DEFAULT_IMAGE)
    p.add_argument('--train_dir', default=TRAIN_DIR, help='ImageFolder whose sorted class names label the outputs.')
    p.add_argument('--landmarks_dir', default=None, help='Default: the run\'s training landmarks_dir.')
    p.add_argument('--topk', type=int, default=6, help='Number of top-contributing prototypes to show.')
    p.add_argument('--output_dir', default=None, help='Default: <model_dir>/prediction_explanations/<img>.')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    device = torch.device(args.device)
    net, margs = load_pipnet(args.model_dir, args.checkpoint, device)
    classes = sorted(os.listdir(args.train_dir))                         # ImageFolder order
    transform = build_transform(margs.net, margs.image_size)
    masked = getattr(margs, 'masked_pool', False)
    lm_dir = args.landmarks_dir or getattr(margs, 'landmarks_dir', None)
    img_size = margs.image_size

    # With masked pooling each prototype is confined to one facial region
    # (proto2region = arange(P) % num_regions), so its region name is recoverable.
    region_names = None
    if masked and hasattr(net._pool, 'proto2region'):
        from pipnet.train import REGION_LANDMARKS
        names = list(REGION_LANDMARKS.keys())
        p2r = net._pool.proto2region.cpu()
        region_names = [names[int(p2r[d])] for d in range(p2r.shape[0])]

    def region_of(d):
        return region_names[d] if region_names else 'global'

    img_name = os.path.splitext(os.path.basename(args.image))[0]
    out_dir = args.output_dir or os.path.join(args.model_dir, 'prediction_explanations', img_name)
    os.makedirs(out_dir, exist_ok=True)

    pil = Image.open(args.image).convert('RGB')
    x = transform(pil).unsqueeze(0).to(device)
    lms = resolve_landmarks(args.image, lm_dir if masked else None, getattr(margs, 'landmark_img_size', 112))
    if lms is not None:
        lms = lms.to(device)

    with torch.no_grad():
        proto_features, pooled, out = net(x, inference=True, lms=lms)     # maps (1,P,h,w), g (1,P), logits (1,C)

    # --- classification ---
    logits = out.squeeze(0)
    probs = F.softmax(logits, dim=0)
    top_c = torch.argsort(logits, descending=True)[:3]
    pred = int(top_c[0])
    print('\n=== prediction for %s ===' % img_name, flush=True)
    for rank, c in enumerate(top_c):
        print('  %d) %-40s logit=%.3f  softmax=%.3f' % (rank + 1, classes[int(c)], logits[c].item(), probs[c].item()))

    # --- prototype contributions to the predicted class: contrib_d = g_d * w_{pred,d} ---
    W = net._classification.weight                                        # (C, P), non-negative
    g = pooled.squeeze(0)                                                 # (P,)
    contrib = (g * W[pred]).cpu()
    order = torch.argsort(contrib, descending=True)
    top = [int(d) for d in order[:args.topk] if contrib[int(d)] > 1e-3]
    print('\nTop prototypes for "%s" (contribution = presence x weight):' % classes[pred])
    print('  %-6s %-12s %-10s %-10s %-10s' % ('proto', 'region', 'contrib', 'presence', 'weight'))
    for d in top:
        print('  p%-5d %-12s %-10.3f %-10.3f %-10.3f' % (
            d, region_of(d), contrib[d].item(), g[d].item(), W[pred, d].item()))

    # --- locate + crop each top prototype's peak patch ---
    wshape = proto_features.shape[-1]
    patchsize = 32
    skip = round((img_size - patchsize) / (wshape - 1))
    img112 = pil.resize((img_size, img_size))
    boxed = img112.copy()
    draw = ImageDraw.Draw(boxed)
    try:                                    # PIL's default font is tiny; use matplotlib's DejaVuSans
        from matplotlib import font_manager
        rank_font = ImageFont.truetype(font_manager.findfont('DejaVu Sans'), 14)
    except Exception:
        rank_font = None
    colors = ['#E6194B', '#3CB44B', '#4363D8', '#F58231', '#911EB4', '#42D4F4', '#F032E6', '#BFEF45']
    patches, boxes = [], []
    for rank, d in enumerate(top):
        fmap = proto_features[0, d]                                       # (h, w)
        mh, ih = torch.max(fmap, dim=0)
        _, iw = torch.max(mh, dim=0)
        ih = int(ih[iw]); iw = int(iw)
        h0, h1, w0, w1 = get_img_coordinates(img_size, proto_features.shape, patchsize, skip, ih, iw)
        boxes.append((h0, h1, w0, w1))
        patches.append(img112.crop((w0, h0, w1, h1)))
        col = colors[rank % len(colors)]
        draw.rectangle([(w0, h0), (w1, h1)], outline=col, width=3)
        draw.text((w0 + 3, h0 + 1), str(rank + 1), fill=col, font=rank_font)

    # --- summary figure: original + boxed image + patches + contribution bars ---
    n = len(top)
    ncol = 2 + max(n, 1)                                   # original | boxed | one col per prototype
    fig = plt.figure(figsize=(2.9 * ncol + 2, 8.5))
    gs = fig.add_gridspec(2, ncol, height_ratios=[3, 2])
    ax_o = fig.add_subplot(gs[0, 0])
    ax_o.imshow(img112); ax_o.axis('off')
    ax_o.set_title('Original', fontsize=16, fontweight='bold')
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.imshow(boxed); ax_b.axis('off')
    ax_b.set_title('\nPrediction: %s\n(Score %.2f)' % (
        textwrap.fill(classes[pred], 22), probs[pred].item()), fontsize=13, fontweight='bold')
    for i, (patch, d) in enumerate(zip(patches, top)):
        axp = fig.add_subplot(gs[0, 2 + i])
        axp.imshow(patch); axp.axis('off')
        axp.set_title('#%d  p%d\n%s\nc=%.2f' % (i + 1, d, region_of(d), contrib[d].item()), fontsize=14,
                      color=colors[i % len(colors)], fontweight='bold')
    axb = fig.add_subplot(gs[1, :])
    axb.barh(range(n)[::-1], [contrib[d].item() for d in top],
             color=[colors[i % len(colors)] for i in range(n)])
    axb.set_yticks(range(n)[::-1])
    axb.set_yticklabels(['#%d p%d  %s  (p=%.2f, w=%.2f)' % (i + 1, d, region_of(d), g[d].item(),
                                                            W[pred, d].item())
                         for i, d in enumerate(top)], fontsize=14, fontweight='bold')
    axb.tick_params(axis='x', labelsize=13)
    for lbl in axb.get_xticklabels():
        lbl.set_fontweight('bold')
    axb.set_xlabel('contribution to "%s"  (presence x weight)' % classes[pred],
                   fontsize=15, fontweight='bold')
    axb.spines[['top', 'right']].set_visible(False)
    fig.suptitle('ProtoGestalt explanation: %s' % img_name, fontsize=18, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    summary = os.path.join(out_dir, 'explanation_%s.png' % img_name)
    fig.savefig(summary, dpi=150)

    # --- also save per-prototype crops + boxed image ---
    boxed.save(os.path.join(out_dir, 'boxed_%s.png' % img_name))
    for i, (patch, d) in enumerate(zip(patches, top)):
        patch.save(os.path.join(out_dir, 'rank%d_p%d_c%.3f.png' % (i + 1, d, contrib[d].item())))
    print('\n[saved] %s' % summary, flush=True)


if __name__ == '__main__':
    main()
