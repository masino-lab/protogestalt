"""
Prototype activation comparison: normal vs. rare disease images.

For every prototype, runs a Mann-Whitney U test to check whether
the max-pooled activation score is significantly different between
the two groups, then applies Benjamini-Hochberg FDR correction.

Usage
-----
python proto_stats_normal_vs_rare.py \
    --checkpoint /data2/masino_lab/abamini/pipnet/runs/run_7_GMDB_syndromes_tuned/checkpoints/net_trained_last \
    --normal_dir /data2/masino_lab/abamini/facial_data/combined_normal_and_rare/test_crop/normal/ \
    --rare_dir   /data2/masino_lab/abamini/facial_data/combined_normal_and_rare/test_crop/rare/ \
    --output_dir /data2/masino_lab/abamini/pipnet/results_mw/ \
    --net        convnext_tiny_26 \
    --image_size 112
"""

import os, sys, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Make sure PIPNet root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipnet.pipnet import PIPNet, get_network


# ── 1. Simple image dataset ──────────────────────────────────────────────────

class ImageFolder(Dataset):
    """Loads all images from a directory (any depth). Returns (tensor, path)."""
    EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

    def __init__(self, root, transform=None):
        self.transform = transform
        self.paths = [
            os.path.join(d, f)
            for d, _, files in os.walk(root)
            for f in sorted(files)
            if os.path.splitext(f)[1].lower() in self.EXTS
        ]
        if not self.paths:
            raise RuntimeError(f'No images found under {root!r}')
        print(f'  {len(self.paths)} images found in {root!r}', flush=True)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.paths[idx]


# ── 2. Load model ────────────────────────────────────────────────────────────

def load_model(checkpoint_path, device, args):
    ckpt = torch.load(checkpoint_path, map_location=device)
    sd   = ckpt['model_state_dict']

    # Infer sizes from saved weight matrix  (num_classes × num_prototypes)
    num_classes, num_prototypes = sd['module._classification.weight'].shape
    print(f'  num_classes={num_classes}, num_prototypes={num_prototypes}', flush=True)

    feature_net, add_on, pool, cls_layer, _ = get_network(num_classes, args)
    net = PIPNet(num_classes, num_prototypes, feature_net, args, add_on, pool, cls_layer)
    net = nn.DataParallel(net).to(device)

    model_sd  = net.state_dict()
    filtered  = {k: v for k, v in sd.items()
                 if k in model_sd and model_sd[k].shape == v.shape}
    net.load_state_dict(filtered, strict=False)
    net.eval()
    return net


# ── 3. Extract activations ───────────────────────────────────────────────────

@torch.no_grad()
def get_activations(net, loader, device):
    """Returns array of shape (N_images, N_prototypes)."""
    rows = []
    for imgs, _ in tqdm(loader, desc='  inference', ncols=70, mininterval=5):
        _, pooled, _ = net(imgs.to(device), inference=True)
        rows.append(pooled.cpu().numpy())
    return np.vstack(rows)


# ── 4. Mann-Whitney U per prototype ─────────────────────────────────────────

def mann_whitney_tests(act_normal, act_rare, alpha=0.05):
    """
    For each prototype p:
      - Compare act_normal[:, p]  vs  act_rare[:, p]  (two-sided)
      - Compute rank-biserial correlation r as effect size
      - Correct p-values with Benjamini-Hochberg FDR

    Returns a DataFrame sorted by q-value.
    """
    P  = act_normal.shape[1]
    n1 = act_normal.shape[0]
    n2 = act_rare.shape[0]

    protos, p_vals, U_vals = [], [], []

    for p in tqdm(range(P), desc='  Mann-Whitney', ncols=70, mininterval=5):
        a = act_normal[:, p]
        b = act_rare[:,   p]

        if np.all(a == 0) and np.all(b == 0):
            U, pval = 0.0, 1.0          # prototype never fires in either group
        else:
            U, pval = mannwhitneyu(a, b, alternative='two-sided')

        protos.append(p)
        U_vals.append(U)
        p_vals.append(pval)

    # BH-FDR correction
    rejected, q_vals, _, _ = multipletests(p_vals, alpha=alpha, method='fdr_bh')

    df = pd.DataFrame({
        'prototype':        protos,
        'mean_normal':      act_normal.mean(axis=0),
        'mean_rare':        act_rare.mean(axis=0),
        'U_statistic':      U_vals,
        'p_value':          p_vals,
        'q_value_BH':       q_vals,
        'significant':      rejected,
        # r > 0 → higher in rare,  r < 0 → higher in normal
        'effect_size_r':    [1 - 2*U / (n1*n2) for U in U_vals],
    })
    return df.sort_values('q_value_BH').reset_index(drop=True)


# ── 5. Plots ─────────────────────────────────────────────────────────────────

def save_plots(df, act_normal, act_rare, out_dir, alpha, n_top):
    # --- Volcano plot ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ['#d62728' if s else '#aec7e8' for s in df['significant']]
    ax.scatter(df['effect_size_r'], -np.log10(df['q_value_BH'].clip(1e-300)),
               c=colors, s=15, alpha=0.8, linewidths=0)
    ax.axhline(-np.log10(alpha), color='grey', ls='--', lw=1,
               label=f'q = {alpha}')
    ax.axvline(0, color='grey', ls='--', lw=0.8)
    ax.set_xlabel('Effect size  r  (+ = higher in rare)', fontsize=10)
    ax.set_ylabel(r'$-\log_{10}(q_\mathrm{BH})$', fontsize=10)
    ax.set_title('Prototype activations: rare vs normal\n(Mann-Whitney, BH-FDR)', fontsize=11)
    ax.legend(fontsize=8)
    for _, row in df[df['significant']].head(8).iterrows():
        ax.annotate(f"P{int(row.prototype)}",
                    (row.effect_size_r, -np.log10(row.q_value_BH + 1e-300)),
                    fontsize=7, ha='center', va='bottom',
                    xytext=(0, 3), textcoords='offset points')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'volcano.png'), dpi=150)
    plt.close()

    # --- Box plots for top significant prototypes ----------------------------
    sig = df[df['significant']].copy()
    if sig.empty:
        return
    top = sig.nlargest(n_top // 2, 'effect_size_r')           # highest in rare
    top = pd.concat([top, sig.nsmallest(n_top // 2,            # highest in normal
                                         'effect_size_r')])
    ncols = min(4, len(top))
    nrows = int(np.ceil(len(top) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(4*ncols, 3.5*nrows), squeeze=False)
    for i, (_, row) in enumerate(top.iterrows()):
        p  = int(row.prototype)
        ax = axes[i // ncols][i % ncols]
        bp = ax.boxplot([act_normal[:, p], act_rare[:, p]],
                        labels=['Normal', 'Rare'], patch_artist=True,
                        widths=0.5, showfliers=False)
        for patch, col in zip(bp['boxes'], ['#4878d0', '#ee854a']):
            patch.set_facecolor(col); patch.set_alpha(0.7)
        ax.set_title(f'P{p}\nr={row.effect_size_r:.2f}  q={row.q_value_BH:.2e}',
                     fontsize=8)
        ax.set_ylabel('Activation', fontsize=7)
        ax.tick_params(labelsize=7)
    for j in range(len(top), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    plt.suptitle(f'Top {len(top)} significant prototypes', fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'top_prototypes.png'), dpi=150,
                bbox_inches='tight')
    plt.close()


# ── 6. CLI ───────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',   required=True)
    p.add_argument('--normal_dir',   required=True)
    p.add_argument('--rare_dir',     required=True)
    p.add_argument('--output_dir',   required=True)
    p.add_argument('--net',          default='convnext_tiny_26')
    p.add_argument('--image_size',   type=int, default=112)
    p.add_argument('--num_features', type=int, default=0)
    p.add_argument('--bias',         action='store_true')
    p.add_argument('--disable_pretrained', action='store_true')
    p.add_argument('--batch_size',   type=int, default=32)
    p.add_argument('--num_workers',  type=int, default=4)
    p.add_argument('--disable_cuda', action='store_true')
    p.add_argument('--alpha',        type=float, default=0.05)
    p.add_argument('--n_top_plots',  type=int,   default=20)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = (torch.device('cpu') if args.disable_cuda or not torch.cuda.is_available()
              else torch.device('cuda'))
    print(f'Device: {device}', flush=True)

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # Load model
    print('\nLoading model ...', flush=True)
    net = load_model(args.checkpoint, device, args)

    # Infer wshape (needed by get_patch_size elsewhere)
    with torch.no_grad():
        dummy = torch.zeros(1, 3, args.image_size, args.image_size, device=device)
        pf, _, _ = net(dummy)
        args.wshape = pf.shape[-1]

    # Extract activations
    def make_loader(path):
        ds = ImageFolder(path, transform)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers,
                          pin_memory=torch.cuda.is_available())

    print('\nExtracting activations — normal ...', flush=True)
    act_normal = get_activations(net, make_loader(args.normal_dir), device)

    print('Extracting activations — rare ...', flush=True)
    act_rare = get_activations(net, make_loader(args.rare_dir), device)

    print(f'\nact_normal shape: {act_normal.shape}  '
          f'(images × prototypes)', flush=True)
    print(f'act_rare   shape: {act_rare.shape}', flush=True)

    # Run tests
    print('\nRunning Mann-Whitney U tests ...', flush=True)
    df = mann_whitney_tests(act_normal, act_rare, alpha=args.alpha)

    # Save CSV
    csv = os.path.join(args.output_dir, 'prototype_stats.csv')
    df.to_csv(csv, index=False)
    print(f'\nSaved: {csv}', flush=True)

    # Print summary
    n_sig = int(df['significant'].sum())
    P     = len(df)
    print(f'\n── Results (α={args.alpha} BH-FDR) ─────────────────────────')
    print(f'Prototypes tested   : {P}')
    print(f'Significant         : {n_sig}  ({100*n_sig/P:.1f}%)')
    print(f'  higher in rare    : {int((df.significant & (df.effect_size_r > 0)).sum())}')
    print(f'  higher in normal  : {int((df.significant & (df.effect_size_r < 0)).sum())}')
    print('\nTop 10 by |effect size|:')
    cols = ['prototype', 'mean_normal', 'mean_rare', 'effect_size_r', 'q_value_BH']
    print(df[df['significant']].reindex(
        df[df['significant']]['effect_size_r'].abs()
          .sort_values(ascending=False).index
    ).head(10)[cols].to_string(index=False))

    # Plots
    print('\nGenerating plots ...', flush=True)
    save_plots(df, act_normal, act_rare, args.output_dir,
               alpha=args.alpha, n_top=args.n_top_plots)
    print(f'\nDone. Results in: {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()