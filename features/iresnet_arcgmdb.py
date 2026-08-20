"""ArcFace glint360k-r100 trunk (GMDB run) as a PIP-Net backbone: --net iresnet100_arcgmdb.

The source checkpoint pickles a lib.models.my_arcface.MyArcFace whose `base` submodule is
the onnx2torch conversion of glint360k_r100.onnx. That ONNX is a BN-FOLDED inference graph
(every conv has an absorbed bias, only 50 BN layers remain), so it can NOT be remapped into
PIP-Net's unfolded IResNet_features. Instead we load the fused trunk directly and reshape its
flattened output back to the 512x7x7 conv feature map that PIP-Net's add-on/pool consume.

Run convert_arcgmdb_backbone.py once to produce the standalone (lib-free) base file, then
train with `--net iresnet100_arcgmdb`. NOTE: native stride-2 gives a 7x7 grid (coarser than
the 14x14 iresnet100 path), so use a larger --mask_radius with --masked_pool. Requires the
`onnx2torch` package at load time (the saved base is an onnx2torch GraphModule).
"""
import os
import torch
import torch.nn as nn

DEFAULT_ARCGMDB_BASE = '/data2/masino_lab/abamini/ArcGMDB_data/arcgmdb_iresnet100_base.pth'


class IResNet100ArcGMDB(nn.Module):
    """Wraps the fused ArcFace trunk to return its 512x(H)x(W) conv feature map.

    `base` flattens to (B, C*H*W) internally (ONNX Flatten, C-major), so a view recovers the
    spatial map exactly."""
    def __init__(self, base, channels=512, height=7, width=7):
        super().__init__()
        self.base = base
        self.channels, self.height, self.width = channels, height, width

    def forward(self, x):
        f = self.base(x)
        return f.view(f.size(0), self.channels, self.height, self.width)

    def __repr__(self):
        return 'iresnet100_arcgmdb_features'


def iresnet100_arcgmdb_features(pretrained=True, base_path=None, **kwargs):
    """Load the converted ArcFace-GMDB trunk. `pretrained` is ignored: the graph itself is
    data-defined, so the file is always loaded (PIP-Net's checkpoint then overwrites the
    weights when a trained run is reloaded)."""
    path = base_path or os.environ.get('ARCGMDB_BASE_PATH', DEFAULT_ARCGMDB_BASE)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "ArcGMDB backbone not found at %s. Run convert_arcgmdb_backbone.py first "
            "(it extracts the lib-free base from the MyArcFace .pth)." % path)
    try:
        base = torch.load(path, map_location='cpu', weights_only=False)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Loading the ArcGMDB backbone needs the 'onnx2torch' package (the saved base is an "
            "onnx2torch GraphModule). pip install onnx2torch. Original: %s" % e)
    base.eval()
    return IResNet100ArcGMDB(base)
