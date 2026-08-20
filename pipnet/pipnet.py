import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from features.resnet_features import resnet18_features, resnet34_features, resnet50_features, resnet50_features_inat, resnet101_features, resnet152_features
from features.convnext_features import convnext_tiny_26_features, convnext_tiny_13_features
from features.iresnet_features import iresnet100_features
from features.iresnet_arcgmdb import iresnet100_arcgmdb_features
import torch
from torch import Tensor

class PIPNet(nn.Module):
    def __init__(self,
                 num_classes: int,
                 num_prototypes: int,
                 feature_net: nn.Module,
                 args: argparse.Namespace,
                 add_on_layers: nn.Module,
                 pool_layer: nn.Module,
                 classification_layer: nn.Module
                 ):
        super().__init__()
        assert num_classes > 0
        self._num_features = args.num_features
        self._num_classes = num_classes
        self._num_prototypes = num_prototypes
        self._net = feature_net
        self._add_on = add_on_layers
        self._pool = pool_layer
        self._classification = classification_layer
        self._multiplier = classification_layer.normalization_multiplier

    def forward(self, xs,  inference=False, lms=None):
        features = self._net(xs)
        proto_features = self._add_on(features)
        # When the pool layer is region-masked it needs the per-image landmarks to
        # build the masks; the global pool ignores them. Passing lms is harmless
        # for the global pool (it accepts no landmarks, so we branch on the type).
        if isinstance(self._pool, MaskedMaxPool):
            pooled = self._pool(proto_features, lms)
        else:
            pooled = self._pool(proto_features)
        if inference:
            clamped_pooled = torch.where(pooled < 0.1, 0., pooled)  #during inference, ignore all prototypes that have 0.1 similarity or lower
            out = self._classification(clamped_pooled) #shape (bs*2, num_classes)
            return proto_features, clamped_pooled, out
        else:
            out = self._classification(pooled) #shape (bs*2, num_classes)
            return proto_features, pooled, out


class MaskedMaxPool(nn.Module):
    """Region-masked max pooling — the architectural alternative to PIP-Net's global
    spatial pooling.

    Each prototype d is assigned (via ``proto2region``) to one facial region g. Its
    presence score is the max activation *only over feature-map locations inside that
    region's binary mask* M_g:

        presence[d] = max over (i,j) in M_g of activation[d, i, j]

    so the prototype is structurally unable to contribute to the prediction from
    outside its landmark region. The per-image masks are built on the fly at the
    feature-map resolution from MediaPipe landmarks: a grid cell belongs to region g
    if it lies within ``radius`` (a fraction of the normalized image) of any of that
    region's landmarks.

    Falls back to global max pooling when no landmarks are supplied (e.g. the
    shape-probe forward pass, or inference without landmarks) and, per (image,
    prototype), when that image's region mask is empty because its landmarks are
    missing — so a missing landmark degrades to standard PIP-Net behaviour rather
    than zeroing the prototype out.
    """
    def __init__(self, proto2region: Tensor, region_landmark_indices, radius: float):
        super().__init__()
        self.register_buffer('proto2region', proto2region.long())
        # list[list[int]] of landmark indices per region (kept on CPU; tiny).
        self.region_landmark_indices = [list(idx) for idx in region_landmark_indices]
        self.radius = float(radius)

    def _build_region_masks(self, lms: Tensor, H: int, W: int) -> Tensor:
        """lms: (B, L, 2) normalized (row, col) in [0,1], -1 = missing.
        Returns a boolean (B, R, H, W) region-membership mask."""
        device, dtype = lms.device, lms.dtype
        B = lms.shape[0]
        rows = torch.linspace(0., 1., steps=H, device=device, dtype=dtype)
        cols = torch.linspace(0., 1., steps=W, device=device, dtype=dtype)
        gr, gc = torch.meshgrid(rows, cols, indexing='ij')
        grid = torch.stack([gr.reshape(-1), gc.reshape(-1)], dim=-1)  # (H*W, 2)
        r2 = self.radius ** 2
        masks = []
        for idx in self.region_landmark_indices:
            idx_t = torch.tensor(idx, device=device, dtype=torch.long)
            pts = lms[:, idx_t, :]                              # (B, k, 2)
            valid = (pts >= 0).all(dim=-1)                      # (B, k)
            d2 = ((grid[None, None] - pts[:, :, None]) ** 2).sum(dim=-1)  # (B, k, H*W)
            within = (d2 <= r2) & valid[:, :, None]
            masks.append(within.any(dim=1).reshape(B, H, W))    # (B, H, W)
        return torch.stack(masks, dim=1)                        # (B, R, H, W)

    def forward(self, proto_features: Tensor, lms: Tensor = None) -> Tensor:
        B, P, H, W = proto_features.shape
        global_pooled = proto_features.flatten(start_dim=2).amax(dim=2)  # (B, P)
        if lms is None:
            return global_pooled
        region_masks = self._build_region_masks(lms, H, W)              # (B, R, H, W)
        proto_masks = region_masks[:, self.proto2region]                # (B, P, H, W)
        neg = torch.finfo(proto_features.dtype).min
        masked = torch.where(proto_masks, proto_features, neg)
        pooled = masked.flatten(start_dim=2).amax(dim=2)                # (B, P)
        # Per (image, prototype) fallback to global pooling for empty masks.
        empty = ~proto_masks.flatten(start_dim=2).any(dim=2)            # (B, P)
        return torch.where(empty, global_pooled, pooled)


base_architecture_to_features = {'resnet18': resnet18_features,
                                 'resnet34': resnet34_features,
                                 'resnet50': resnet50_features,
                                 'resnet50_inat': resnet50_features_inat,
                                 'resnet101': resnet101_features,
                                 'resnet152': resnet152_features,
                                 'convnext_tiny_26': convnext_tiny_26_features,
                                 'convnext_tiny_13': convnext_tiny_13_features,
                                 'iresnet100': iresnet100_features,
                                 'iresnet100_arcgmdb': iresnet100_arcgmdb_features}

# adapted from https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
class NonNegLinear(nn.Module):
    """Applies a linear transformation to the incoming data with non-negative weights`
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(NonNegLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.normalization_multiplier = nn.Parameter(torch.ones((1,),requires_grad=True))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input,torch.relu(self.weight), self.bias)


def get_network(num_classes: int, args: argparse.Namespace): 
    features = base_architecture_to_features[args.net](pretrained=not args.disable_pretrained)
    features_name = str(features).upper()
    if 'next' in args.net:
        features_name = str(args.net).upper()
    if features_name.startswith('RES') or features_name.startswith('CONVNEXT') or features_name.startswith('IRESNET'):
        first_add_on_layer_in_channels = \
            [i for i in features.modules() if isinstance(i, nn.Conv2d)][-1].out_channels
    else:
        raise Exception('other base architecture NOT implemented')
    
    
    if args.num_features == 0:
        num_prototypes = first_add_on_layer_in_channels
        print("Number of prototypes: ", num_prototypes, flush=True)
        add_on_layers = nn.Sequential(
            nn.Softmax(dim=1), #softmax over every prototype for each patch, such that for every location in image, sum over prototypes is 1                
    )
    else:
        num_prototypes = args.num_features
        print("Number of prototypes set from", first_add_on_layer_in_channels, "to", num_prototypes,". Extra 1x1 conv layer added.", flush=True)
        # A learnable 1x1 conv lets prototypes be recombinations of the backbone
        # features instead of the raw channels. The BatchNorm before the softmax
        # normalizes each prototype channel so the softmax does not saturate on a
        # few high-magnitude channels (critical for face-recognition backbones like
        # ArcFace IResNet, whose raw layer4 features otherwise collapse the prototype
        # space to ~8 active prototypes and stall the align-loss pretraining).
        add_on_layers = nn.Sequential(
            nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=num_prototypes, kernel_size=1, stride = 1, padding=0, bias=True),
            nn.BatchNorm2d(num_prototypes),
            nn.Softmax(dim=1), #softmax over every prototype for each patch, such that for every location in image, sum over prototypes is 1
    )
    if getattr(args, 'masked_pool', False):
        # Region-masked pooling: confine each prototype to a facial-landmark region.
        from pipnet.train import REGION_LANDMARKS
        region_landmark_indices = list(REGION_LANDMARKS.values())
        num_regions = len(region_landmark_indices)
        proto2region = torch.arange(num_prototypes) % num_regions
        pool_layer = MaskedMaxPool(proto2region, region_landmark_indices, args.mask_radius)
        print("Region-masked pooling ENABLED: %d prototypes confined across %d regions %s (radius=%.3f)" % (
            num_prototypes, num_regions, list(REGION_LANDMARKS.keys()), args.mask_radius), flush=True)
    else:
        pool_layer = nn.Sequential(
                    nn.AdaptiveMaxPool2d(output_size=(1,1)), #outputs (bs, ps,1,1)
                    nn.Flatten() #outputs (bs, ps)
                    )

    if args.bias:
        classification_layer = NonNegLinear(num_prototypes, num_classes, bias=True)
    else:
        classification_layer = NonNegLinear(num_prototypes, num_classes, bias=False)
        
    return features, add_on_layers, pool_layer, classification_layer, num_prototypes


    