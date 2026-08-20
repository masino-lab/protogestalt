import os
import torch
import torch.nn as nn


model_dir = '/data2/masino_lab/abamini/models'

iresnet_weight_paths = {
    'iresnet100': os.path.join(model_dir, 'arcface_iresnet100.pth'),
}


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class IBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out


class IResNet_features(nn.Module):
    """InsightFace IResNet trunk, stripped of the embedding head.

    The original head (bn2 / dropout / flatten / fc(512*7*7 -> 512) / features-BN1d)
    is intentionally omitted so forward(x) returns the 4D conv feature map that
    PIP-Net's add-on / pool / classification layers consume.

    PIP-Net modification: layer4 uses stride=1 (instead of the native stride=2),
    yielding a 14x14 prototype grid at 112-pixel input — matching the spatial
    resolution PIP-Net's existing ResNet50 backbone path produces at 112 input.
    """

    def __init__(self, block, layers, name='iresnet'):
        super().__init__()
        self.inplanes = 64
        self._name = name

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)

        self.layer1 = self._make_layer(block, 64,  layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def __repr__(self):
        return f'{self._name}_features'


def _load_arcface_weights(model, weight_path):
    if not os.path.exists(weight_path):
        print(
            f"Pretrained IResNet weights not found at {weight_path}. "
            f"Download an InsightFace ArcFace IR-100 checkpoint (e.g. backbone.pth from "
            f"https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch) "
            f"and place it there. Continuing with random init.",
            flush=True,
        )
        return
    sd = torch.load(weight_path, map_location='cpu')
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    sd = {k.replace('module.', '', 1) if k.startswith('module.') else k: v for k, v in sd.items()}
    drop_prefixes = ('bn2.', 'dropout.', 'fc.', 'features.')
    sd = {k: v for k, v in sd.items() if not k.startswith(drop_prefixes)}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        print(f"IResNet pretrained load — unexpected keys ignored: {unexpected}", flush=True)
    if missing:
        print(f"IResNet pretrained load — missing keys (left at init): {missing}", flush=True)


def iresnet100_features(pretrained=False, **kwargs):
    """ArcFace IR-100 backbone, PIP-Net-adapted (layer4 stride=1).

    At 112x112 input this produces a 512x14x14 feature map.
    """
    model = IResNet_features(IBasicBlock, [3, 13, 30, 3], name='iresnet100')
    if pretrained:
        _load_arcface_weights(model, iresnet_weight_paths['iresnet100'])
    return model
