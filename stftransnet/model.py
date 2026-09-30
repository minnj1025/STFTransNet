"""STFTransNet: RCN two-stream encoder + cross-attention + TCN.

Module names follow the experiment code so the trained weights load with strict=True.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from torchvision.models import ResNet18_Weights, resnet18


# ---------------------------------------------------------------- RCN (ResNet-18 + CBAM)
class SAM(nn.Module):
    """Spatial attention."""

    def __init__(self, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, dilation=1, bias=bias)

    def forward(self, x):
        pooled = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        return self.conv(pooled) * x


class CAM(nn.Module):
    """Channel attention."""

    def __init__(self, channels, r):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(channels, channels // r, bias=True),
            nn.GELU(),
            nn.Linear(channels // r, channels, bias=True),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        mx = self.linear(F.adaptive_max_pool2d(x, 1).view(b, c)).view(b, c, 1, 1)
        avg = self.linear(F.adaptive_avg_pool2d(x, 1).view(b, c)).view(b, c, 1, 1)
        return torch.sigmoid(mx + avg) * x


class CBAM(nn.Module):
    def __init__(self, channels, r):
        super().__init__()
        self.sam = SAM(bias=False)
        self.cam = CAM(channels=channels, r=r)

    def forward(self, x):
        return self.sam(self.cam(x)) + x


class ResNet18WithCBAM(nn.Module):
    """ResNet-18 (ImageNet) with a residual CBAM after every stage; returns a 512-d feature."""

    def __init__(self, pretrained=True):
        super().__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.cbam_layers = nn.ModuleList([CBAM(c, r=16) for c in (64, 128, 256, 512)])
        self.resnet.fc = nn.Identity()
        self.gelu = nn.GELU()

    def forward(self, x):
        r = self.resnet
        x = r.maxpool(self.gelu(r.bn1(r.conv1(x))))
        for stage, cbam in zip((r.layer1, r.layer2, r.layer3, r.layer4), self.cbam_layers):
            x = cbam(stage(x))
        return torch.flatten(r.avgpool(x), 1)


class CrossAttention(nn.Module):
    """Query from the face-crop feature, key/value from the full-frame feature."""

    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, frame_feat, face_feat):
        kv, q = frame_feat.unsqueeze(1), face_feat.unsqueeze(1)
        out, _ = self.multihead_attn(q, kv, kv)
        return out.squeeze(1)


# ---------------------------------------------------------------- TCN
def _xavier_gelu(weight):
    # GELU has no torch gain entry; sqrt(2) is used, as in the original TCN code.
    nn.init.xavier_uniform_(weight, gain=np.sqrt(2))


class TemporalConv1d(nn.Conv1d):
    """Non-causal 'same' convolution; an odd total padding is completed on the right."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        self.pad_len = (kernel_size - 1) * dilation
        super().__init__(in_channels, out_channels, kernel_size, 1, self.pad_len // 2, dilation)

    def forward(self, x):
        if self.pad_len % 2:
            x = F.pad(x, [0, 1])
        return super().forward(x)


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout):
        super().__init__()
        self.conv1 = weight_norm(TemporalConv1d(n_inputs, n_outputs, kernel_size, dilation))
        self.conv2 = weight_norm(TemporalConv1d(n_outputs, n_outputs, kernel_size, dilation))
        self.activation1, self.activation2, self.activation_final = nn.GELU(), nn.GELU(), nn.GELU()
        self.dropout1, self.dropout2 = nn.Dropout(dropout), nn.Dropout(dropout)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        _xavier_gelu(self.conv1.weight)
        _xavier_gelu(self.conv2.weight)
        if self.downsample is not None:
            _xavier_gelu(self.downsample.weight)

    def forward(self, x):
        out = self.dropout1(self.activation1(self.conv1(x)))
        out = self.dropout2(self.activation2(self.conv2(out)))
        res = x if self.downsample is None else self.downsample(x)
        return self.activation_final(out + res), out


class TCN(nn.Module):
    """Dilated TCN with skip connections summed over all blocks. Input (N, L, C)."""

    def __init__(self, num_inputs, num_channels, kernel_size, dilations, dropout):
        super().__init__()
        self.downsample_skip_connection = nn.ModuleList(
            [nn.Conv1d(c, num_channels[-1], 1) if c != num_channels[-1] else None for c in num_channels])
        for layer in self.downsample_skip_connection:
            if layer is not None:
                _xavier_gelu(layer.weight)
        self.activation_skip_out = nn.GELU()
        self.network = nn.ModuleList([
            TemporalBlock(num_inputs if i == 0 else num_channels[i - 1], c, kernel_size, d, dropout)
            for i, (c, d) in enumerate(zip(num_channels, dilations))])

    def forward(self, x):
        x = x.transpose(1, 2)
        skips = []
        for i, layer in enumerate(self.network):
            x, skip = layer(x)
            if self.downsample_skip_connection[i] is not None:
                skip = self.downsample_skip_connection[i](skip)
            if i < len(self.network) - 1:
                skips.append(skip)
        skips.append(x)
        x = self.activation_skip_out(torch.stack(skips, dim=0).sum(dim=0))
        return x.transpose(1, 2)


# ---------------------------------------------------------------- STFTransNet
class STFTransNet(nn.Module):
    """Two-stream spatial encoder fused by cross-attention, then a TCN.

    The TCN runs along the sequence of frames in a mini-batch (the frames of one batch
    form the temporal axis), so the batch order and size are part of the model input.
    """

    def __init__(self, num_classes=4, pretrained=True):
        super().__init__()
        self.resnet_cbam1 = ResNet18WithCBAM(pretrained)  # full frame
        self.resnet_cbam2 = ResNet18WithCBAM(pretrained)  # face crop
        self.cross_attention = CrossAttention(embed_dim=512)
        self.mytcn = TCN(512, [512, 256, 128], kernel_size=10, dilations=[2, 3, 4], dropout=0.25)
        self.res_linear = nn.Linear(512, 128)
        self.final_linear = nn.Linear(128, num_classes)

    def forward(self, frame, face):
        if torch.count_nonzero(face) > 0:
            f_frame = self.resnet_cbam1(frame)
            fused = self.cross_attention(f_frame, self.resnet_cbam2(face))
        else:  # no face crop anywhere in the batch
            f_frame = fused = self.resnet_cbam1(frame)

        temporal = self.mytcn(fused.unsqueeze(0)).squeeze(0)  # (B, 128)
        return self.final_linear(temporal + self.res_linear(f_frame))
