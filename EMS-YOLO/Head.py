import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from SNN_framework.EMSblock import EMSBlock1, EMSBlock2, MSBlock
from SNN_framework.neurons import LIFNeuron, TDBN


class SnnUpsample(nn.Module):
    def __init__(self, scale_factor=2, mode="nearest"):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)

    def forward(self, x):
        # x: [T, B, C, H, W]
        return torch.stack([self.upsample(x[t]) for t in range(x.shape[0])])


class SnnConcat(nn.Module):
    def forward(self, tensors):
        # tensors: list of [T, B, C, H, W] -> concat on channel dim=2
        return torch.cat(tensors, dim=2)


class MembraneHead(nn.Module):
    """
    Reads out detection predictions from spiking features.
    Averages spikes over T, then applies a 1x1 conv to get anchor predictions.
    Output: [B, na*(5+nc), H, W]
    """
    def __init__(self, in_ch, num_anchors, num_classes):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, num_anchors * (5 + num_classes), kernel_size=1)

    def forward(self, x):
        # x: [T, B, C, H, W]
        # average over time dimension to get a rate-coded output
        x = x.mean(dim=0)           # [B, C, H, W]
        return self.conv(x)         # [B, na*(5+nc), H, W]


class EMSYOLOHead(nn.Module):
    """
    2-scale spiking YOLO head matching paper figure 6.

    p5 → EMSBlock1(512→256) ──────────────→ EMSBlock2(256→512) → head_p5
                    ↓
             EMSBlock1(256→128)
                    ↓
                Upsample
                    ↓
    p4 →         Concat(128+256=384) ──→ EMSBlock1(384→256) → head_p4
    """
    def __init__(self, num_classes=2, num_anchors=3, decay=0.25):
        super().__init__()

        self.nc = num_classes
        self.na = num_anchors

        # p5 shared feature: 512→256
        self.block_p5       = EMSBlock1(512, 256, stride=1, decay=decay)

        # p5 detection branch: 256→512
        self.block_p5_out   = EMSBlock2(256, 512, stride=1, decay=decay)

        # p4 path: reduce 256→128, upsample, concat with p4 (128+256=384), detect 384→256
        self.block_p4_reduce = EMSBlock1(256, 128, stride=1, decay=decay)
        self.upsample        = SnnUpsample(scale_factor=2)
        self.concat          = SnnConcat()
        self.block_p4_out    = EMSBlock1(384, 256, stride=1, decay=decay)

        # output heads
        self.head_p5 = MembraneHead(512, num_anchors, num_classes)
        self.head_p4 = MembraneHead(256, num_anchors, num_classes)

    def forward(self, p4, p5):
        # p4: [T, B, 256, H/16, W/16]
        # p5: [T, B, 512, H/32, W/32]

        # shared p5 features
        x = self.block_p5(p5)               # [T, B, 256, H/32, W/32]

        # p5 detection branch
        p5_det = self.block_p5_out(x)       # [T, B, 512, H/32, W/32]

        # p4 branch
        x = self.block_p4_reduce(x)         # [T, B, 128, H/32, W/32]
        x = self.upsample(x)                # [T, B, 128, H/16, W/16]
        x = self.concat([x, p4])            # [T, B, 384, H/16, W/16]
        p4_det = self.block_p4_out(x)       # [T, B, 256, H/16, W/16]

        # readout: average spikes over T, apply 1x1 conv
        pred_p4 = self.head_p4(p4_det)      # [B, na*(5+nc), H/16, W/16]
        pred_p5 = self.head_p5(p5_det)      # [B, na*(5+nc), H/32, W/32]

        return pred_p4, pred_p5


if __name__ == "__main__":
    T, B = 5, 1
    p4 = torch.randn(T, B, 256, 40, 40)
    p5 = torch.randn(T, B, 512, 20, 20)

    head = EMSYOLOHead(num_classes=2, num_anchors=3)
    head.eval()

    with torch.no_grad():
        pred_p4, pred_p5 = head(p4, p5)

    print(f"pred_p4: {list(pred_p4.shape)}")  # [B, 3*7, 40, 40] = [1, 21, 40, 40]
    print(f"pred_p5: {list(pred_p5.shape)}")  # [B, 3*7, 20, 20] = [1, 21, 20, 20]
    print(f"head params: {sum(p.numel() for p in head.parameters()):,}")
    print("pass")