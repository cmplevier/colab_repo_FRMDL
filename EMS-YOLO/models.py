import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from SNN_framework.ResNetBackbones import EMSResNet34
from Head import EMSYOLOHead
from torch import nn

class EMSYOLO(nn.Module):
    def __init__(self, T=5, decay=0.25, num_classes=80, num_anchors=3):
        super().__init__()

        self.T = T
        self.nc = num_classes
        self.na = num_anchors
        self.strides = [16, 32]

        self.backbone = EMSResNet34(T=T, decay=decay)

        self.head = EMSYOLOHead(
            in_channels=(256,512),
            num_classes=num_classes,
            num_anchors=num_anchors,
            decay=decay,
        )

    def set_T(self, T):
        self.T = T
        self.backbone.T = T

    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)

        p4, p5 = self.backbone(x)

        pred_p4, pred_p5 = self.head(p4, p5)

        return pred_p4, pred_p5
