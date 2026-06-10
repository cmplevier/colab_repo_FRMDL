import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from SNN_framework.ResNetBackbones import EMSResNet10, EMSResNet18, EMSResNet34
from Head import EMSYOLOHead
from torch import nn

_BACKBONES = {
    "ems_resnet10": EMSResNet10,
    "ems_resnet18": EMSResNet18,
    "ems_resnet34": EMSResNet34,
}


class EMSYOLO(nn.Module):
    def __init__(self, backbone="ems_resnet34", T=5, decay=0.25, num_classes=80, num_anchors=3):
        super().__init__()

        self.T = T
        self.nc = num_classes
        self.na = num_anchors
        self.strides = [16, 32]

        if backbone not in _BACKBONES:
            raise ValueError(f"Unknown backbone '{backbone}'. Choose from: {list(_BACKBONES)}")

        self.backbone = _BACKBONES[backbone](T=T, decay=decay)

        self.head = EMSYOLOHead(
            num_classes=num_classes,
            num_anchors=num_anchors,
            decay=decay,
        )

    def set_T(self, T):
        self.T = T
        self.backbone.T = T

    def forward(self, x):
        # x: (B, C, H, W) for COCO — replicated to (T, B, C, H, W)
        # x: (T, B, C, H, W) for Gen1 — passed through directly
        if x.dim() == 4:
            x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)

        p4, p5 = self.backbone(x)

        pred_p4, pred_p5 = self.head(p4, p5)

        return pred_p4, pred_p5
