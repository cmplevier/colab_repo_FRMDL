# stack ESMBlock1 and ESMBlock2 into ESM-ResNet10
# EMS-ResNet10 backbone

import torch.nn as nn
import torch

from .EMSblock import EMSBlock1, EMSBlock2, SnnConv2d
from .neurons import TDBN, LIFNeuron

# table 1: COCO2017 dataset - EMS-ResNet34 - T=4
# table 2: Gen1 dataset - EMS-ResNet10 (Figure 2) - T=5, firing rate, number of parameters
# table 3: impact of residual on Gen1 - EMS-ResNet18 - firing rate, number of parameters, energy efficiency


# FOR ALL: report  mAP at IOU=0.5 (mAP@0.5) and the average AP between 0.5 and 0.95 (mAP@0.5:0.95)
def create_EMSModule(in_ch, out_ch, num_blocks, stride=2, decay=0.25):
    """
    First block: ConcatBlock_ms = EMSBlock2 (downsamples + increases channels)
    Remaining blocks: BasicBlock_ms = EMSBlock1 (same spatial size + channels)

    num_blocks=1 -> just EMSBlock2 (ResNet-10 style)
    num_blocks=2 -> EMSBlock2 + 1x EMSBlock1 (ResNet-18 style)
    num_blocks=3 -> EMSBlock2 + 2x EMSBlock1 (ResNet-34 style)
    """
    layers = []

    layers.append(EMSBlock2(in_ch, out_ch, stride=stride, decay=decay)) # always starts w. EMSBlock2

    for _ in range(num_blocks - 1):
        layers.append(EMSBlock1(out_ch, out_ch, stride=1, decay=decay))

    return nn.Sequential(*layers)

# for reproducing table 2
# T = 5
# the event camera input first passes through Conv + BN block (see figure 2 in the paper)
class EMSResNet10(nn.Module):
    """
    event frames → backbone (Conv + BN → EMSModule2 → EMSModule2 → EMSModule2 → EMSModule2) → YOLO head
    """
    def __init__(self, T = 5, decay=0.25):
        super().__init__()

        self.T = T
        self.decay = decay

        # Stem (conv2)
        self.conv1 = SnnConv2d(3, 32, 3, stride=2, padding=1)
        self.stem_tdbn = TDBN(32)

        # Stages (in_ch, out_ch, num_blocks, stride=2, decay=0.25)
        self.conv2 = create_EMSModule(32, 64, num_blocks=1, stride=2, decay=decay) # P2/4
        self.conv3 = create_EMSModule(64, 128, num_blocks=1, stride=2, decay=decay) # P3/8
        self.conv4 = create_EMSModule(128, 256, num_blocks=1, stride=2, decay=decay) # P4/16
        self.conv5 = create_EMSModule(256, 512, num_blocks=1, stride=2, decay=decay) # P5/32

    def forward(self, x):
        x = self.stem_tdbn(self.conv1(x))
        x = self.conv2(x)
        x = self.conv3(x)

        p4 = self.conv4(x) # [T, B, 256, H/16, W/16]
        p5 = self.conv5(p4) # [T, B, 512, H/32, W/32]

        return p4, p5
    
# for reproducing table 3
class EMSResNet18(nn.Module):

    def __init__(self, T=4, decay=0.25): # T is not mentioned in the paper!!
        super().__init__()

        self.T = T
        self.decay = decay

        # stem: conv2
        self.conv1 = SnnConv2d(3, 32, 3, stride=2, padding=1)
        self.stem_tdbn = TDBN(32)

        self.conv2 = create_EMSModule(32,  64,  num_blocks=2, stride=2, decay=decay)  # P2/4
        self.conv3 = create_EMSModule(64,  128, num_blocks=2, stride=2, decay=decay)  # P3/8
        self.conv4 = create_EMSModule(128, 256, num_blocks=2, stride=2, decay=decay)  # P4/16
        self.conv5 = create_EMSModule(256, 512, num_blocks=2, stride=2, decay=decay)

    def forward(self, x):
        x = self.stem_tdbn(self.conv1(x))
        x = self.conv2(x)
        x = self.conv3(x)

        p4 = self.conv4(x) # [T, B, 256, H/16, W/16]
        p5 = self.conv5(p4) # [T, B, 512, H/32, W/32]

        return p4, p5

# for reproducing table 1
# T = 4
# the deepest backbone
class EMSResNet34(nn.Module):

    def __init__(self, T=5, decay=0.25):
        super().__init__()

        self.T = T
        self.decay = decay

        # stem: conv2
        self.conv1 = SnnConv2d(3, 32, 3, stride=2, padding=1)
        self.stem_tdbn = TDBN(32)

        self.conv2 = create_EMSModule(32, 64,  num_blocks=3, stride=2, decay=decay) # P2/4
        self.conv3 = create_EMSModule(64, 128, num_blocks=4, stride=2, decay=decay) # P3/8
        self.conv4 = create_EMSModule(128, 256, num_blocks=6, stride=2, decay=decay) # P4/16
        self.conv5 = create_EMSModule(256, 512, num_blocks=3, stride=2, decay=decay) # P5/32

    def forward(self, x):
        x = self.stem_tdbn(self.conv1(x))
        x = self.conv2(x)
        x = self.conv3(x)

        p4 = self.conv4(x) # [T, B, 256, H/16, W/16]
        p5 = self.conv5(p4) # [T, B, 512, H/32, W/32]

        return p4, p5

if __name__ == "__main__":
    # parameters
    model = EMSResNet18(T=4, decay=0.25)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"EMSResNet18 Trainable parameters: {trainable_params:,}")

    model = EMSResNet10(T=4, decay=0.25)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"EMSResNet10 Trainable parameters: {trainable_params:,}")

    model = EMSResNet34(T=4, decay=0.25)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"EMSResNet34 Trainable parameters: {trainable_params:,}")
    
    configurations = [
        ("EMSResNet10", EMSResNet10, 5),
        ("EMSResNet18", EMSResNet18, 5),
        ("EMSResNet34", EMSResNet34, 4),
    ]

    for name, ModelClass, T in configurations:
        print(f"\nTesting {name} at T={T}...")

        model = ModelClass(T=T)
        model.eval()

        x = torch.randn(T, 1, 3, 640, 640)

        with torch.no_grad():
            p4, p5 = model(x)

        print(f"input: {list(x.shape)}")
        print(f"p4: {list(p4.shape)}")
        print(f"p5: {list(p5.shape)}")

        assert p4.shape == torch.Size([T, 1, 256, 40, 40])
        assert p5.shape == torch.Size([T, 1, 512, 20, 20])

        print("pass")

    print("\nAll good.")
