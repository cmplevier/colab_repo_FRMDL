# EMS-Block1 and EMS-Block2 (full-spike residual blocks (Section 4.2))
# Energy-Efficient Menbrane-Shortcut

import torch.nn as nn
import torch

from .neurons import LIFNeuron, TDBN

# LIFNeuron: spiking activation: converts membrane potential to binary spikes
# TDBN: threshold-dependent batch normalisation: normalize before each LIF

class SnnConv2d(nn.Module):
    """
    spatial Conv2d applied independently at each timestep
    - the same convolution weights are shared across all timesteps
    input: [T, B, C, H, W] 
    output: [T, B, C_out, H_out, W_out]
    """
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=False):
        super().__init__()
        # standard 2D convolution: learns spatial filters
        # bias=False because TDBN already has a learnable bias
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        # x: [T, B, C, H, W]
        # x[t] slices one time step: [B, C, H, W] -> valid input for nn.Conv2d
        # torch.stack: reassembles the T results back into [T, B, C_out, H_out, W_out]
        return torch.stack([self.conv(x[t]) for t in range(x.shape[0])])


class SnnMaxPool2d(nn.Module):
    """
    spatial MaxPool2d applied independently at each timestep
    - used to downsample feature maps while preserving temporal structure
    input: [T, B, C, H, W] 
    returns: [T, B, C, H_out, W_out]
    """
    def __init__(self, kernel_size, stride):
        super().__init__()
        # max pooling reduces spatial resolution by taking the max in each kernel window
        # used instead of strided convolution on the shortcut (cheaper, no learned parameters)
        self.pool = nn.MaxPool2d(kernel_size, stride)

    def forward(self, x):
        # x: [T, B, C, H, W]
        # same pattern as SnnConv2d -> apply per time step, stack results
        return torch.stack([self.pool(x[t]) for t in range(x.shape[0])])

class MSBlock(nn.Module):
    """
    Multi-Scale Block - applied after residual addition in both EMS block types.
    Keeps spatial size AND channel count constant.

    main path: LIF -> Conv3x3 -> TDBN -> LIF -> Conv3x3 -> TDBN
    skip: identity (x unchanged)
    output: main + x
    """

    def __init__(self, in_ch, decay=0.25):
        super().__init__()

        # LCB 1
        self.lif1  = LIFNeuron(decay=decay)
        self.conv1 = SnnConv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1)
        self.bn1   = TDBN(in_ch)

        # LCB 2
        self.lif2  = LIFNeuron(decay=decay)
        self.conv2 = SnnConv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1)
        self.bn2   = TDBN(in_ch)

    def forward(self, x):
        # x: [T, B, in_ch, H, W]
        out = self.bn1(self.conv1(self.lif1(x)))
        out = self.bn2(self.conv2(self.lif2(out)))
        return out + x

# Type 1 Residual Block
class EMSBlock1(nn.Module):
    """
    - used when channel number is constant or decreasing
    main path: LIF -> Conv3x3 -> TDBN -> LIF -> Conv3x3 -> TDBN
    shortcut: identity
    output: main + shortcut
    """
    def __init__(self, in_ch, out_ch, stride=1, decay=0.25):
        super().__init__()

        # main path:

        # LCB
        self.lif1 = LIFNeuron(decay=decay) # converts input x to spikes before first conv

        self.conv1 = SnnConv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)

        self.bn1 = TDBN(out_ch) # normalise after conv1, before lif2

        # LCB
        self.lif2 = LIFNeuron(decay=decay) # converts tdbn1 output to spikes before second conv

        self.conv2 = SnnConv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)

        self.bn2 = TDBN(out_ch) # normalise after conv2, before addition with shortcut

        # MS Block
        self.ms_block = MSBlock(out_ch, decay=decay)

        # shortcut path:
        
        if stride != 1 or in_ch != out_ch:

            self.shortcut = nn.Sequential(
                SnnMaxPool2d(stride, stride), # match spatial size first
                LIFNeuron(decay=decay), # convert to spikes
                SnnConv2d(in_ch, out_ch, kernel_size=1), # match channels
                TDBN(out_ch),)
        else:
            self.shortcut = nn.Identity() # stride=1 and same channels: x passes through unchanged

    def forward(self, x):
        # x: [T, B, C, H, W]

        # main path:
        out = self.bn1(
            self.conv1(
                self.lif1(x)
                )
            ) # [T, B, out_ch, H', W']
        
        out = self.bn2(
            self.conv2(
                self.lif2(out)
                )
            ) # [T, B, out_ch, H', W']

        # shortcut path:
        shortcut = self.shortcut(x)

        # add main path and shortcut (eq.7: X_L = Add(Fr(X_{L-1}), Fs(X_{L-1})))
        return self.ms_block(out + shortcut) # [T, B, out_ch, H', W']

# Type 2 Residual Block
class EMSBlock2(nn.Module):
    """
    - used when channel number increases

    Main path: LIF -> Conv3x3 -> TDBN -> LIF -> Conv3x3 -> TDBN
    Shortcut path: MaxPool -> LIF -> Conv1x1 -> TDBN

    - concat(shortcut_features, pooled_input) to reach out_ch total
        -> concatenation preserves separate spike feature streams instead of merging them through projection addition
    output: main + concat_output
    """
    def __init__(self, in_ch, out_ch, stride=2, decay=0.25):
        super().__init__()

        assert out_ch > in_ch, "EMSBlock2 is for increasing channels only"

        # main path:

        # LCB
        self.lif1 = LIFNeuron(decay=decay)

        self.conv1 = SnnConv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)

        self.bn1 = TDBN(out_ch)

        # LCB
        self.lif2 = LIFNeuron(decay=decay)

        self.conv2 = SnnConv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)

        self.bn2 = TDBN(out_ch)

        # shortcut path:

        # first pool input
        self.pool = SnnMaxPool2d(kernel_size=stride, stride=stride)
        # then LCB
        self.shortcut_lif = LIFNeuron(decay=decay)

        self.shortcut_conv = SnnConv2d(in_ch, out_ch - in_ch, kernel_size=1, stride=1, padding=0)

        self.shortcut_bn = TDBN(out_ch - in_ch)

        # MS Block
        self.ms_block = MSBlock(out_ch, decay=decay)

        
    def forward(self, x):
        # x: [T, B, in_ch, H, W]

        # main path:
        out = self.bn1(
            self.conv1(
                self.lif1(x)
                )
            ) # [T, B, out_ch, H', W']
        
        out = self.bn2(
            self.conv2(
                self.lif2(out)
                )
            )# [ T, B, out_ch, H', W']

        # shortcut path:
        # pool input
        pooled_x = self.pool(x)

        shortcut = self.shortcut_bn(
            self.shortcut_conv(
                self.shortcut_lif(pooled_x)
            )
        ) # [T, B, out_ch - in_ch, H, W]

        # concatenate shortcut with x along channel dimension (dim=2 in [T, B, C, H, W])
        # shortcut: [T, B, out_ch - in_ch, H, W]
        # x: [T, B, in_ch, H, W]
        # result: [T, B, out_ch, H', W']
        shortcut = torch.cat([shortcut, pooled_x], dim=2)

        # add main path and shortcut
        return self.ms_block(out + shortcut) # [T, B, out_ch, H', W']
