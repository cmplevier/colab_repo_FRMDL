# LIF neuron, surrogate gradient, TDBN

import torch
import torch.nn as nn

# global hyperparameters
THRESHOLD = 0.65 # firing threshold
DECAY = 0.25 # decay - membrane leak factor
                    # forgets past input over time
                    # -> close to 1.0 - slow decay
                    # -> close to 0.0 - fast decay
TIME_STEPS = 5  # number of time steps T

class SurrogateSpikeFunction(torch.autograd.Function):
    # Heaviside step function in the forward pass, rectangular surrogate gradient in the backward pass
    # -> allow SNNs to be trained using gradient-based methods
    # -> replace non-differentiable spike function with a smooth approximation during backprop

    @staticmethod
    # forward: Heavyside step function
    # shape [T, B, C, H, W]
    def forward(context, x, threshold=THRESHOLD):
        context.save_for_backward(x) # stores the membrane potential 
        context.threshold = threshold # threshold needed for backward 
        return (x >= threshold).float() # returns 1.0 if threshold was reached (Heaviside step function)

    @staticmethod
    # backward: surrogate
    # shape: [T, B, C, H, W]
    def backward(context, gradient_output): # grad_output is the gradient of the loss w.r.t the output
        x = context.saved_tensors[0] # retrieves the stored membrane potential tensor 
        
        # surrogate gradient (eq.3): 1/a inside window [Vth - a/2, Vth + a/2], 0 outside
        # here a = 1.0, so the rectangular window is [threshold - 0.5, threshold + 0.5]
        gradient = gradient_output * (
            (x >= context.threshold - 0.5) & (x <= context.threshold + 0.5) # 1.0 when membrane potential within range
        ).float()

        return gradient, None # only interested in the gradient w.r.t. x


class LIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire neuron
    Receives the full temporal sequence [T, B, C, H, W] and processes each time step in a loop (the neuron accumulates input over time)
    
    x: input current 
    decay (tau):  decay factor for membrane leakage (eq.1)
    threshold (Vth): membrane potential required to fire a spike (eq.2)

    At each timestep:
        1. update membrane (eq.1)
        2. emit spike (eq.2)
        3. reset membrane - Vth is zeroed once neuron emits a spike (eq.1, reset term)
    """
    def __init__(self, decay=DECAY, threshold=THRESHOLD):
        super().__init__()
        self.decay = decay
        self.threshold = threshold

    def forward(self, x):
        # x: [T, B, C, H, W] - full temporal sequence

        T = x.shape[0] # the number of timesteps
        membrane = torch.zeros_like(x[0]) # V^(t,n+1)_i initialised to 0: [B, C, H, W]
        spike = torch.zeros_like(x[0]) # X^(t,n+1)_i initialised to 0: [B, C, H, W]
        output = torch.zeros_like(x) # container for X^(t,n+1)_i at each time step: [T, B, C, H, W]

        for t in range(T):
            # membrane update (eq.1)
            # V^(t+1,n+1)_i = tau * V^(t,n+1)_i * (1 - X^(t,n+1)_i) + sum_j(W^n_(ij) * X^(t+1,n)_j)
            # (1 - spike.detach()) resets membrane to 0 where a spike fired in the previous step
            # x[t] is the weighted input from the previous layer: sum_j(W^n_ij * X^(t+1,n)_j)
            membrane = self.decay * membrane * (1.0 - spike.detach()) + x[t]

            # spike emission (eq.2)
            # X^(t+1,n+1)_i = H(V^(t+1,n+1)_i - V_(th))
            # H is the Heaviside step function: 1.0 if membrane >= threshold, 0.0 otherwise
            spike = SurrogateSpikeFunction.apply(membrane, self.threshold)

            output[t] = spike

        return output # [T, B, C, H, W]


class TDBN(nn.Module):
    """
    Threshold-dependent Batch Normalization (eq.4 and eq.5) 
        - keeps neuron activity in a usable range, prevents dead or saturated neurons

    Replaces the raw weighted input sum in Eq. 1 with a normalised version (Eq. 4):
        V^(t+1,n+1)_i = tau * V^(t,n+1)_i * (1 - X^(t,n+1)_i) + TDBN(I^(t+1)_i)

    TDBN is defined as:
    TDBN(I) = lambda_i * (alpha * Vth) * (I - mu_ci) / sqrt(sigma^2_ci + eps) + beta_i
    where mu_ci and sigma^2_ci are the channel mean and variance over the mini-batch,
    alpha * Vth is the threshold-dependent scale factor,
    and lambda_i, beta_i are learnable parameters.

    In practice this is BatchNorm3d with gamma initialised to Vth (the threshold)
    rather than the default 1.0, which puts the output in the right range for
    the LIF neuron from the start of training.

    Input/output shape: [T, B, C, H, W]
    """
    def __init__(self, num_features):
        super().__init__()
        self.bn = nn.BatchNorm3d(num_features)
        nn.init.constant_(self.bn.weight, THRESHOLD) # lambda initialised to Vth (eq.5)
        nn.init.zeros_(self.bn.bias) # beta initialised to 0 (eq.5)

    def forward(self, x):
        # x: [T, B, C, H, W] 
        # goal: [B, C, T, H, W] for BatchNorm3d's expected [N, C, D, H, W]
        y = x.transpose(0, 2).contiguous() # [C, B, T, H, W]
        y = y.transpose(0, 1).contiguous() # [B, C, T, H, W]
        y = self.bn(y)
        y = y.transpose(0, 1).contiguous() # [C, B, T, H, W]
        y = y.transpose(0, 2).contiguous() # [T, B, C, H, W]
        return y