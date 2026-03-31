"""Position-wise transition network for MSA activations.

This module provides the feed-forward transition applied after MSA attention
updates. It is the MSA counterpart of the transition blocks commonly used in
transformer-style architectures and Evoformer implementations.
"""

import torch
import torch.nn as nn


def zero_init_linear(linear: nn.Linear):
    nn.init.zeros_(linear.weight)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


class MSATransition(nn.Module):
    def __init__(self, c_m=256, expansion=4):
        super().__init__()
        self.layer_norm = nn.LayerNorm(c_m)
        self.linear_1 = nn.Linear(c_m, expansion * c_m)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(expansion * c_m, c_m)

        # zero-init final projection
        zero_init_linear(self.linear_2)

    def forward(self, m, msa_mask=None):
        x = self.layer_norm(m)
        x = self.linear_1(x)
        x = self.act(x)
        x = self.linear_2(x)

        if msa_mask is not None:
            x = x * msa_mask.unsqueeze(-1)
        return x

class PairTransition(nn.Module):
    def __init__(self, c_z=128, expansion=4):
        super().__init__()
        self.layer_norm = nn.LayerNorm(c_z)
        self.linear_1 = nn.Linear(c_z, expansion * c_z)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(expansion * c_z, c_z)

        # zero-init final projection
        zero_init_linear(self.linear_2)

    def forward(self, z, pair_mask=None):
        x = self.layer_norm(z)
        x = self.linear_1(x)
        x = self.act(x)
        x = self.linear_2(x)

        if pair_mask is not None:
            x = x * pair_mask.unsqueeze(-1)
        return x
