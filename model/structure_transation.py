"""Single-representation transition and backbone-update layers.

This file contains the feed-forward transition used inside the structure module
and the rigid-frame update head that predicts rotation and translation deltas
for each residue during structural refinement.
"""

import torch
import torch.nn as nn
import math
from model.quaternion_to_matrix import *


class StructureTransition(nn.Module):
    """
    Canonical AF2 Structure Module transition:
    3 linear layers with ReLU and width c_s.
    """
    def __init__(self, c_s=256, dropout=0.1):
        super().__init__()
        self.lin1 = nn.Linear(c_s, c_s)
        self.lin2 = nn.Linear(c_s, c_s)
        self.lin3 = nn.Linear(c_s, c_s)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        nn.init.zeros_(self.lin3.weight)
        nn.init.zeros_(self.lin3.bias)

    def forward(self, s, mask=None):
        x = self.lin1(s)
        x = self.act(x)
        x = self.dropout(x)

        x = self.lin2(x)
        x = self.act(x)
        x = self.dropout(x)

        x = self.lin3(x)

        if mask is not None:
            x = x * mask.unsqueeze(-1)
        return x
    

class BackboneUpdate(nn.Module):
    """
    Predicts local frame update:
      - dt in R^3
      - quaternion q = [1, b, c, d], then normalize
    """
    def __init__(self, c_s=256):
        super().__init__()
        self.linear = nn.Linear(c_s, 6)

        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, s, mask=None):
        """
        s: [B, L, c_s]
        returns:
          dR: [B, L, 3, 3]
          dt: [B, L, 3]
        """
        out = self.linear(s)

        dt = out[..., :3]
        bcd = out[..., 3:]    # [B, L, 3]

        ones = torch.ones_like(bcd[..., :1])
        q = torch.cat([ones, bcd], dim=-1)  # [B, L, 4]
        q = q / torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(1e-8)

        dR = quaternion_to_rotation_matrix(q)

        if mask is not None:
            dt = dt * mask.unsqueeze(-1)
            eye = torch.eye(3, device=s.device, dtype=s.dtype).view(1, 1, 3, 3)
            dR = torch.where(mask[..., None, None].bool(), dR, eye)

        return dR, dt
