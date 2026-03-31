"""Torsion-angle supervision for predicted angle vectors.

This file contains the auxiliary torsion loss that compares predicted and true
sin-cos angle encodings under a residue and torsion mask. It is intentionally
simple and plugs into the global AlphaFold-style loss dictionary.
"""

import torch 
import torch.nn as nn

class TorsionLoss(nn.Module):
    """
    Torsion angle loss.

    Compares predicted normalized 2D torsion vectors [sin(theta), cos(theta)]
    against true torsion vectors using squared Euclidean error (MSE-style).

    Inputs
    ------
    torsion_pred : [B, L, T, 2]
    torsion_true : [B, L, T, 2]
    torsion_mask : [B, L, T]   or broadcastable to that

    Returns
    -------
    loss : scalar
    """

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, torsion_pred, torsion_true, torsion_mask=None):
        """
        torsion_pred: [B, L, T, 2]
        torsion_true: [B, L, T, 2]
        torsion_mask: [B, L, T]
        """
        sq_error = ((torsion_pred - torsion_true) ** 2).sum(dim=-1)   # [B,L,T]

        if torsion_mask is not None:
            sq_error = sq_error * torsion_mask
            denom = torsion_mask.sum().clamp_min(1.0)
        else:
            denom = torch.tensor(
                sq_error.numel(),
                device=sq_error.device,
                dtype=sq_error.dtype)

        loss = sq_error.sum() / denom
        return loss
