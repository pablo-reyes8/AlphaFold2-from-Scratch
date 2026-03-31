"""Frame Aligned Point Error loss for structural supervision.

This file measures predicted point errors after expressing coordinates in local
backbone frames. The resulting FAPE signal is one of the main geometric losses
used to supervise frame and position predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.ipa_transformations import *

class FAPELoss(nn.Module):
    """
    Frame Aligned Point Error (FAPE)

    Computes the position error of atom j measured in the local frame of residue i,
    comparing predicted vs true structures.

    Inputs
    ------
    R_pred : [B, L, 3, 3]
    t_pred : [B, L, 3]
    x_pred : [B, L, 3]      predicted atom coordinates (PoC: C_alpha = t_pred)

    R_true : [B, L, 3, 3]
    t_true : [B, L, 3]
    x_true : [B, L, 3]      true atom coordinates

    mask   : [B, L]         valid residue mask

    Returns
    -------
    loss : scalar
    """

    def __init__(self, length_scale=10.0, clamp_distance=10.0, eps=1e-8):
        super().__init__()
        self.length_scale = length_scale
        self.clamp_distance = clamp_distance
        self.eps = eps

    def forward(
        self,
        R_pred, t_pred, x_pred,
        R_true, t_true, x_true,
        mask=None):
      
        """
        Returns scalar FAPE loss.
        """
        B, L, _ = x_pred.shape

        # --------------------------------------------------
        # Expand so that every atom x_j is evaluated in every frame T_i
        # We want:
        #   frames indexed by i   -> [B, L, 1, ...]
        #   points indexed by j   -> [B, 1, L, 3]
        # Result after inverse transform: [B, L, L, 3]
        # --------------------------------------------------
        R_pred_exp = R_pred.unsqueeze(2)   # [B, L, 1, 3, 3]
        t_pred_exp = t_pred.unsqueeze(2)   # [B, L, 1, 3]
        x_pred_exp = x_pred.unsqueeze(1)   # [B, 1, L, 3]

        R_true_exp = R_true.unsqueeze(2)   # [B, L, 1, 3, 3]
        t_true_exp = t_true.unsqueeze(2)   # [B, L, 1, 3]
        x_true_exp = x_true.unsqueeze(1)   # [B, 1, L, 3]

        # --------------------------------------------------
        # Convert all points x_j into each residue-local frame i
        # x_local[i,j] = T_i^{-1}(x_j)
        # Shape: [B, L, L, 3]
        # --------------------------------------------------
        x_pred_local = invert_apply_transform(R_pred_exp, t_pred_exp, x_pred_exp)
        x_true_local = invert_apply_transform(R_true_exp, t_true_exp, x_true_exp)

        # Pointwise local-frame error
        error = torch.sqrt(((x_pred_local - x_true_local) ** 2).sum(dim=-1) + self.eps) 

        # clip distance to stabilize gradients
        error = torch.clamp(error, max=self.clamp_distance)

        # normalize, as in AF-style FAPE
        error = error / self.length_scale

        # Masking
        # Pair (i,j) is valid only if both residues are valid
        if mask is not None:
            pair_mask = mask[:, :, None] * mask[:, None, :] 
            error = error * pair_mask
            denom = pair_mask.sum().clamp_min(1.0)
        else:
            denom = torch.tensor(B * L * L, device=error.device, dtype=error.dtype)

        loss = error.sum() / denom
        return loss
