"""Per-residue pLDDT supervision built from coordinate agreement.

This module approximates lDDT targets from predicted and true coordinates,
discretizes those scores into bins, and applies cross-entropy to the pLDDT
head outputs used as a confidence auxiliary loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PlddtLoss(nn.Module):
    """
    pLDDT supervision loss.

    Builds a true per-residue lDDT-like score from predicted and true coordinates,
    discretizes it into bins, and applies cross-entropy against pLDDT logits.

    Inputs
    ------
    plddt_logits : [B, L, num_bins]
    x_pred       : [B, L, 3]
    x_true       : [B, L, 3]
    mask         : [B, L]

    Returns
    -------
    loss : scalar
    """

    def __init__(
        self,
        num_bins=50,
        inclusion_radius=15.0,
        eps=1e-8):
      
        super().__init__()
        self.num_bins = num_bins
        self.inclusion_radius = inclusion_radius
        self.eps = eps

    def forward(self, plddt_logits, x_pred, x_true, mask=None):
        """
        plddt_logits: [B, L, num_bins]
        x_pred:       [B, L, 3]
        x_true:       [B, L, 3]
        mask:         [B, L]
        """
        B, L, num_bins = plddt_logits.shape

        assert num_bins == self.num_bins, (
            f"Expected num_bins={self.num_bins}, got {num_bins}")


        # Pairwise distances
        diff_true = x_true[:, :, None, :] - x_true[:, None, :, :]   # [B,L,L,3]
        diff_pred = x_pred[:, :, None, :] - x_pred[:, None, :, :]   # [B,L,L,3]

        d_true = torch.sqrt((diff_true ** 2).sum(dim=-1) + self.eps)  # [B,L,L]
        d_pred = torch.sqrt((diff_pred ** 2).sum(dim=-1) + self.eps)  # [B,L,L]

        dist_error = torch.abs(d_pred - d_true)  # [B,L,L]


        # --------------------------------------------------
        # Valid comparison pairs for lDDT:
        # - true distance within radius
        # - not self-pair
        # - both residues valid
        # --------------------------------------------------
        if mask is None:
            mask = torch.ones(B, L, device=x_true.device, dtype=x_true.dtype)

        pair_mask = mask[:, :, None] * mask[:, None, :]  # [B,L,L]

        not_self = 1.0 - torch.eye(L, device=x_true.device, dtype=x_true.dtype).unsqueeze(0)  # [1,L,L]
        within_radius = (d_true < self.inclusion_radius).to(x_true.dtype)

        valid_pairs = pair_mask * not_self * within_radius   

        # lDDT-style agreement scores at thresholds
        s_05 = (dist_error < 0.5).to(x_true.dtype)
        s_10 = (dist_error < 1.0).to(x_true.dtype)
        s_20 = (dist_error < 2.0).to(x_true.dtype)
        s_40 = (dist_error < 4.0).to(x_true.dtype)

        pair_score = 0.25 * (s_05 + s_10 + s_20 + s_40)  # [B,L,L], in [0,1]

        # --------------------------------------------------
        # Per-residue true lDDT target
        # For each residue i, average across valid j
        # --------------------------------------------------
        numer = (pair_score * valid_pairs).sum(dim=-1)          
        denom = valid_pairs.sum(dim=-1).clamp_min(1.0)         

        true_lddt = numer / denom                                # [B,L] in [0,1]
        true_score = 100.0 * true_lddt                           # [B,L] in [0,100]

        # --------------------------------------------------
        # Discretize score into num_bins bins
        # Bin width = 100 / num_bins
        # Values in [0,100], cap 100 -> last bin
        # --------------------------------------------------
        bin_index = torch.floor(true_score * self.num_bins / 100.0).long()
        bin_index = torch.clamp(bin_index, min=0, max=self.num_bins - 1)  # [B,L]

        # --------------------------------------------------
        # CE over residues with valid supervision
        # A residue is supervised if:
        # - mask[i] valid
        # - it has at least one valid neighbor
        # --------------------------------------------------
        residue_valid = (mask > 0) & (valid_pairs.sum(dim=-1) > 0)   # [B,L]

        logits_flat = plddt_logits.reshape(B * L, self.num_bins)
        targets_flat = bin_index.reshape(B * L)
        valid_flat = residue_valid.reshape(B * L)

        per_res_loss = F.cross_entropy(logits_flat,
            targets_flat,
            reduction="none")  # [B*L]

        per_res_loss = per_res_loss * valid_flat.to(per_res_loss.dtype)
        denom = valid_flat.sum().clamp_min(1)

        loss = per_res_loss.sum() / denom
        return loss
