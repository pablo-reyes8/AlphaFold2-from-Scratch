"""Distogram supervision used by the AlphaFold-style loss.

This module discretizes true pairwise distances into bins and applies
cross-entropy against distogram logits predicted from the pair representation.
It provides one of the auxiliary geometric objectives in training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DistogramLoss(nn.Module):
    """
    Distogram cross-entropy loss.

    Takes true coordinates, builds true pairwise distance matrix,
    discretizes distances into bins, and applies cross-entropy against
    distogram logits.

    Inputs
    ------
    distogram_logits : [B, L, L, num_bins]
    x_true           : [B, L, 3]
    mask             : [B, L]

    Returns
    -------
    loss : scalar
    """

    def __init__(self, num_bins=64, min_bin=2.0, max_bin=22.0, eps=1e-8):
        super().__init__()
        self.num_bins = num_bins
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.eps = eps

        # num_bins-1 thresholds define num_bins classes
        boundaries = torch.linspace(min_bin, max_bin, steps=num_bins - 1)
        self.register_buffer("boundaries", boundaries)

    def forward(self, distogram_logits, x_true, mask=None):
        """
        distogram_logits: [B, L, L, num_bins]
        x_true:           [B, L, 3]
        mask:             [B, L]
        """
        B, L, _, num_bins = distogram_logits.shape
        assert num_bins == self.num_bins, (
            f"Expected num_bins={self.num_bins}, got {num_bins}")

        # True pairwise distances
        diff = x_true[:, :, None, :] - x_true[:, None, :, :]   
        d_true = torch.sqrt((diff ** 2).sum(dim=-1) + self.eps) 

        # Discretize distances into bin indices in [0, num_bins-1]
        # bucketize with num_bins-1 boundaries gives exactly num_bins classes
        target_bins = torch.bucketize(d_true, self.boundaries)   # [B,L,L], long



        # Flatten for cross entropy
        # logits CE expects [N, C], target [N]
        logits_flat = distogram_logits.reshape(B * L * L, self.num_bins)
        targets_flat = target_bins.reshape(B * L * L)

        per_pair_loss = F.cross_entropy(logits_flat,
            targets_flat,
            reduction="none").reshape(B, L, L)

        # Mask valid residue pairs
        if mask is not None:
            pair_mask = mask[:, :, None] * mask[:, None, :]  
            per_pair_loss = per_pair_loss * pair_mask
            denom = pair_mask.sum().clamp_min(1.0)
        else:
            denom = torch.tensor(B * L * L, device=per_pair_loss.device, dtype=per_pair_loss.dtype)

        loss = per_pair_loss.sum() / denom
        return loss
