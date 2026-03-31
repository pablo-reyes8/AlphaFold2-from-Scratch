"""Higher-level AlphaFold-oriented metric wrappers.

This module builds on the lower-level geometry helpers to expose convenient
metric functions such as RMSD for AlphaFold-style predictions. It is kept
separate from the core utilities to simplify call sites.
"""

import torch
from training.metrics_utils import *


def rmsd_metric(
    x_pred: torch.Tensor,
    x_true: torch.Tensor,
    mask: torch.Tensor,
    align: bool = True,
    eps: float = 1e-8):

    """
    RMSD per structure and mean RMSD over batch.

    Inputs
    ------
    x_pred : [B, L, 3]
    x_true : [B, L, 3]
    mask   : [B, L]

    Returns
    -------
    rmsd_per_sample : [B]
    rmsd_mean       : scalar tensor
    """
    if align:
        x_pred_use, _, _ = kabsch_align(x_pred, x_true, mask, eps=eps)
    else:
        x_pred_use = x_pred

    sq_err = ((x_pred_use - x_true) ** 2).sum(dim=-1)   # [B,L]
    mask_f = mask.to(sq_err.dtype)

    mse_per_sample = (sq_err * mask_f).sum(dim=-1) / mask_f.sum(dim=-1).clamp_min(1.0)
    rmsd_per_sample = torch.sqrt(mse_per_sample + eps)
    rmsd_mean = rmsd_per_sample.mean()

    return rmsd_per_sample, rmsd_mean

def tm_score_metric(
    x_pred: torch.Tensor,
    x_true: torch.Tensor,
    mask: torch.Tensor,
    align: bool = True,
    eps: float = 1e-8):

    """
    TM-score per structure and batch mean.

    Inputs
    ------
    x_pred : [B, L, 3]
    x_true : [B, L, 3]
    mask   : [B, L]

    Returns
    -------
    tm_per_sample : [B]
    tm_mean       : scalar tensor
    """
    if align:
        x_pred_use, _, _ = kabsch_align(x_pred, x_true, mask, eps=eps)
    else:
        x_pred_use = x_pred

    dist = torch.sqrt(((x_pred_use - x_true) ** 2).sum(dim=-1) + eps)  # [B,L]
    mask_f = mask.to(dist.dtype)

    L_eff = mask_f.sum(dim=-1).clamp_min(1.0)  # [B]

    # Standard-ish d0 formula, clamped for small proteins
    d0 = 1.24 * torch.clamp(L_eff - 15.0, min=1.0) ** (1.0 / 3.0) - 1.8
    d0 = torch.clamp(d0, min=0.5)  # avoid pathological small values

    score = 1.0 / (1.0 + (dist / d0[:, None]) ** 2)  # [B,L]
    tm_per_sample = (score * mask_f).sum(dim=-1) / L_eff
    tm_mean = tm_per_sample.mean()

    return tm_per_sample, tm_mean

def gdt_ts_metric(
    x_pred: torch.Tensor,
    x_true: torch.Tensor,
    mask: torch.Tensor,
    align: bool = True,
    thresholds=(1.0, 2.0, 4.0, 8.0),
    eps: float = 1e-8):

    """
    GDT-TS per structure and batch mean.

    Inputs
    ------
    x_pred : [B, L, 3]
    x_true : [B, L, 3]
    mask   : [B, L]

    Returns
    -------
    gdt_per_sample : [B]
    gdt_mean       : scalar tensor
    """
    if align:
        x_pred_use, _, _ = kabsch_align(x_pred, x_true, mask, eps=eps)
    else:
        x_pred_use = x_pred

    dist = torch.sqrt(((x_pred_use - x_true) ** 2).sum(dim=-1) + eps)  # [B,L]
    mask_f = mask.to(dist.dtype)
    L_eff = mask_f.sum(dim=-1).clamp_min(1.0)

    scores = []
    for thr in thresholds:
        within = (dist <= thr).to(dist.dtype)
        frac = (within * mask_f).sum(dim=-1) / L_eff
        scores.append(frac)

    gdt_per_sample = torch.stack(scores, dim=-1).mean(dim=-1)  # [B]
    gdt_mean = gdt_per_sample.mean()

    return gdt_per_sample, gdt_mean


