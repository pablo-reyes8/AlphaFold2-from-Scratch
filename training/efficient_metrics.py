"""Fast structure metrics computed from predicted and true coordinates.

This module exposes efficient batched RMSD, TM-score, and GDT-TS calculations
for logging during training. The goal is to provide cheap structural signals
without coupling the training loop to notebook utilities.
"""

import torch
from training.metrics_utils import *

@torch.no_grad()
def compute_structure_metrics(
    x_pred: torch.Tensor,
    x_true: torch.Tensor,
    mask: torch.Tensor,
    align: bool = True,
    thresholds=(1.0, 2.0, 4.0, 8.0),
    eps: float = 1e-8):

    """
    Compute RMSD, TM-score, and GDT-TS efficiently.

    Inputs
    ------
    x_pred : [B, L, 3]
    x_true : [B, L, 3]
    mask   : [B, L]

    Returns
    -------
    metrics : dict
        {
            "rmsd_per_sample": [B],
            "tm_per_sample":   [B],
            "gdt_ts_per_sample":[B],
            "rmsd": scalar,
            "tm_score": scalar,
            "gdt_ts": scalar
        }
    """
    if align:
        x_pred_use, _, _ = kabsch_align(x_pred, x_true, mask, eps=eps)
    else:
        x_pred_use = x_pred

    mask_f = mask.to(x_pred_use.dtype)
    L_eff = mask_f.sum(dim=-1).clamp_min(1.0)  # [B]

    # Distancias por residuo [B, L]
    sq_err = ((x_pred_use - x_true) ** 2).sum(dim=-1)
    dist = torch.sqrt(sq_err + eps)


    # RMSD
    mse_per_sample = (sq_err * mask_f).sum(dim=-1) / L_eff
    rmsd_per_sample = torch.sqrt(mse_per_sample + eps)
    rmsd_mean = rmsd_per_sample.mean()


    # TM-score
    d0 = 1.24 * torch.clamp(L_eff - 15.0, min=1.0) ** (1.0 / 3.0) - 1.8
    d0 = torch.clamp(d0, min=0.5)

    tm_terms = 1.0 / (1.0 + (dist / d0[:, None]) ** 2)
    tm_per_sample = (tm_terms * mask_f).sum(dim=-1) / L_eff
    tm_mean = tm_per_sample.mean()

    # GDT-TS
    gdt_parts = []
    for thr in thresholds:
        within = (dist <= thr).to(dist.dtype)
        frac = (within * mask_f).sum(dim=-1) / L_eff
        gdt_parts.append(frac)

    gdt_ts_per_sample = torch.stack(gdt_parts, dim=-1).mean(dim=-1)
    gdt_ts_mean = gdt_ts_per_sample.mean()

    return {
        "rmsd_per_sample": rmsd_per_sample,
        "tm_per_sample": tm_per_sample,
        "gdt_ts_per_sample": gdt_ts_per_sample,
        "rmsd": rmsd_mean,
        "tm_score": tm_mean,
        "gdt_ts": gdt_ts_mean}
