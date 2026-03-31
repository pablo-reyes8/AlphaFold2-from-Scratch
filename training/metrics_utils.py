"""Low-level geometry and masking utilities for metric computation.

The functions in this file implement masked reductions, centering, Kabsch
alignment, and related tensor operations reused by the higher-level structural
metrics that appear during evaluation and logging.
"""

import math
import torch
import torch.nn.functional as F


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim=None, keepdim=False, eps: float = 1e-8):
    mask = mask.to(x.dtype)
    num = (x * mask).sum(dim=dim, keepdim=keepdim)
    den = mask.sum(dim=dim, keepdim=keepdim).clamp_min(eps)
    return num / den


def center_coordinates(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8):
    """
    x:    [B, L, 3]
    mask: [B, L]

    returns:
      x_centered: [B, L, 3]
      centroid:   [B, 1, 3]
    """
    centroid = masked_mean(x, mask[..., None], dim=1, keepdim=True, eps=eps)
    x_centered = x - centroid
    x_centered = x_centered * mask[..., None]
    return x_centered, centroid


def kabsch_align(x_pred: torch.Tensor, x_true: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8):
    """
    Batched Kabsch alignment.

    Inputs
    ------
    x_pred : [B, L, 3]
    x_true : [B, L, 3]
    mask   : [B, L]

    Returns
    -------
    x_pred_aligned : [B, L, 3]
    R              : [B, 3, 3]
    t              : [B, 1, 3]
    """
    B, L, _ = x_pred.shape
    mask_f = mask.to(x_pred.dtype)

    x_pred_c, pred_centroid = center_coordinates(x_pred, mask, eps=eps)
    x_true_c, true_centroid = center_coordinates(x_true, mask, eps=eps)

    # Covariance: H = X_pred^T X_true
    H = torch.matmul((x_pred_c * mask_f[..., None]).transpose(1, 2), x_true_c)  # [B, 3, 3]

    U, S, Vh = torch.linalg.svd(H, full_matrices=False)
    V = Vh.transpose(-1, -2)

    # Reflection correction
    det = torch.det(torch.matmul(V, U.transpose(-1, -2)))
    D = torch.eye(3, device=x_pred.device, dtype=x_pred.dtype).unsqueeze(0).repeat(B, 1, 1)
    D[:, -1, -1] = torch.where(det < 0, -1.0, 1.0)

    R = torch.matmul(torch.matmul(V, D), U.transpose(-1, -2))  # [B,3,3]

    x_pred_aligned = torch.matmul(x_pred_c, R.transpose(-1, -2)) + true_centroid
    x_pred_aligned = x_pred_aligned * mask[..., None]

    t = true_centroid - torch.matmul(pred_centroid, R.transpose(-1, -2))
    return x_pred_aligned, R, t
