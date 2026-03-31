"""Helper functions shared across structural loss modules.

This module provides vector normalization and canonical backbone-frame
construction from N, CA, and C coordinates. The full loss orchestrator uses
these utilities to convert raw coordinates into rigid frames.
"""

import torch

def normalize_vec(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(eps)


def build_backbone_frames(
    coords_n: torch.Tensor,
    coords_ca: torch.Tensor,
    coords_c: torch.Tensor,
    mask: torch.Tensor = None,
    eps: float = 1e-8):
    """
    Build per-residue backbone frames from N, CA, C coordinates.

    Inputs
    ------
    coords_n  : [B, L, 3]
    coords_ca : [B, L, 3]
    coords_c  : [B, L, 3]
    mask      : [B, L] optional backbone-valid mask

    Returns
    -------
    R : [B, L, 3, 3]
        Rotation matrix whose columns are orthonormal frame axes.
    t : [B, L, 3]
        Translation = CA coordinates
    """
    # origin
    t = coords_ca  # [B, L, 3]

    # first axis: CA -> C
    e1 = coords_c - coords_ca
    e1 = normalize_vec(e1, eps=eps)

    # helper vector: CA -> N
    u2 = coords_n - coords_ca

    # Gram-Schmidt: remove projection of u2 onto e1
    proj = (u2 * e1).sum(dim=-1, keepdim=True) * e1
    e2 = u2 - proj
    e2 = normalize_vec(e2, eps=eps)

    # third axis
    e3 = torch.cross(e1, e2, dim=-1)
    e3 = normalize_vec(e3, eps=eps)

    # optional re-orthogonalization of e2 for extra numerical stability
    e2 = torch.cross(e3, e1, dim=-1)
    e2 = normalize_vec(e2, eps=eps)

    # Stack as columns: R @ x_local + t = x_global
    R = torch.stack([e1, e2, e3], dim=-1)  # [B, L, 3, 3] # R tiene los ejes ortonormales del frame local 
    # x_global = R*x_local + t 

    if mask is not None:
        eye = torch.eye(3, device=R.device, dtype=R.dtype).view(1, 1, 3, 3)
        R = torch.where(mask[..., None, None].bool(), R, eye)
        t = torch.where(mask[..., None].bool(), t, torch.zeros_like(t))

    return R, t
