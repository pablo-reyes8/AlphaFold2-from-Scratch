"""Quaternion and rigid-frame utilities for structure updates.

This module converts quaternion parametrizations into rotation matrices and
composes rigid transformations. The structure code depends on these helpers to
maintain valid frame geometry during iterative refinement.
"""

import torch

def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    q: [..., 4]  quaternion normalized as [a, b, c, d]
    returns R: [..., 3, 3]
    """
    a, b, c, d = q.unbind(dim=-1)

    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    ab, ac, ad = a*b, a*c, a*d
    bc, bd, cd = b*c, b*d, c*d

    R = torch.stack([
        aa + bb - cc - dd,  2*(bc - ad),        2*(bd + ac),
        2*(bc + ad),        aa - bb + cc - dd,  2*(cd - ab),
        2*(bd - ac),        2*(cd + ab),        aa - bb - cc + dd], dim=-1)

    return R.reshape(q.shape[:-1] + (3, 3))

def compose_frames(R: torch.Tensor, t: torch.Tensor,
                   dR: torch.Tensor, dt: torch.Tensor):
    """
    R_new = R @ dR
    t_new = R @ dt + t

    R:  [B, L, 3, 3]
    t:  [B, L, 3]
    dR: [B, L, 3, 3]
    dt: [B, L, 3]
    """
    R_new = torch.matmul(R, dR)
    t_new = torch.matmul(R, dt.unsqueeze(-1)).squeeze(-1) + t
    return R_new, t_new
