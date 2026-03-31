"""Rigid-frame coordinate transformation helpers for IPA and structure code.

The functions defined here move points between local and global frames and are
shared by the structure module, FAPE loss, and full model forward pass when
reconstructing backbone coordinates from local templates.
"""

import torch

def apply_transform(R: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Transforma puntos del marco local al global.

    x_global = R @ x_local + t

    Args:
        R: [..., 3, 3]
        t: [..., 3]
        x: [..., 3]

    Returns:
        x_global: [..., 3]
    """
    return torch.matmul(R, x.unsqueeze(-1)).squeeze(-1) + t


def invert_apply_transform(R: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Transforma puntos del marco global al local.

    x_local = R^T @ (x_global - t)

    Args:
        R: [..., 3, 3]
        t: [..., 3]
        x: [..., 3]

    Returns:
        x_local: [..., 3]
    """
    x_shifted = x - t
    R_t = R.transpose(-1, -2)
    return torch.matmul(R_t, x_shifted.unsqueeze(-1)).squeeze(-1)
