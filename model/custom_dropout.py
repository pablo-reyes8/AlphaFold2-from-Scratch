"""Shared AlphaFold-style dropout utilities.

This module centralizes the shared-mask dropout variants used by the Evoformer
family of stacks so the same canonical behavior can be reused in the main
trunk, template stack, and extra MSA stack.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SharedDropout(nn.Module):
    """Dropout with a mask shared across a chosen tensor dimension."""

    def __init__(self, p: float, shared_dim: int):
        super().__init__()
        if not (0.0 <= p < 1.0):
            raise ValueError(f"dropout probability must be in [0, 1), got {p}")
        self.p = float(p)
        self.shared_dim = int(shared_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) or self.p == 0.0:
            return x

        ndim = x.ndim
        shared_dim = self.shared_dim if self.shared_dim >= 0 else ndim + self.shared_dim
        if not (0 <= shared_dim < ndim):
            raise ValueError(
                f"shared_dim={self.shared_dim} is invalid for tensor with ndim={ndim}"
            )

        mask_shape = list(x.shape)
        mask_shape[shared_dim] = 1

        keep_prob = 1.0 - self.p
        mask = x.new_empty(mask_shape).bernoulli_(keep_prob) / keep_prob
        return x * mask


class DropoutRowwise(SharedDropout):
    """Share a dropout mask across the row dimension."""

    def __init__(self, p: float):
        super().__init__(p=p, shared_dim=-3)


class DropoutColumnwise(SharedDropout):
    """Share a dropout mask across the column dimension."""

    def __init__(self, p: float):
        super().__init__(p=p, shared_dim=-2)
