"""Auxiliary supervision over intermediate structure-module states.

This module mirrors the AlphaFold-style idea of supervising each structure
block with a lightweight backbone FAPE term and, when available, an
intermediate torsion-angle loss. It operates on stacked per-block predictions
emitted by the model forward pass.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model.losses.fape_loss import FAPELoss
from model.losses.torsion_loss import TorsionLoss


class StructureAuxLoss(nn.Module):
    """Compute mean auxiliary losses over intermediate structure-module blocks."""

    def __init__(
        self,
        *,
        fape_length_scale: float = 10.0,
        fape_clamp_distance: float = 10.0,
        fape_eps: float = 1e-12,
    ):
        super().__init__()
        self.fape_loss_fn = FAPELoss(
            length_scale=fape_length_scale,
            clamp_distance=fape_clamp_distance,
            eps=fape_eps,
        )
        self.torsion_loss_fn = TorsionLoss()

    @staticmethod
    def _zero_scalar(*, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros((), device=device, dtype=dtype)

    def forward(
        self,
        *,
        R_blocks: torch.Tensor | None,
        t_blocks: torch.Tensor | None,
        R_true: torch.Tensor,
        t_true: torch.Tensor,
        coords_ca: torch.Tensor,
        backbone_mask: torch.Tensor | None,
        torsion_blocks: torch.Tensor | None = None,
        torsion_true: torch.Tensor | None = None,
        torsion_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        device = coords_ca.device
        dtype = coords_ca.dtype
        zero = self._zero_scalar(device=device, dtype=dtype)

        aux_fape_loss = zero
        if (R_blocks is not None) and (t_blocks is not None):
            per_block_fape = []
            for block_idx in range(R_blocks.shape[0]):
                per_block_fape.append(
                    self.fape_loss_fn(
                        R_pred=R_blocks[block_idx].float(),
                        t_pred=t_blocks[block_idx].float(),
                        x_pred=t_blocks[block_idx].float(),
                        R_true=R_true.float(),
                        t_true=t_true.float(),
                        x_true=coords_ca.float(),
                        mask=backbone_mask,
                    )
                )
            aux_fape_loss = torch.stack(per_block_fape, dim=0).mean().to(dtype=dtype)

        aux_torsion_loss = zero
        if (
            (torsion_blocks is not None)
            and (torsion_true is not None)
            and (torsion_mask is not None)
        ):
            per_block_torsion = []
            for block_idx in range(torsion_blocks.shape[0]):
                per_block_torsion.append(
                    self.torsion_loss_fn(
                        torsion_pred=torsion_blocks[block_idx],
                        torsion_true=torsion_true,
                        torsion_mask=torsion_mask,
                    )
                )
            aux_torsion_loss = torch.stack(per_block_torsion, dim=0).mean().to(dtype=dtype)

        aux_loss = aux_fape_loss + aux_torsion_loss
        return {
            "aux_loss": aux_loss,
            "aux_fape_loss": aux_fape_loss,
            "aux_torsion_loss": aux_torsion_loss,
        }
