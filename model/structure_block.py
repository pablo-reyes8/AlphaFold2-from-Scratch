"""AlphaFold-style structure module built from repeated IPA updates.

This module refines the single representation into residue frames through
Invariant Point Attention, transition layers, and backbone updates. It is the
bridge between latent features and explicit 3D geometry. It also exposes the
canonical AlphaFold-style rotation stop-gradient between blocks and an optional
backbone auxiliary loss over intermediate structures.
"""

import torch
import torch.nn as nn
import math

from model.invariant_point_attention import * 
from model.ipa_transformations import * 
from model.structure_transation import * 
from model.losses.fape_loss import FAPELoss
from model.losses.loss_helpers import build_backbone_frames


class StructureModule(nn.Module):
    """
    AF2-style structure module with optional block-specific parameters.

    Modes
    -----
    use_block_specific_params = False  (default)
        Reuses the same IPA / Transition / BackboneUpdate across all blocks.
        More memory efficient.

    use_block_specific_params = True
        Uses separate parameters per block via ModuleList.
        More canonical AF2-style.

    Notes
    -----
    - Rotation update always comes from BackboneUpdate.
    - Translation update can optionally come from separate linear heads
      when use_block_specific_params=True.
    - When enabled, intermediate backbone frames accumulate a backbone-only
      auxiliary FAPE term and stop gradients through orientations between
      blocks, matching the canonical AF2 training trick as closely as possible
      without moving the torsion head into this module.
    """

    def __init__(
        self,
        c_s=256,
        c_z=128,
        num_blocks=8,
        ipa_heads=8,
        ipa_scalar_dim=32,
        ipa_qk_points=4,
        ipa_v_points=8,
        transition_expansion=4,
        dropout=0.1,
        trans_scale_factor=10.0,
        use_block_specific_params=False,
        stop_rotation_gradients=True,
        aux_fape_enabled=True,
        aux_fape_length_scale=10.0,
        aux_fape_clamp_distance=10.0,
        aux_fape_eps=1e-12,
    ):
        super().__init__()

        self.num_blocks = num_blocks
        self.trans_scale_factor = trans_scale_factor
        self.use_block_specific_params = use_block_specific_params
        self.stop_rotation_gradients = bool(stop_rotation_gradients)
        self.aux_fape_enabled = bool(aux_fape_enabled)
        self.dropout = nn.Dropout(dropout)
        self.aux_fape_loss = FAPELoss(
            length_scale=aux_fape_length_scale,
            clamp_distance=aux_fape_clamp_distance,
            eps=aux_fape_eps,
        )
        self.last_aux_loss = None
        self.last_aux_per_block = None

        if self.use_block_specific_params:
            # More canonical: separate params per block
            self.ipas = nn.ModuleList([
                InvariantPointAttention(
                    c_s=c_s,
                    c_z=c_z,
                    num_heads=ipa_heads,
                    c_hidden=ipa_scalar_dim,
                    num_qk_points=ipa_qk_points,
                    num_v_points=ipa_v_points,
                )
                for _ in range(num_blocks)])

            self.transitions = nn.ModuleList([
                StructureTransition(
                    c_s=c_s,
                    expansion=transition_expansion,
                    dropout=dropout,
                )
                for _ in range(num_blocks)])

            self.backbone_updates = nn.ModuleList([
                BackboneUpdate(c_s=c_s)
                for _ in range(num_blocks)])

            # Separate translation heads per block
            self.translation_heads = nn.ModuleList([
                nn.Linear(c_s, 3)
                for _ in range(num_blocks)])

            for head in self.translation_heads:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)

        else:
            # Memory-efficient: shared params across all blocks
            self.ipa = InvariantPointAttention(
                c_s=c_s,
                c_z=c_z,
                num_heads=ipa_heads,
                c_hidden=ipa_scalar_dim,
                num_qk_points=ipa_qk_points,
                num_v_points=ipa_v_points)

            self.transition = StructureTransition(
                c_s=c_s,
                expansion=transition_expansion,
                dropout=dropout)

            self.backbone_update = BackboneUpdate(c_s=c_s)

    def _compute_aux_backbone_fape(
        self,
        *,
        R: torch.Tensor,
        t: torch.Tensor,
        R_true: torch.Tensor,
        t_true: torch.Tensor,
        coords_ca: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        return self.aux_fape_loss(
            R_pred=R.float(),
            t_pred=t.float(),
            x_pred=t.float(),
            R_true=R_true.float(),
            t_true=t_true.float(),
            x_true=coords_ca.float(),
            mask=(mask.float() if mask is not None else None),
        )

    def forward(
        self,
        s,
        z,
        mask=None,
        *,
        coords_n: torch.Tensor | None = None,
        coords_ca: torch.Tensor | None = None,
        coords_c: torch.Tensor | None = None,
        backbone_mask: torch.Tensor | None = None,
        return_aux: bool = False,
        return_intermediates: bool = False,
    ):
        """
        s: [B, L, c_s]
        z: [B, L, L, c_z]
        mask: [B, L]

        Optional backbone targets allow this module to accumulate the
        AlphaFold-style intermediate auxiliary loss on backbone frames.

        returns:
          s: [B, L, c_s]
          R: [B, L, 3, 3]
          t: [B, L, 3]
          aux_loss: scalar, only when return_aux=True
          intermediates: dict of stacked per-block tensors, only when
            return_intermediates=True
        """
        B, L, _ = s.shape
        device, dtype = s.device, s.dtype
        aux_mask = backbone_mask if backbone_mask is not None else mask

        can_compute_aux = (
            self.aux_fape_enabled
            and coords_n is not None
            and coords_ca is not None
            and coords_c is not None
        )
        R_true = None
        t_true = None
        if can_compute_aux:
            R_true, t_true = build_backbone_frames(
                coords_n=coords_n,
                coords_ca=coords_ca,
                coords_c=coords_c,
                mask=aux_mask,
            )

        R = torch.eye(3, device=device, dtype=dtype).view(1, 1, 3, 3).repeat(B, L, 1, 1)
        t = torch.zeros(B, L, 3, device=device, dtype=dtype)
        aux_losses = []
        s_intermediates = []
        R_intermediates = []
        t_intermediates = []

        for i in range(self.num_blocks):
            if self.use_block_specific_params:
                s = s + self.dropout(self.ipas[i](s, z, R, t, mask)[0])
                s = s + self.dropout(self.transitions[i](s, mask))

                # rotation update from block-specific BackboneUpdate
                dR, _ = self.backbone_updates[i](s, mask)

                # translation update from separate block-specific linear head
                dt = self.translation_heads[i](s) * self.trans_scale_factor

                if mask is not None:
                    dt = dt * mask.unsqueeze(-1)

            else:
                s = s + self.dropout(self.ipa(s, z, R, t, mask)[0])
                s = s + self.dropout(self.transition(s, mask))

                # shared BackboneUpdate returns both rotation and translation
                dR, dt = self.backbone_update(s, mask)

                if mask is not None:
                    dt = dt * mask.unsqueeze(-1)

            R, t = compose_frames(R, t, dR, dt)

            if return_intermediates:
                s_intermediates.append(s)
                R_intermediates.append(R)
                t_intermediates.append(t)

            if can_compute_aux:
                aux_losses.append(
                    self._compute_aux_backbone_fape(
                        R=R,
                        t=t,
                        R_true=R_true,
                        t_true=t_true,
                        coords_ca=coords_ca,
                        mask=aux_mask,
                    )
                )

            if self.stop_rotation_gradients and i < (self.num_blocks - 1):
                R = R.detach()

        if aux_losses:
            aux_per_block = torch.stack(aux_losses)
            aux_loss = aux_per_block.mean()
        else:
            aux_per_block = torch.zeros(0, device=device, dtype=dtype)
            aux_loss = torch.zeros((), device=device, dtype=dtype)

        self.last_aux_per_block = aux_per_block.detach()
        self.last_aux_loss = aux_loss.detach()
        intermediates = None
        if return_intermediates:
            intermediates = {
                "single": torch.stack(s_intermediates, dim=0),
                "R": torch.stack(R_intermediates, dim=0),
                "t": torch.stack(t_intermediates, dim=0),
            }

        if return_aux and return_intermediates:
            return s, R, t, aux_loss, intermediates
        if return_aux:
            return s, R, t, aux_loss
        if return_intermediates:
            return s, R, t, intermediates
        return s, R, t
