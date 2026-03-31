"""AlphaFold-style structure module built from repeated IPA updates.

This module refines the single representation into residue frames through
Invariant Point Attention, transition layers, and backbone updates. It is the
bridge between latent features and explicit 3D geometry.
"""

import torch
import torch.nn as nn
import math

from model.invariant_point_attention import * 
from model.ipa_transformations import * 
from model.structure_transation import * 


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
    ):
        super().__init__()

        self.num_blocks = num_blocks
        self.trans_scale_factor = trans_scale_factor
        self.use_block_specific_params = use_block_specific_params
        self.dropout = nn.Dropout(dropout)

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

    def forward(self, s, z, mask=None):
        """
        s: [B, L, c_s]
        z: [B, L, L, c_z]
        mask: [B, L]

        returns:
          s: [B, L, c_s]
          R: [B, L, 3, 3]
          t: [B, L, 3]
        """
        B, L, _ = s.shape
        device, dtype = s.device, s.dtype

        R = torch.eye(3, device=device, dtype=dtype).view(1, 1, 3, 3).repeat(B, L, 1, 1)
        t = torch.zeros(B, L, 3, device=device, dtype=dtype)

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

        return s, R, t
