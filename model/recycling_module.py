"""Recycling embedder for re-injecting previous trunk representations.

This module implements the AlphaFold2-style recycling step that updates the
first MSA row and pair representation before the Evoformer using recycled
single, pair, and position-derived distance information.
"""

import torch
import torch.nn as nn


class RecyclingEmbedder(nn.Module):
    """
    AlphaFold2 recycling embedder.

    This module injects recycled information into:
      - the first MSA row (target row),
      - the pair representation z,
      - and z again via distance-binned embeddings from previous positions.

    Inputs
    ------
    m : [B, N_msa, L, c_m]
    z : [B, L, L, c_z]

    prev_m1 : [B, L, c_m] or None
        Previous first-row MSA representation.

    prev_z : [B, L, L, c_z] or None
        Previous pair representation.

    prev_positions : [B, L, 3] or None
        Previous pseudo-beta / C-beta-like positions used for recycled dgram.

    seq_mask : [B, L], optional
    msa_mask : [B, N_msa, L], optional

    Returns
    -------
    m : [B, N_msa, L, c_m]
    z : [B, L, L, c_z]

    Notes
    -----
    Canonically, AlphaFold recycles:
      - m_prev first row,
      - z_prev,
      - previous positions via a binned distance embedding.
    """

    def __init__(
        self,
        c_m: int = 256,
        c_z: int = 128,
        min_bin: float = 3.25,
        max_bin: float = 20.75,
        num_bins: int = 15,
        recycle_single_enabled: bool = True,
        recycle_pair_enabled: bool = True,
        recycle_position_enabled: bool = True):

        super().__init__()

        self.c_m = c_m
        self.c_z = c_z
        self.min_bin = float(min_bin)
        self.max_bin = float(max_bin)
        self.num_bins = int(num_bins)

        self.recycle_single_enabled = bool(recycle_single_enabled)
        self.recycle_pair_enabled = bool(recycle_pair_enabled)
        self.recycle_position_enabled = bool(recycle_position_enabled)

        self.single_norm = nn.LayerNorm(c_m)
        self.pair_norm = nn.LayerNorm(c_z)
        self.pos_embedding = nn.Embedding(self.num_bins, c_z)

    @staticmethod
    def get_target_row_mask(seq_mask=None, msa_mask=None):
        row_mask = seq_mask
        if msa_mask is not None:
            msa_row_mask = msa_mask[:, 0, :]
            row_mask = msa_row_mask if row_mask is None else row_mask * msa_row_mask
        return row_mask

    @staticmethod
    def build_pair_mask(seq_mask=None):
        if seq_mask is None:
            return None
        return seq_mask[:, :, None] * seq_mask[:, None, :]

    @staticmethod
    def backbone_to_pseudo_beta(backbone_coords, seq_tokens=None, gly_token_idx=8):
        if backbone_coords is None:
            return None

        if backbone_coords.shape[-2] < 3:
            ca_index = 1 if backbone_coords.shape[-2] > 1 else 0
            return backbone_coords[:, :, ca_index, :]

        n = backbone_coords[:, :, 0, :]
        ca = backbone_coords[:, :, 1, :]
        c = backbone_coords[:, :, 2, :]

        b = ca - n
        c_vec = c - ca
        a = torch.cross(b, c_vec, dim=-1)
        cb = (
            (-0.58273431 * a)
            + (0.56802827 * b)
            - (0.54067466 * c_vec)
            + ca)

        if seq_tokens is None:
            return cb

        gly_mask = (seq_tokens == gly_token_idx).unsqueeze(-1)
        return torch.where(gly_mask, ca, cb)

    @classmethod
    def extract_prev_positions(cls, seq_tokens, backbone_coords, t, gly_token_idx=8):
        if backbone_coords is not None:
            pseudo_beta = cls.backbone_to_pseudo_beta(
                backbone_coords,
                seq_tokens=seq_tokens,
                gly_token_idx=gly_token_idx,
            )
            if pseudo_beta is not None:
                return pseudo_beta
        return t

    def _apply_single_recycle(self, m, prev_m1, row_mask=None):
        if (not self.recycle_single_enabled) or (prev_m1 is None):
            return m

        update = self.single_norm(prev_m1)
        if row_mask is not None:
            update = update * row_mask.unsqueeze(-1)

        m = m.clone()
        m[:, 0, :, :] = m[:, 0, :, :] + update

        if row_mask is not None:
            m[:, 0, :, :] = m[:, 0, :, :] * row_mask.unsqueeze(-1)

        return m

    def _apply_pair_recycle(self, z, prev_z, pair_mask=None):
        if (not self.recycle_pair_enabled) or (prev_z is None):
            return z

        z = z + self.pair_norm(prev_z)
        if pair_mask is not None:
            z = z * pair_mask.unsqueeze(-1)
        return z

    def _positions_to_dgram_update(self, prev_positions, dtype, pair_mask=None):
        deltas = prev_positions[:, :, None, :] - prev_positions[:, None, :, :]
        sq_dist = deltas.pow(2).sum(dim=-1).float()

        boundaries = torch.linspace(
            self.min_bin,
            self.max_bin,
            self.num_bins - 1,
            device=prev_positions.device,
            dtype=sq_dist.dtype,
        ).pow(2)

        bin_ids = torch.bucketize(sq_dist, boundaries)
        update = self.pos_embedding(bin_ids).to(dtype=dtype)

        if pair_mask is not None:
            update = update * pair_mask.unsqueeze(-1)

        return update

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        prev_m1: torch.Tensor | None = None,
        prev_z: torch.Tensor | None = None,
        prev_positions: torch.Tensor | None = None,
        seq_mask: torch.Tensor | None = None,
        msa_mask: torch.Tensor | None = None,
    ):
        row_mask = self.get_target_row_mask(seq_mask=seq_mask, msa_mask=msa_mask)
        pair_mask = self.build_pair_mask(seq_mask=seq_mask)

        m = self._apply_single_recycle(m, prev_m1=prev_m1, row_mask=row_mask)
        z = self._apply_pair_recycle(z, prev_z=prev_z, pair_mask=pair_mask)

        if self.recycle_position_enabled and (prev_positions is not None):
            z = z + self._positions_to_dgram_update(
                prev_positions,
                dtype=z.dtype,
                pair_mask=pair_mask,
            )

        return m, z
