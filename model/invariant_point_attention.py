"""Invariant Point Attention implementation for structure updates.

This file contains the IPA block that mixes single and pair representations in
a geometry-aware way using frames and point projections. The structure module
uses it as its main attention mechanism.
"""

import math
import torch.nn.functional as F
import torch
import torch.nn as nn
from model.ipa_transformations import *


class InvariantPointAttention(nn.Module):
    """
    AlphaFold2 Invariant Point Attention (IPA).

    Notes
    -----
    This implementation keeps the original IPA formulation:
    - scalar attention term
    - pair bias term
    - invariant point-distance term with canonical w_L and w_C scaling
    - pair aggregation as sum_j a_ij * z_ij
    - point-value aggregation in global frame followed by projection back to
      the local frame of residue i

    Memory optimization
    -------------------
    The spatial point-distance logits are computed without explicitly
    materializing:

        diff = q_pts_global[:, :, None, :, :, :] - k_pts_global[:, None, :, :, :, :]

    whose shape would be:

        [B, L, L, H, Pqk, 3]

    Instead, we use:

        ||q_i - k_j||^2 = ||q_i||^2 + ||k_j||^2 - 2 q_i^T k_j

    and optionally compute the key dimension j in chunks.

    Inputs
    ------
    s : [B, L, c_s]
        Single representation.
    z : [B, L, L, c_z]
        Pair representation.
    R : [B, L, 3, 3]
        Rotation matrices of current residue frames.
    t : [B, L, 3]
        Translation vectors of current residue frames.
    mask : [B, L], optional
        Residue mask.

    Returns
    -------
    s_update : [B, L, c_s]
        Single-representation update produced by IPA.
    attn : [B, H, L, L]
        Attention weights over residues.
    """

    def __init__(
        self,
        c_s=256,
        c_z=128,
        num_heads=8,          # AF2 canonical: 12
        c_hidden=16,          # AF2 canonical: 16
        num_qk_points=4,      # AF2 canonical: 4
        num_v_points=8,       # AF2 canonical: 8
        point_logits_chunk_size=None,  # e.g. None, 64, 128
    ):
        super().__init__()

        assert c_s > 0 and c_z > 0
        assert num_heads > 0 and c_hidden > 0
        assert num_qk_points > 0 and num_v_points > 0
        assert point_logits_chunk_size is None or point_logits_chunk_size > 0

        self.c_s = c_s
        self.c_z = c_z
        self.num_heads = num_heads
        self.c_hidden = c_hidden
        self.num_qk_points = num_qk_points
        self.num_v_points = num_v_points
        self.point_logits_chunk_size = point_logits_chunk_size

        # scalar qkv
        self.linear_q = nn.Linear(c_s, num_heads * c_hidden, bias=False)
        self.linear_k = nn.Linear(c_s, num_heads * c_hidden, bias=False)
        self.linear_v = nn.Linear(c_s, num_heads * c_hidden, bias=False)

        # pair bias -> per head
        self.linear_pair_bias = nn.Linear(c_z, num_heads, bias=False)

        # point q/k/v in local frames
        self.linear_q_pts = nn.Linear(
            c_s,
            num_heads * num_qk_points * 3,bias=False,)
        
        self.linear_k_pts = nn.Linear(
            c_s,
            num_heads * num_qk_points * 3,bias=False)
        
        self.linear_v_pts = nn.Linear(
            c_s,
            num_heads * num_v_points * 3,bias=False)

        # trainable positive weights for spatial logits, one per head
        self.point_weights = nn.Parameter(torch.zeros(num_heads))

        # final projection back to single representation
        out_dim = (
            num_heads * c_hidden +          # scalar attended values
            num_heads * num_v_points * 3 +  # local point outputs
            num_heads * num_v_points +      # norms of local point outputs
            num_heads * c_z                 # attended pair features
        )

        self.output_linear = nn.Linear(out_dim, c_s)

        # Initialize so softplus(point_weights) starts near 1.0
        nn.init.constant_(self.point_weights, math.log(math.expm1(1.0)))

    def _compute_sq_dist_chunked(
        self,
        q_pts_global,
        k_pts_global,
        chunk_size=None):
        """
        Memory-efficient squared point distances for IPA.

        Computes:
            sq_dist[b, h, i, j] =
                sum_p || q[b, i, h, p] - k[b, j, h, p] ||^2

        without materializing:
            [B, L, L, H, Pqk, 3]

        Parameters
        ----------
        q_pts_global : torch.Tensor
            [B, L, H, Pqk, 3]
        k_pts_global : torch.Tensor
            [B, L, H, Pqk, 3]
        chunk_size : int or None
            Chunk size over the key/residue dimension j.

        Returns
        -------
        sq_dist : torch.Tensor
            [B, H, L, L]
        """
        B, L, H, Pqk, three = q_pts_global.shape
        assert three == 3
        assert k_pts_global.shape == (B, L, H, Pqk, 3)

        # ||q_i||^2 summed over q/k points and xyz.
        # Shape: [B, L, H]
        q_sq = (q_pts_global ** 2).sum(dim=(-1, -2))

        # Non-chunked path:
        # Still avoids the full diff tensor [B,L,L,H,P,3].
        if chunk_size is None or chunk_size >= L:
            # Shape: [B, L, H]
            k_sq = (k_pts_global ** 2).sum(dim=(-1, -2))

            # Dot product over Pqk and xyz.
            # Shape: [B, L, L, H]
            qk_dot = torch.einsum("bihpc,bjhpc->bijh", q_pts_global, k_pts_global,)

            # Shape: [B, L, L, H]
            sq_dist = q_sq[:, :, None, :] + k_sq[:, None, :, :] - 2.0 * qk_dot

            # Numerical safety: due to fp16/bf16/fp32 roundoff,
            # tiny negative values can appear.
            sq_dist = sq_dist.clamp_min(0.0)

            # Shape: [B, H, L, L]
            return sq_dist.permute(0, 3, 1, 2).contiguous()

        # Chunked path over the key residue dimension j.
        sq_dist_chunks = []

        for j_start in range(0, L, chunk_size):
            j_end = min(j_start + chunk_size, L)

            # Shape: [B, J, H, Pqk, 3]
            k_chunk = k_pts_global[:, j_start:j_end]

            # Shape: [B, J, H]
            k_sq = (k_chunk ** 2).sum(dim=(-1, -2))

            # Shape: [B, L, J, H]
            qk_dot = torch.einsum( "bihpc,bjhpc->bijh",q_pts_global,k_chunk,)

            # Shape: [B, L, J, H]
            sq_dist_chunk = (
                q_sq[:, :, None, :]
                + k_sq[:, None, :, :] - 2.0 * qk_dot)

            sq_dist_chunk = sq_dist_chunk.clamp_min(0.0)

            # Shape: [B, H, L, J]
            sq_dist_chunk = sq_dist_chunk.permute(0, 3, 1, 2).contiguous()

            sq_dist_chunks.append(sq_dist_chunk)

        # Shape: [B, H, L, L]
        sq_dist = torch.cat(sq_dist_chunks, dim=-1)

        return sq_dist

    def forward(self, s, z, R, t, mask=None):
        B, L, _ = s.shape
        H = self.num_heads
        C = self.c_hidden
        Pqk = self.num_qk_points
        Pv = self.num_v_points

        # -------------------------
        # scalar q, k, v
        # -------------------------
        q = self.linear_q(s).view(B, L, H, C)
        k = self.linear_k(s).view(B, L, H, C)
        v = self.linear_v(s).view(B, L, H, C)

        # Scalar logits.
        # Shape: [B, H, L, L]
        scalar_logits = torch.einsum("bihc,bjhc->bhij", q, k) / math.sqrt(C)

        # -------------------------
        # pair bias
        # -------------------------
        # Shape: [B, H, L, L]
        pair_bias = self.linear_pair_bias(z).permute(0, 3, 1, 2)

        # -------------------------
        # point q, k, v in local frame
        # -------------------------
        q_pts_local = self.linear_q_pts(s).view(B, L, H, Pqk, 3)
        k_pts_local = self.linear_k_pts(s).view(B, L, H, Pqk, 3)
        v_pts_local = self.linear_v_pts(s).view(B, L, H, Pv, 3)

        # -------------------------
        # local frame -> global frame
        # -------------------------
        q_pts_global = apply_transform(
            R[:, :, None, None, :, :],
            t[:, :, None, None, :],
            q_pts_local,)  # [B, L, H, Pqk, 3]

        k_pts_global = apply_transform(
            R[:, :, None, None, :, :],
            t[:, :, None, None, :],
            k_pts_local,)  # [B, L, H, Pqk, 3]

        v_pts_global = apply_transform(
            R[:, :, None, None, :, :],
            t[:, :, None, None, :],
            v_pts_local,)  # [B, L, H, Pv, 3]

        # -------------------------
        # spatial logits
        # -------------------------
        # Memory-efficient point distances.
        #
        # Avoids materializing:
        #   diff: [B, L, L, H, Pqk, 3]
        #
        # Computes:
        #   sq_dist[b, h, i, j] =
        #       sum_p ||q_pts_global[b,i,h,p] - k_pts_global[b,j,h,p]||^2
        #
        # Shape:
        #   sq_dist: [B, H, L, L]
        sq_dist = self._compute_sq_dist_chunked(
            q_pts_global=q_pts_global,
            k_pts_global=k_pts_global,
            chunk_size=self.point_logits_chunk_size )

        gamma = F.softplus(self.point_weights).view(1, H, 1, 1)

        # Canonical AF2 IPA scaling factors.
        w_c = math.sqrt(2.0 / (9.0 * Pqk))
        w_l = math.sqrt(1.0 / 3.0)

        spatial_term = -0.5 * gamma * w_c * sq_dist

        # -------------------------
        # total logits + mask
        # -------------------------
        logits = w_l * (scalar_logits + pair_bias + spatial_term)

        if mask is not None:
            pair_mask = mask[:, :, None] * mask[:, None, :]  # [B, L, L]
            logits = logits.masked_fill(pair_mask[:, None, :, :] == 0, -1e9)

        attn = torch.softmax(logits, dim=-1)

        # -------------------------
        # scalar value aggregation
        # -------------------------
        scalar_out = torch.einsum(
            "bhij,bjhc->bihc",
            attn,
            v, )  # [B, L, H, C]

        scalar_out = scalar_out.reshape(B, L, H * C)

        # -------------------------
        # pair feature aggregation
        # -------------------------
        pair_out = torch.einsum(
            "bhij,bijc->bihc",
            attn,
            z, )  # [B, L, H, c_z]

        pair_out = pair_out.reshape(B, L, H * self.c_z)

        # -------------------------
        # point value aggregation in global frame
        # -------------------------
        point_out_global = torch.einsum(
            "bhij,bjhpc->bihpc",
            attn,
            v_pts_global, )  # [B, L, H, Pv, 3]

        # -------------------------
        # global frame -> local frame of residue i
        # -------------------------
        point_out_local = invert_apply_transform(
            R[:, :, None, None, :, :],
            t[:, :, None, None, :],
            point_out_global, )  # [B, L, H, Pv, 3]

        point_out = point_out_local.reshape(B, L, H * Pv * 3)

        point_norms = torch.sqrt(
            (point_out_local ** 2).sum(dim=-1) + 1e-8 )  # [B, L, H, Pv]

        point_norms = point_norms.reshape(B, L, H * Pv)

        # -------------------------
        # final single update
        # -------------------------
        s_update = torch.cat(
            [
                scalar_out,
                point_out,
                point_norms,
                pair_out,
            ],
            dim=-1,
        )

        s_update = self.output_linear(s_update)

        if mask is not None:
            s_update = s_update * mask.unsqueeze(-1)

        return s_update, attn