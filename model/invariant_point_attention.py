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

    This class is configured below with reduced dimensions for small-dataset /
    memory-constrained experiments.

    Canonical AlphaFold2 Structure Module IPA values from the supplement:
    - num_heads = 12
    - c_hidden = 16
    - num_qk_points = 4
    - num_v_points = 8

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
        num_heads=8,       # AF2 canonical: 12
        c_hidden=16,       # AF2 canonical: 16
        num_qk_points=4,   # AF2 canonical: 4
        num_v_points=8,    # AF2 canonical: 8
    ):
        
        super().__init__()
        assert c_s > 0 and c_z > 0
        assert num_heads > 0 and c_hidden > 0
        assert num_qk_points > 0 and num_v_points > 0
        assert c_s > 0 and c_z > 0

        self.c_s = c_s
        self.c_z = c_z
        self.num_heads = num_heads
        self.c_hidden = c_hidden
        self.num_qk_points = num_qk_points
        self.num_v_points = num_v_points

        self.layer_norm_s = nn.LayerNorm(c_s)
        self.layer_norm_z = nn.LayerNorm(c_z)

        # scalar qkv
        self.linear_q = nn.Linear(c_s, num_heads * c_hidden, bias=False)
        self.linear_k = nn.Linear(c_s, num_heads * c_hidden, bias=False)
        self.linear_v = nn.Linear(c_s, num_heads * c_hidden, bias=False)

        # pair bias -> per head
        self.linear_pair_bias = nn.Linear(c_z, num_heads, bias=False)

        # point q/k/v in local frames
        self.linear_q_pts = nn.Linear(c_s, num_heads * num_qk_points * 3, bias=False)
        self.linear_k_pts = nn.Linear(c_s, num_heads * num_qk_points * 3, bias=False)
        self.linear_v_pts = nn.Linear(c_s, num_heads * num_v_points * 3, bias=False)

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
        nn.init.constant_(self.point_weights, math.log(math.expm1(1.0)))

    def forward(self, s, z, R, t, mask=None):
        B, L, _ = s.shape
        H = self.num_heads
        C = self.c_hidden
        Pqk = self.num_qk_points
        Pv = self.num_v_points

        s = self.layer_norm_s(s)
        z = self.layer_norm_z(z)

        # scalar q, k, v
        q = self.linear_q(s).view(B, L, H, C)
        k = self.linear_k(s).view(B, L, H, C)
        v = self.linear_v(s).view(B, L, H, C)   # [B,L,H,C]

        # scalar logits: [B,H,L,L]
        scalar_logits = torch.einsum("bihc,bjhc->bhij", q, k) / math.sqrt(C)


        # pair bias
        pair_bias = self.linear_pair_bias(z).permute(0, 3, 1, 2)   # [B,H,L,L]

        # point q, k, v in local frame
        q_pts_local = self.linear_q_pts(s).view(B, L, H, Pqk, 3)
        k_pts_local = self.linear_k_pts(s).view(B, L, H, Pqk, 3)
        v_pts_local = self.linear_v_pts(s).view(B, L, H, Pv, 3)

        # local -> global
        q_pts_global = apply_transform(R[:, :, None, None, :, :],
                                       t[:, :, None, None, :], q_pts_local)   # [B,L,H,Pqk,3]

        k_pts_global = apply_transform(R[:, :, None, None, :, :],
                                       t[:, :, None, None, :], k_pts_local)   # [B,L,H,Pqk,3]

        v_pts_global = apply_transform(R[:, :, None, None, :, :],
                                       t[:, :, None, None, :], v_pts_local)   # [B,L,H,Pv,3]


        # -------------------------
        # spatial logits
        # -------------------------
        # q_pts_global: [B,i,H,P,3]
        # k_pts_global: [B,j,H,P,3]
        diff = q_pts_global[:, :, None, :, :, :] - k_pts_global[:, None, :, :, :, :]
        # [B, i, j, H, Pqk, 3]

        sq_dist = (diff ** 2).sum(dim=-1).sum(dim=-1)   # [B, i, j, H]
        sq_dist = sq_dist.permute(0, 3, 1, 2)           # [B, H, L, L]

        gamma = F.softplus(self.point_weights).view(1, H, 1, 1)
        w_c = math.sqrt(2.0 / (9.0 * Pqk))
        w_l = math.sqrt(1.0 / 3.0)
        spatial_term = -0.5 * gamma * w_c * sq_dist


        # total logits + mask
        logits = w_l * (scalar_logits + pair_bias + spatial_term) # Los logits de atencion dependen de atencion normal de transformers + 
                                                                  # Informacion PairWise que viene de la z del evofermer + 
                                                                  # La parte invariant de los puntos cercanos (si i j estan cerca)

        if mask is not None:
            pair_mask = mask[:, :, None] * mask[:, None, :]   # [B,L,L]
            logits = logits.masked_fill(pair_mask[:, None, :, :] == 0, -1e9)

        attn = torch.softmax(logits, dim=-1)


        # scalar value aggregation
        scalar_out = torch.einsum("bhij,bjhc->bihc", attn, v)   # [B,L,H,C] # Atencion estandar 
        scalar_out = scalar_out.reshape(B, L, H * C)


        # pair feature aggregation
        pair_out = torch.einsum("bhij,bijc->bihc", attn, z)  # [B,L,H,c_z]
        pair_out = pair_out.reshape(B, L, H * self.c_z)


        # point value aggregation in global frame
        # -------------------------
        # v_pts_global: [B,j,H,Pv,3]
        point_out_global = torch.einsum("bhij,bjhpc->bihpc", attn, v_pts_global) # Cada residuo j atiende varios puntos 3d cercanos
        # [B,L,H,Pv,3]

        # global -> local frame of residue i
        point_out_local = invert_apply_transform(
            R[:, :, None, None, :, :], t[:, :, None, None, :],point_out_global)   # [B,L,H,Pv,3]

        point_out = point_out_local.reshape(B, L, H * Pv * 3)
        point_norms = torch.sqrt((point_out_local ** 2).sum(dim=-1) + 1e-8)
        point_norms = point_norms.reshape(B, L, H * Pv)

        # final single update
        s_update = torch.cat([scalar_out, point_out, point_norms, pair_out], dim=-1)
        s_update = self.output_linear(s_update)

        if mask is not None:
            s_update = s_update * mask.unsqueeze(-1)

        return s_update, attn
