"""Extra MSA stack and global column attention used before the Evoformer.

This module implements the AlphaFold2-style extra MSA path that refines the
pair representation ``z`` before the main Evoformer stack. It also contains
the global column attention variant used inside those extra MSA blocks.
"""

import torch
import torch.nn as nn
from model.custom_dropout import DropoutColumnwise, DropoutRowwise
from model.msa_row_attention import * 
from model.triange_attention import * 
from model.triangle_multiplication import * 
from model.outer_product_mean import *
from model.msa_transitions import *

class MSAColumnGlobalAttention(nn.Module):
    """
    AF2 Algorithm 19.

    Global query over sequences at each residue position.
    """

    def __init__(self, c_m=64, num_heads=8, c_hidden=8):
        super().__init__()
        assert c_m == num_heads * c_hidden, "Require c_m = num_heads * c_hidden"

        self.num_heads = num_heads
        self.c_hidden = c_hidden

        self.layer_norm = nn.LayerNorm(c_m)
        self.linear_q = nn.Linear(c_m, num_heads * c_hidden, bias=False)
        self.linear_k = nn.Linear(c_m, num_heads * c_hidden, bias=False)
        self.linear_v = nn.Linear(c_m, num_heads * c_hidden, bias=False)
        self.linear_g = nn.Linear(c_m, num_heads * c_hidden, bias=True)
        self.output_linear = nn.Linear(num_heads * c_hidden, c_m)
        init_gate_linear(self.linear_g)
        zero_init_linear(self.output_linear)

    def forward(self, m, msa_mask=None):
        B, S, L, _ = m.shape
        H = self.num_heads
        C = self.c_hidden

        x = self.layer_norm(m)
        k = self.linear_k(x).view(B, S, L, H, C)
        v = self.linear_v(x).view(B, S, L, H, C)
        g = torch.sigmoid(self.linear_g(x)).view(B, S, L, H, C)

        if msa_mask is not None:
            denom = msa_mask.sum(dim=1).clamp(min=1.0)  # [B,L]
            q_input = (
                x * msa_mask.unsqueeze(-1)
            ).sum(dim=1) / denom.unsqueeze(-1)
        else:
            q_input = x.mean(dim=1)  # [B,L,c_m]

        q = self.linear_q(q_input).view(B, L, H, C)
        logits = torch.einsum("blhc,bslhc->blhs", q, k) / (C ** 0.5)  # [B,L,H,S]

        if msa_mask is not None:
            logits = logits.masked_fill(msa_mask.permute(0, 2, 1).unsqueeze(2) == 0, -1e9)

        attn = torch.softmax(logits, dim=-1)  # [B,L,H,S]
        out_global = torch.einsum("blhs,bslhc->blhc", attn, v)  # [B,L,H,C]
        out = out_global[:, None, :, :, :].expand(B, S, L, H, C)
        out = g * out
        out = out.reshape(B, S, L, H * C)
        out = self.output_linear(out)

        if msa_mask is not None:
            out = out * msa_mask.unsqueeze(-1)

        return out



class ExtraMsaBlock(nn.Module):
    """
    AF2 Algorithm 18 block.
    """

    def __init__(
        self,
        c_e=64,
        c_z=128,
        c_hidden_opm=32,
        c_hidden_tri_mul=128,
        num_heads_msa=8,
        num_heads_pair=4,
        c_hidden_msa_att=8,
        c_hidden_pair_att=32,
        transition_expansion=4,
        dropout_msa=0.15,
        dropout_pair=0.25):

        super().__init__()
        self.msa_row_attn = MSARowAttentionWithPairBias(
            c_m=c_e,
            c_z=c_z,
            num_heads=num_heads_msa,
            c_hidden=c_hidden_msa_att,
        )
        self.msa_col_global_attn = MSAColumnGlobalAttention(
            c_m=c_e,
            num_heads=num_heads_msa,
            c_hidden=c_hidden_msa_att,
        )
        self.msa_transition = MSATransition(
            c_m=c_e,
            expansion=transition_expansion,
        )
        self.outer_product_mean = OuterProductMean(
            c_m=c_e,
            c_hidden=c_hidden_opm,
            c_z=c_z,
        )
        self.tri_mul_out = TriangleMultiplicationOutgoing(c_z=c_z, c_hidden=c_hidden_tri_mul)
        self.tri_mul_in = TriangleMultiplicationIncoming(c_z=c_z, c_hidden=c_hidden_tri_mul)
        self.tri_attn_start = TriangleAttentionStartingNode(c_z=c_z, num_heads=num_heads_pair, c_hidden=c_hidden_pair_att)
        self.tri_attn_end = TriangleAttentionEndingNode(c_z=c_z, num_heads=num_heads_pair, c_hidden=c_hidden_pair_att)
        self.pair_transition = PairTransition(c_z=c_z, expansion=transition_expansion)

        self.msa_row_dropout = DropoutRowwise(dropout_msa)
        self.pair_row_dropout = DropoutRowwise(dropout_pair)
        self.pair_col_dropout = DropoutColumnwise(dropout_pair)
        self._zero_init_residual_projections()

    def _zero_init_residual_projections(self):
        zero_init_linear(self.msa_row_attn.output_linear)
        zero_init_linear(self.msa_col_global_attn.output_linear)
        zero_init_linear(self.outer_product_mean.output_linear)
        zero_init_linear(self.tri_mul_out.output_linear)
        zero_init_linear(self.tri_mul_in.output_linear)
        zero_init_linear(self.tri_attn_start.output_linear)
        zero_init_linear(self.tri_attn_end.output_linear)

    def forward(self, e, z, extra_msa_mask=None, pair_mask=None):
        e = e + self.msa_row_dropout(self.msa_row_attn(e, z, extra_msa_mask))
        e = e + self.msa_col_global_attn(e, extra_msa_mask)
        e = e + self.msa_transition(e, extra_msa_mask)

        z = z + self.outer_product_mean(e, extra_msa_mask)
        z = z + self.pair_row_dropout(self.tri_mul_out(z, pair_mask))
        z = z + self.pair_row_dropout(self.tri_mul_in(z, pair_mask))
        z = z + self.pair_row_dropout(self.tri_attn_start(z, pair_mask))
        z = z + self.pair_col_dropout(self.tri_attn_end(z, pair_mask))
        z = z + self.pair_transition(z, pair_mask)
        return e, z


class ExtraMsaStack(nn.Module):
    """
    AF2-style Extra MSA stack.

    Inputs
    ------
    m : [B, N_msa, L, c_m]
        Passed through unchanged, returned only for interface consistency.
    z : [B, L, L, c_z]
    extra_msa_feat : [B, N_extra, L, extra_dim] or None
    seq_mask : [B, L], optional
    extra_msa_mask : [B, N_extra, L], optional

    Returns
    -------
    m : unchanged
    z : updated pair representation
    """

    def __init__(
        self,
        c_m=256,
        c_z=128,
        extra_dim=25,
        c_e=64,
        num_blocks=4,
        c_hidden_opm=32,
        c_hidden_tri_mul=128,
        num_heads_msa=8,
        num_heads_pair=4,
        c_hidden_msa_att=8,
        c_hidden_pair_att=32,
        transition_expansion=4,
        dropout_msa=0.15,
        dropout_pair=0.25,
    ):
        super().__init__()
        self.extra_proj = nn.Linear(extra_dim, c_e)

        self.blocks = nn.ModuleList([
            ExtraMsaBlock(
                c_e=c_e,
                c_z=c_z,
                c_hidden_opm=c_hidden_opm,
                c_hidden_tri_mul=c_hidden_tri_mul,
                num_heads_msa=num_heads_msa,
                num_heads_pair=num_heads_pair,
                c_hidden_msa_att=c_hidden_msa_att,
                c_hidden_pair_att=c_hidden_pair_att,
                transition_expansion=transition_expansion,
                dropout_msa=dropout_msa,
                dropout_pair=dropout_pair,
            )
            for _ in range(num_blocks)
        ])

    @staticmethod
    def build_pair_mask(seq_mask=None):
        if seq_mask is None:
            return None
        return seq_mask[:, :, None] * seq_mask[:, None, :]

    def forward(
        self,
        m,
        z,
        extra_msa_feat=None,
        seq_mask=None,
        extra_msa_mask=None,
    ):
        if extra_msa_feat is None:
            return m, z

        e = self.extra_proj(extra_msa_feat)
        if extra_msa_mask is not None:
            e = e * extra_msa_mask.unsqueeze(-1)

        pair_mask = self.build_pair_mask(seq_mask=seq_mask)

        for block in self.blocks:
            e, z = block(e, z, extra_msa_mask=extra_msa_mask, pair_mask=pair_mask)

        return m, z
