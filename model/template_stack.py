"""Template stack modules used to inject template angles and pair features.

This file contains the AlphaFold2-style template pair stack, template
pointwise attention, and helper utilities for concatenating template angle
embeddings into the MSA stream before the Evoformer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.triange_attention import * 
from model.triangle_multiplication import *
from model.msa_transitions import *


def normalize_template_mask(
    template_mask,
    *,
    batch_size=None,
    num_templates=None,
    length=None,
    device=None,
    dtype=None,
):
    if template_mask is None:
        if batch_size is None or num_templates is None or length is None:
            return None
        if dtype is None:
            dtype = torch.float32
        return torch.ones(batch_size, num_templates, length, device=device, dtype=dtype)

    if template_mask.ndim == 2:
        if length is None:
            raise ValueError("length is required when template_mask has shape [B, T].")
        return template_mask[:, :, None].expand(-1, -1, length)

    if template_mask.ndim == 3:
        return template_mask

    raise ValueError("template_mask must have shape [B, T] or [B, T, L].")


class TemplatePairStackBlock(nn.Module):
    def __init__(self, c_t=64, num_heads=4, c_hidden_att=16, c_hidden_mul=64, transition_expansion=2, dropout=0.25):
        super().__init__()
        self.tri_attn_start = TriangleAttentionStartingNode(c_z=c_t, num_heads=num_heads, c_hidden=c_hidden_att)
        self.tri_attn_end = TriangleAttentionEndingNode(c_z=c_t, num_heads=num_heads, c_hidden=c_hidden_att)
        self.tri_mul_out = TriangleMultiplicationOutgoing(c_z=c_t, c_hidden=c_hidden_mul)
        self.tri_mul_in = TriangleMultiplicationIncoming(c_z=c_t, c_hidden=c_hidden_mul)
        self.pair_transition = PairTransition(c_z=c_t, expansion=transition_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, t, pair_mask=None):
        t = t + self.dropout(self.tri_attn_start(t, pair_mask))
        t = t + self.dropout(self.tri_attn_end(t, pair_mask))
        t = t + self.dropout(self.tri_mul_out(t, pair_mask))
        t = t + self.dropout(self.tri_mul_in(t, pair_mask))
        t = t + self.dropout(self.pair_transition(t, pair_mask))
        return t


class TemplatePairStack(nn.Module):
    """
    Canonical AF2-style TemplatePairStack:
      2 blocks + final LayerNorm.
    """

    def __init__(self, c_t=64, num_blocks=2, num_heads=4, c_hidden_att=16, c_hidden_mul=64, transition_expansion=2, dropout=0.25):
        super().__init__()
        self.blocks = nn.ModuleList([
            TemplatePairStackBlock(
                c_t=c_t,
                num_heads=num_heads,
                c_hidden_att=c_hidden_att,
                c_hidden_mul=c_hidden_mul,
                transition_expansion=transition_expansion,
                dropout=dropout,
            )
            for _ in range(num_blocks)
        ])
        self.final_ln = nn.LayerNorm(c_t)

    def forward(self, t, pair_mask=None):
        for block in self.blocks:
            t = block(t, pair_mask=pair_mask)
        return self.final_ln(t)


class TemplatePointwiseAttention(nn.Module):
    """
    AF2 Algorithm 17.

    Inputs
    ------
    t : [B, N_templ, L, L, c_t]
        Processed template pair representations.
    z : [B, L, L, c_z]
        Current pair representation.

    Returns
    -------
    dz : [B, L, L, c_z]
        Template-derived update to z.
    """

    def __init__(self, c_z=128, c_t=64, num_heads=4, c_hidden=16):
        super().__init__()
        self.num_heads = num_heads
        self.c_hidden = c_hidden

        self.z_layer_norm = nn.LayerNorm(c_z)
        self.t_layer_norm = nn.LayerNorm(c_t)
        self.linear_q = nn.Linear(c_z, num_heads * c_hidden, bias=False)
        self.linear_k = nn.Linear(c_t, num_heads * c_hidden, bias=False)
        self.linear_v = nn.Linear(c_t, num_heads * c_hidden, bias=False)
        self.output_linear = nn.Linear(num_heads * c_hidden, c_z)

    def forward(self, t, z, template_mask=None):
        B, T, L, _, _ = t.shape
        H = self.num_heads
        C = self.c_hidden

        z_norm = self.z_layer_norm(z)
        t_norm = self.t_layer_norm(t)

        q = self.linear_q(z_norm).view(B, L, L, H, C)
        k = self.linear_k(t_norm).view(B, T, L, L, H, C)
        v = self.linear_v(t_norm).view(B, T, L, L, H, C)

        logits = torch.einsum("bijhc,btijhc->btijh", q, k) / (C ** 0.5)
        pair_mask = None

        if template_mask is not None:
            template_mask = normalize_template_mask(template_mask, length=L)
            pair_mask = template_mask[:, :, :, None] * template_mask[:, :, None, :]  # [B,T,L,L]
            logits = logits.masked_fill(pair_mask.unsqueeze(-1) == 0, -1e9)

        attn = torch.softmax(logits, dim=1)
        if pair_mask is not None:
            attn = attn * pair_mask.unsqueeze(-1).to(attn.dtype)
            attn = attn / attn.sum(dim=1, keepdim=True).clamp_min(1e-8)

        out = torch.einsum("btijh,btijhc->bijhc", attn, v)
        out = out.reshape(B, L, L, H * C)
        return self.output_linear(out)


class TemplateAngleEmbedder(nn.Module):
    """
    Small MLP used to embed template_angle_feat to c_m channels.
    """

    def __init__(self, angle_dim=51, c_m=256):
        super().__init__()
        self.linear_1 = nn.Linear(angle_dim, c_m)
        self.linear_2 = nn.Linear(c_m, c_m)

    def forward(self, x):
        return self.linear_2(F.relu(self.linear_1(x)))


def augment_msa_mask_with_template_mask(msa_mask, template_mask, *, length=None):
    """
    Helper for canonical template-angle concatenation.

    Inputs
    ------
    msa_mask : [B, N_msa, L] or None
    template_mask : [B, N_templ, L] or None

    Returns
    -------
    new_msa_mask : [B, N_msa + N_templ, L] or None
    """
    if template_mask is None:
        return msa_mask
    if length is None:
        if msa_mask is None:
            raise ValueError("length is required when msa_mask is None and template_mask is provided.")
        length = msa_mask.shape[-1]

    template_mask = normalize_template_mask(template_mask, length=length)
    if msa_mask is None:
        return template_mask
    return torch.cat([msa_mask, template_mask], dim=1)


class TemplateStack(nn.Module):
    """
    AF2-style template module.

    It updates:
      - m by concatenating embedded template_angle_feat rows
      - z by adding TemplatePointwiseAttention over TemplatePairStack outputs

    Inputs
    ------
    m : [B, N_msa, L, c_m]
    z : [B, L, L, c_z]

    template_angle_feat : [B, N_templ, L, angle_dim] or None
    template_pair_feat  : [B, N_templ, L, L, pair_dim] or None
    template_mask       : [B, N_templ, L] or None

    Returns
    -------
    m : [B, N_msa + N_templ, L, c_m] if angle feats are provided, else unchanged
    z : [B, L, L, c_z]
    """

    def __init__(
        self,
        c_m=256,
        c_z=128,
        template_angle_dim=51,
        template_pair_dim=88,
        c_t=64,
        num_blocks=2,
        num_heads=4,
        c_hidden_att=16,
        c_hidden_mul=64,
        transition_expansion=2,
        dropout=0.25,
    ):
        super().__init__()
        self.angle_embedder = TemplateAngleEmbedder(angle_dim=template_angle_dim, c_m=c_m)
        self.template_pair_proj = nn.Linear(template_pair_dim, c_t)
        self.template_pair_stack = TemplatePairStack(
            c_t=c_t,
            num_blocks=num_blocks,
            num_heads=num_heads,
            c_hidden_att=c_hidden_att,
            c_hidden_mul=c_hidden_mul,
            transition_expansion=transition_expansion,
            dropout=dropout,
        )
        self.template_pointwise_attention = TemplatePointwiseAttention(
            c_z=c_z,
            c_t=c_t,
            num_heads=num_heads,
            c_hidden=c_hidden_att,
        )

    def forward(
        self,
        m,
        z,
        template_angle_feat=None,
        template_pair_feat=None,
        template_mask=None,
    ):
        # angle path: concat to MSA rows
        if template_angle_feat is not None:
            B, T_angle, L, _ = template_angle_feat.shape
            angle_mask = normalize_template_mask(
                template_mask,
                batch_size=B,
                num_templates=T_angle,
                length=L,
                device=template_angle_feat.device,
                dtype=template_angle_feat.dtype,
            )
            angle_rows = self.angle_embedder(template_angle_feat)  # [B,T,L,c_m]
            angle_rows = angle_rows * angle_mask.unsqueeze(-1)
            m = torch.cat([m, angle_rows], dim=1)

        # pair path: process each template independently, then attend across templates into z
        if template_pair_feat is not None:
            B, T, L, _, _ = template_pair_feat.shape
            t = self.template_pair_proj(template_pair_feat)  # [B,T,L,L,c_t]

            t_out = []
            template_row_mask = normalize_template_mask(
                template_mask,
                batch_size=B,
                num_templates=T,
                length=L,
                device=template_pair_feat.device,
                dtype=template_pair_feat.dtype,
            )
            pair_mask = template_row_mask[:, :, :, None] * template_row_mask[:, :, None, :]  # [B,T,L,L]

            for ti in range(T):
                t_i = t[:, ti, :, :, :]
                pm_i = pair_mask[:, ti, :, :]
                t_i = self.template_pair_stack(t_i, pair_mask=pm_i)
                t_out.append(t_i)

            t = torch.stack(t_out, dim=1)  # [B,T,L,L,c_t]
            z = z + self.template_pointwise_attention(t, z, template_mask=template_row_mask)

        return m, z
