"""MSA row attention with pair bias for the Evoformer trunk.

This file implements the row-wise attention update that uses the current pair
representation as an additive bias. It is one of the main information pathways
from pair features back into the MSA stream.
"""

import math
import torch
import torch.nn as nn


class MSARowAttentionWithPairBias(nn.Module):
    """
    AlphaFold2-style MSA Row Attention with Pair Bias.

    Inputs:
        m:        [B, N, L, c_m]
        z:        [B, L, L, c_z]
        msa_mask: [B, N, L]   (optional)

    Output:
        m_update: [B, N, L, c_m]
    """

    def __init__(self, c_m=256, c_z=128, num_heads=8, c_hidden=32):
        super().__init__()
        assert c_m == num_heads * c_hidden, "Require c_m = num_heads * c_hidden"

        self.c_m = c_m
        self.c_z = c_z
        self.num_heads = num_heads
        self.c_hidden = c_hidden

        self.m_layer_norm = nn.LayerNorm(c_m)
        self.z_layer_norm = nn.LayerNorm(c_z)

        self.linear_q = nn.Linear(c_m, num_heads * c_hidden, bias=False)
        self.linear_k = nn.Linear(c_m, num_heads * c_hidden, bias=False)
        self.linear_v = nn.Linear(c_m, num_heads * c_hidden, bias=False)

        # pair bias added to logits before softmax
        self.linear_bias = nn.Linear(c_z, num_heads, bias=False)

        self.linear_gate = nn.Linear(c_m, num_heads * c_hidden)
        self.output_linear = nn.Linear(num_heads * c_hidden, c_m)

        self.sigmoid = nn.Sigmoid()

    def forward(self, m, z, msa_mask=None):
        """
        m: [B, N, L, c_m]
        z: [B, L, L, c_z]
        msa_mask: [B, N, L]
        """
        B, N, L, _ = m.shape

        m_norm = self.m_layer_norm(m)
        z_norm = self.z_layer_norm(z)

        q = self.linear_q(m_norm).view(B, N, L, self.num_heads, self.c_hidden)
        k = self.linear_k(m_norm).view(B, N, L, self.num_heads, self.c_hidden)
        v = self.linear_v(m_norm).view(B, N, L, self.num_heads, self.c_hidden)

        # [B, L, L, H]
        pair_bias = self.linear_bias(z_norm)

        # attention over residue positions j for each row n
        # logits: [B, N, H, I, J]
        logits = torch.einsum("bnihc,bnjhc->bnhij", q, k) / math.sqrt(self.c_hidden)

        # add pair bias before softmax
        logits = logits + pair_bias.permute(0, 3, 1, 2).unsqueeze(1)  # [B,1,H,I,J]

        if msa_mask is not None:
            key_mask = msa_mask.unsqueeze(2).unsqueeze(3)  # [B,N,1,1,L]
            logits = logits.masked_fill(key_mask == 0, -1e9)

        attn = torch.softmax(logits, dim=-1)

        out = torch.einsum("bnhij,bnjhc->bnihc", attn, v)

        gate = self.sigmoid(self.linear_gate(m_norm)).view(B, N, L, self.num_heads, self.c_hidden)
        out = out * gate

        out = out.reshape(B, N, L, self.num_heads * self.c_hidden)
        out = self.output_linear(out)

        if msa_mask is not None:
            out = out * msa_mask.unsqueeze(-1)

        return out

