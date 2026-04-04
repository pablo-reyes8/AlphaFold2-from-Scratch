"""MSA column attention used inside the Evoformer.

This module mixes information across alignment rows at each residue position.
It complements row attention with pair bias and helps propagate context through
the MSA dimension of the model.
"""

import torch
import torch.nn as nn
import math

from model.msa_transitions import init_gate_linear

class MSAColumnAttention(nn.Module):
    """
    AlphaFold2-style MSA Column Attention.

    Inputs:
        m:        [B, N, L, c_m]
        msa_mask: [B, N, L]   (optional)

    Output:
        m_update: [B, N, L, c_m]
    """

    def __init__(self, c_m=256, num_heads=8, c_hidden=32):
        super().__init__()
        assert c_m == num_heads * c_hidden, "Require c_m = num_heads * c_hidden"

        self.c_m = c_m
        self.num_heads = num_heads
        self.c_hidden = c_hidden

        self.layer_norm = nn.LayerNorm(c_m)

        self.linear_q = nn.Linear(c_m, num_heads * c_hidden, bias=False)
        self.linear_k = nn.Linear(c_m, num_heads * c_hidden, bias=False)
        self.linear_v = nn.Linear(c_m, num_heads * c_hidden, bias=False)

        self.linear_gate = nn.Linear(c_m, num_heads * c_hidden)
        self.output_linear = nn.Linear(num_heads * c_hidden, c_m)

        self.sigmoid = nn.Sigmoid()
        init_gate_linear(self.linear_gate)

    def forward(self, m, msa_mask=None):
        """
        m: [B, N, L, c_m]
        msa_mask: [B, N, L]
        """
        B, N, L, _ = m.shape

        m_norm = self.layer_norm(m)

        q = self.linear_q(m_norm).view(B, N, L, self.num_heads, self.c_hidden)
        k = self.linear_k(m_norm).view(B, N, L, self.num_heads, self.c_hidden)
        v = self.linear_v(m_norm).view(B, N, L, self.num_heads, self.c_hidden)

        # attend over MSA sequences n for each residue position l
        # logits: [B, L, H, N_q, N_k]
        logits = torch.einsum("bnlhc,bmlhc->blhnm", q, k) / math.sqrt(self.c_hidden)

        if msa_mask is not None:
            key_mask = msa_mask.permute(0, 2, 1).unsqueeze(2).unsqueeze(3)  # [B,L,1,1,N]
            logits = logits.masked_fill(key_mask == 0, -1e9)

        attn = torch.softmax(logits, dim=-1)

        v_t = v.permute(0, 2, 1, 3, 4)  # [B, L, N, H, C]
        out = torch.einsum("blhnm,blmhc->bnlhc", attn, v_t)

        gate = self.sigmoid(self.linear_gate(m_norm)).view(B, N, L, self.num_heads, self.c_hidden)
        out = out * gate

        out = out.reshape(B, N, L, self.num_heads * self.c_hidden)
        out = self.output_linear(out)

        if msa_mask is not None:
            out = out * msa_mask.unsqueeze(-1)

        return out
