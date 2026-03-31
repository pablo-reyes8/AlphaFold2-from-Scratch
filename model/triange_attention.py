"""Triangle attention updates for pair representations.

This module implements the starting-node and ending-node triangle attention
operations that refine pair features using third-residue context. They are core
geometric mixing operators inside the Evoformer pair stack.
"""

import math
import torch
import torch.nn as nn


class TriangleAttentionStartingNode(nn.Module):
    """
    AlphaFold2-style Triangle Attention (Starting Node).

    Input:
        z:         [B, L, L, c_z]
        pair_mask: [B, L, L]   (optional)

    Output:
        z_update:  [B, L, L, c_z]

    Idea:
        For each fixed starting node i, attention is performed over k on z[i, k]
        to update z[i, j]. Equivalently, for each row i we do attention across
        the second residue index.
    """

    def __init__(self, c_z=128, num_heads=4, c_hidden=32):
        super().__init__()
        assert c_z == num_heads * c_hidden, "Require c_z = num_heads * c_hidden"

        self.c_z = c_z
        self.num_heads = num_heads
        self.c_hidden = c_hidden

        self.input_layer_norm = nn.LayerNorm(c_z)

        self.linear_q = nn.Linear(c_z, num_heads * c_hidden, bias=False)
        self.linear_k = nn.Linear(c_z, num_heads * c_hidden, bias=False)
        self.linear_v = nn.Linear(c_z, num_heads * c_hidden, bias=False)

        # pair bias per head
        self.linear_bias = nn.Linear(c_z, num_heads, bias=False)

        # gating
        self.linear_gate = nn.Linear(c_z, num_heads * c_hidden)

        self.output_linear = nn.Linear(num_heads * c_hidden, c_z)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z, pair_mask=None):
        """
        z: [B, L, L, c_z]
        pair_mask: [B, L, L]
        """
        B, L, _, _ = z.shape
        z_norm = self.input_layer_norm(z)

        q = self.linear_q(z_norm).view(B, L, L, self.num_heads, self.c_hidden)
        k = self.linear_k(z_norm).view(B, L, L, self.num_heads, self.c_hidden)
        v = self.linear_v(z_norm).view(B, L, L, self.num_heads, self.c_hidden)

        bias = self.linear_bias(z_norm)

        # attention logits over k for each fixed (i, j)
        # q from z[i,j], keys from z[i,k]
        # result: [B, i, h, j, k]
        logits = torch.einsum("bijhc,bikhc->bihjk", q, k) / math.sqrt(self.c_hidden)

        # add pair bias from candidate key edge (i,k)
        key_bias = bias.permute(0, 1, 3, 2).unsqueeze(3)
        logits = logits + key_bias

        if pair_mask is not None:
            # key mask on (i,k)
            key_mask = pair_mask.unsqueeze(2).unsqueeze(3)
            logits = logits.masked_fill(key_mask == 0, -1e9)

        attn = torch.softmax(logits, dim=-1)  # over k

        # weighted sum of values z[i,k]
        # [B, i, j, h, c]
        out = torch.einsum("bihjk,bikhc->bijhc", attn, v)

        gate = self.sigmoid(self.linear_gate(z_norm)).view(B, L, L, self.num_heads, self.c_hidden)
        out = out * gate

        out = out.reshape(B, L, L, self.num_heads * self.c_hidden)
        out = self.output_linear(out)

        if pair_mask is not None:
            out = out * pair_mask.unsqueeze(-1)

        return out


class TriangleAttentionEndingNode(nn.Module):
    """
    AlphaFold2-style Triangle Attention (Ending Node).

    Input:
        z:         [B, L, L, c_z]
        pair_mask: [B, L, L]   (optional)

    Output:
        z_update:  [B, L, L, c_z]

    Idea:
        For each fixed ending node j, attention is performed over k on z[k, j]
        to update z[i, j].
    """

    def __init__(self, c_z=128, num_heads=4, c_hidden=32):
        super().__init__()
        assert c_z == num_heads * c_hidden, "Require c_z = num_heads * c_hidden"

        self.c_z = c_z
        self.num_heads = num_heads
        self.c_hidden = c_hidden

        self.input_layer_norm = nn.LayerNorm(c_z)

        self.linear_q = nn.Linear(c_z, num_heads * c_hidden, bias=False)
        self.linear_k = nn.Linear(c_z, num_heads * c_hidden, bias=False)
        self.linear_v = nn.Linear(c_z, num_heads * c_hidden, bias=False)

        self.linear_bias = nn.Linear(c_z, num_heads, bias=False)
        self.linear_gate = nn.Linear(c_z, num_heads * c_hidden)

        self.output_linear = nn.Linear(num_heads * c_hidden, c_z)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z, pair_mask=None):
        """
        z: [B, L, L, c_z]
        pair_mask: [B, L, L]
        """
        B, L, _, _ = z.shape
        z_norm = self.input_layer_norm(z)

        q = self.linear_q(z_norm).view(B, L, L, self.num_heads, self.c_hidden)
        k = self.linear_k(z_norm).view(B, L, L, self.num_heads, self.c_hidden)
        v = self.linear_v(z_norm).view(B, L, L, self.num_heads, self.c_hidden)

        bias = self.linear_bias(z_norm)  # [B,i,j,h]

        # For each fixed j, query z[i,j] attends over keys z[k,j]
        # logits: [B, j, h, i, k]
        logits = torch.einsum("bijhc,bkjhc->bhijk", q, k) / math.sqrt(self.c_hidden)
        logits = logits.permute(0, 3, 1, 2, 4)  # [B, j, h, i, k]

        # bias from candidate key edge (k,j)
        key_bias = bias.permute(0, 2, 3, 1).unsqueeze(3)  # [B, j, h, 1, k]
        logits = logits + key_bias

        if pair_mask is not None:
            # mask over key edges (k,j)
            key_mask = pair_mask.transpose(1, 2).unsqueeze(2).unsqueeze(3)  # [B, j, 1, 1, k]
            logits = logits.masked_fill(key_mask == 0, -1e9)

        attn = torch.softmax(logits, dim=-1)  # over k

        # values are z[k,j]
        v_t = v.permute(0, 2, 1, 3, 4)  # [B, j, k, h, c]
        out = torch.einsum("bjhik,bjkhc->bijhc", attn, v_t)

        gate = self.sigmoid(self.linear_gate(z_norm)).view(B, L, L, self.num_heads, self.c_hidden)
        out = out * gate

        out = out.reshape(B, L, L, self.num_heads * self.c_hidden)
        out = self.output_linear(out)

        if pair_mask is not None:
            out = out * pair_mask.unsqueeze(-1)

        return out
