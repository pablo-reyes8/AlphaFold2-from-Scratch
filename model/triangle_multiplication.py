"""Triangle multiplicative updates for pair representations.

This file implements the outgoing and incoming triangle multiplication blocks
used by the pair stack. These operators capture transitive interactions across
triplets of residues and are standard in AlphaFold-style architectures.
"""

import torch
import torch.nn as nn
import math

class TriangleMultiplicationOutgoing(nn.Module):
    """
    AlphaFold2-style Triangle Multiplicative Update (Outgoing).

    Input:
        z:         [B, L, L, c_z]
        pair_mask: [B, L, L]   (optional)

    Output:
        z_update:  [B, L, L, c_z]
    """

    def __init__(self, c_z=128, c_hidden=128, dropout=0.1, eps=1e-8):
        super().__init__()

        self.c_hidden = c_hidden
        self.eps = eps

        self.input_layer_norm = nn.LayerNorm(c_z)

        self.linear_a_p = nn.Linear(c_z, c_hidden)
        self.linear_b_p = nn.Linear(c_z, c_hidden)

        self.linear_a_g = nn.Linear(c_z, c_hidden)
        self.linear_b_g = nn.Linear(c_z, c_hidden)

        self.output_layer_norm = nn.LayerNorm(c_hidden)
        self.output_linear = nn.Linear(c_hidden, c_z)
        self.output_gate = nn.Linear(c_z, c_z)

        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, z, pair_mask=None):
        """
        z: [B, L, L, c_z]
        pair_mask: [B, L, L]
        """
        z_norm = self.input_layer_norm(z)

        a = self.linear_a_p(z_norm) * self.sigmoid(self.linear_a_g(z_norm))
        b = self.linear_b_p(z_norm) * self.sigmoid(self.linear_b_g(z_norm))

        if pair_mask is not None:
            mask = pair_mask.unsqueeze(-1).to(z.dtype)
            a = a * mask
            b = b * mask

        # outgoing: aggregate through k using (i,k) and (j,k)
        # x[b,i,j,c] = sum_k a[b,i,k,c] * b[b,j,k,c]
        x = torch.einsum("bikc,bjkc->bijc", a, b)

        # normalize by effective number of valid k's
        if pair_mask is not None:
            valid_k = torch.einsum("bik,bjk->bij", pair_mask, pair_mask).to(z.dtype)
            x = x / torch.sqrt(valid_k.unsqueeze(-1).clamp_min(1.0))
        else:
            L = z.shape[1]
            x = x / math.sqrt(L)

        x = self.output_layer_norm(x)
        x = self.output_linear(x)

        gate = self.sigmoid(self.output_gate(z_norm))
        x = x * gate
        x = self.dropout(x)

        if pair_mask is not None:
            x = x * pair_mask.unsqueeze(-1).to(z.dtype)

        return x


class TriangleMultiplicationIncoming(nn.Module):
    """
    AlphaFold2-style Triangle Multiplicative Update (Incoming).

    Input:
        z:         [B, L, L, c_z]
        pair_mask: [B, L, L]   (optional)

    Output:
        z_update:  [B, L, L, c_z]
    """

    def __init__(self, c_z=128, c_hidden=128, dropout=0.1, eps=1e-8):
        super().__init__()

        self.c_hidden = c_hidden
        self.eps = eps

        self.input_layer_norm = nn.LayerNorm(c_z)

        self.linear_a_p = nn.Linear(c_z, c_hidden)
        self.linear_b_p = nn.Linear(c_z, c_hidden)

        self.linear_a_g = nn.Linear(c_z, c_hidden)
        self.linear_b_g = nn.Linear(c_z, c_hidden)

        self.output_layer_norm = nn.LayerNorm(c_hidden)
        self.output_linear = nn.Linear(c_hidden, c_z)
        self.output_gate = nn.Linear(c_z, c_z)

        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, z, pair_mask=None):
        """
        z: [B, L, L, c_z]
        pair_mask: [B, L, L]
        """
        z_norm = self.input_layer_norm(z)

        a = self.linear_a_p(z_norm) * self.sigmoid(self.linear_a_g(z_norm))  # [B,L,L,c_h]
        b = self.linear_b_p(z_norm) * self.sigmoid(self.linear_b_g(z_norm))  # [B,L,L,c_h]

        if pair_mask is not None:
            mask = pair_mask.unsqueeze(-1).to(z.dtype)
            a = a * mask
            b = b * mask

        # incoming: aggregate through k using (k,i) and (k,j)
        # x[b,i,j,c] = sum_k a[b,k,i,c] * b[b,k,j,c]
        x = torch.einsum("bkic,bkjc->bijc", a, b)

        # normalize by effective number of valid k's
        if pair_mask is not None:
            valid_k = torch.einsum("bki,bkj->bij", pair_mask, pair_mask).to(z.dtype)
            x = x / torch.sqrt(valid_k.unsqueeze(-1).clamp_min(1.0))
        else:
            L = z.shape[1]
            x = x / math.sqrt(L)

        x = self.output_layer_norm(x)
        x = self.output_linear(x)

        gate = self.sigmoid(self.output_gate(z_norm))
        x = x * gate
        x = self.dropout(x)

        if pair_mask is not None:
            x = x * pair_mask.unsqueeze(-1).to(z.dtype)

        return x
