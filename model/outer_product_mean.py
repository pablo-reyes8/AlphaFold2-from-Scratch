"""Outer Product Mean update from MSA to pair representation.

This file converts aggregated MSA statistics into pairwise features using an
outer-product construction followed by a projection. It is the main bridge
from the alignment stream into the pair stream inside the Evoformer.
"""

import torch
import torch.nn as nn

class OuterProductMean(nn.Module):
    """
    AlphaFold2-style Outer Product Mean.

    Input:
        m:        [B, N_msa, L, c_m]
        msa_mask: [B, N_msa, L]   (optional)

    Output:
        z_update: [B, L, L, c_z]
    """

    def __init__(self, c_m=256, c_hidden=32, c_z=128):
        super().__init__()

        self.layer_norm = nn.LayerNorm(c_m)

        self.linear_a = nn.Linear(c_m, c_hidden)
        self.linear_b = nn.Linear(c_m, c_hidden)

        self.output_linear = nn.Linear(c_hidden * c_hidden, c_z)

    def forward(self, m, msa_mask=None):
        """
        m: [B, N, L, c_m]
        msa_mask: [B, N, L]
        """
        B, N, L, _ = m.shape

        m = self.layer_norm(m)

        a = self.linear_a(m)   # [B, N, L, c_hidden]
        b = self.linear_b(m)   # [B, N, L, c_hidden]

        if msa_mask is not None:
            mask = msa_mask.unsqueeze(-1)   # [B, N, L, 1]
            a = a * mask
            b = b * mask

        # Outer product mean over MSA dimension N
        # result: [B, L, L, c_hidden, c_hidden]
        outer = torch.einsum('bnic,bnjd->bijcd', a, b)

        if msa_mask is not None:
            # solo promedia secuencias válidas para el par (i,j)
            pair_mask = torch.einsum('bni,bnj->bij', msa_mask, msa_mask)  # [B, L, L]
            outer = outer / (pair_mask[..., None, None] + 1e-8)
        else:
            outer = outer / N

        outer = outer.reshape(B, L, L, -1)
        z_update = self.output_linear(outer)

        return z_update
