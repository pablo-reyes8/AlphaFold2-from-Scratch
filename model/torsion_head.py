"""Torsion prediction head operating on structure representations.

This module predicts normalized torsion-angle vectors from the initial and
final single representations. It provides the angle supervision branch used by
the main model and by the AlphaFold-style loss.
"""

import torch
import torch.nn as nn

class TorsionResBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.zeros_(self.lin1.bias)
        nn.init.xavier_uniform_(self.lin2.weight)
        nn.init.zeros_(self.lin2.bias)

    def forward(self, x):
        y = self.ln(x)
        y = self.act(self.lin1(y))
        y = self.dropout(y)
        y = self.lin2(y)
        return x + y


class TorsionHead(nn.Module):
    """
    Improved AF2-like torsion head.

    Inputs:
        s_initial: [B, L, c_s]   # single repr before StructureModule
        s_final:   [B, L, c_s]   # single repr after StructureModule
        mask:      [B, L]        (optional)

    Output:
        torsions: [B, L, n_torsions, 2]
                  normalized 2D vectors, interpretable as sin/cos-like pairs
    """
    def __init__(
        self,
        c_s=256,
        hidden=256,
        n_torsions=7,
        num_res_blocks=2,
        dropout=0.1):

        super().__init__()
        self.n_torsions = n_torsions

        self.ln_initial = nn.LayerNorm(c_s)
        self.ln_final = nn.LayerNorm(c_s)

        self.input_proj = nn.Linear(2 * c_s, hidden)

        self.resblocks = nn.ModuleList([
            TorsionResBlock(hidden, dropout=dropout)
            for _ in range(num_res_blocks)])

        self.output_ln = nn.LayerNorm(hidden)
        self.output = nn.Linear(hidden, 2 * n_torsions)

        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

        # Important for stable initialization
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, s_initial, s_final, mask=None):
        x = torch.cat(
            [self.ln_initial(s_initial), self.ln_final(s_final)],
            dim=-1)  # [B, L, 2*c_s]

        x = self.input_proj(x)

        for block in self.resblocks:
            x = block(x)

        x = self.output(self.output_ln(x))  # [B, L, 2*n_torsions]
        x = x.view(*x.shape[:-1], self.n_torsions, 2)

        x = x / torch.linalg.norm(x, dim=-1, keepdim=True).clamp_min(1e-8)

        if mask is not None:
            x = x * mask.unsqueeze(-1).unsqueeze(-1)

        return x
