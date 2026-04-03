"""Stacked Evoformer trunk for iterative representation refinement.

This module repeats the Evoformer block multiple times and provides the higher
level interface used by the main model. It is the central trunk that updates
MSA features and pair features before structure prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.evoformer_block import *

class EvoformerStack(nn.Module):
    def __init__(
        self,
        num_blocks=4,
        c_m=256,
        c_z=128,
        c_hidden_opm=32,
        c_hidden_tri_mul=128,
        num_heads_msa=8,
        num_heads_pair=4,
        c_hidden_msa_att=32,
        c_hidden_pair_att=32,
        transition_expansion=4,
        dropout=0.15,
        pair_stack_enabled=True,
        triangle_multiplication_enabled=True,
        triangle_attention_enabled=True,
        pair_transition_enabled=True):

        super().__init__()
        self.pair_stack_enabled = bool(pair_stack_enabled)
        self.triangle_multiplication_enabled = bool(triangle_multiplication_enabled)
        self.triangle_attention_enabled = bool(triangle_attention_enabled)
        self.pair_transition_enabled = bool(pair_transition_enabled)
        self.blocks = nn.ModuleList([
            EvoformerBlock(
                c_m=c_m,
                c_z=c_z,
                c_hidden_opm=c_hidden_opm,
                c_hidden_tri_mul=c_hidden_tri_mul,
                num_heads_msa=num_heads_msa,
                num_heads_pair=num_heads_pair,
                c_hidden_msa_att=c_hidden_msa_att,
                c_hidden_pair_att=c_hidden_pair_att,
                transition_expansion=transition_expansion,
                dropout=dropout,
                pair_stack_enabled=self.pair_stack_enabled,
                triangle_multiplication_enabled=self.triangle_multiplication_enabled,
                triangle_attention_enabled=self.triangle_attention_enabled,
                pair_transition_enabled=self.pair_transition_enabled,)  for _ in range(num_blocks)])

    def forward(self, m, z, msa_mask=None, pair_mask=None):
        for block in self.blocks:
            m, z = block(m, z, msa_mask, pair_mask)
        return m, z
