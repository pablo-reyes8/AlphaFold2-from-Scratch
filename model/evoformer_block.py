"""Single Evoformer block used inside the stacked trunk.

This file groups the row attention, column attention, transition, outer
product mean, triangle attention, and triangle multiplication updates that
together refine the MSA and pair representations in one block.
"""

import torch
import torch.nn as nn

from model.input_embedder import * 
from model.msa_colum_attention import * 
from model.msa_row_attention import * 
from model.msa_transitions import * 
from model.triange_attention import * 
from model.triangle_multiplication import *
from model.outer_product_mean import *


class EvoformerBlock(nn.Module):
    """
    Canonical AlphaFold2-style Evoformer block.

    Order
    -----
    1) MSA Row Attention with Pair Bias
    -> residual update on m
    Uses the current pair representation z as a bias to refine the MSA
    representation row-wise.

    2) MSA Column Attention
    -> residual update on m
    Mixes information across MSA sequences at each residue position.

    3) MSA Transition
    -> residual update on m
    Position-wise feed-forward transformation on the MSA representation.

    4) Outer Product Mean
    -> residual update on z
    Main bridge from MSA space to pair space.
    Converts information from m into pairwise signals and injects them into z.
    Intuitively, it summarizes co-evolutionary patterns across the MSA and
    uses them to refine residue-pair features.

    5) Triangle Multiplication Outgoing
    -> residual update on z
    Refines pair features through triangular message passing.
    For a target pair (i, j), the pair (i, j) is fixed and a third residue k
    is scanned over. The update is built from relations such as (i, k) and
    (j, k), i.e. two edges pointing toward k.

    6) Triangle Multiplication Incoming
    -> residual update on z
    Complementary triangular message passing on z.
    For a target pair (i, j), the pair (i, j) is fixed and a third residue k
    is scanned over. The update is built from relations such as (k, i) and
    (k, j), i.e. two edges coming from k.

    In both triangle multiplication modules:
    - the target pair (i, j) is fixed
    - k is the intermediate residue that is scanned / aggregated over

    7) Triangle Attention Starting Node
    -> residual update on z
    Triangle-aware self-attention on pair features.
    For a target pair (i, j), z[i, j] is the pair being updated, and attention
    is performed over relations of the form (i, k). In other words, the query
    comes from the target pair z[i, j], while keys/values come from other pair
    vectors z[i, k] along the same starting node i.

    8) Triangle Attention Ending Node
    -> residual update on z
    Complementary triangle-aware self-attention on pair features.
    For a target pair (i, j), z[i, j] is the pair being updated, and attention
    is performed over relations of the form (k, j). The query comes from the
    target pair z[i, j], while keys/values come from other pair vectors z[k, j]
    along the same ending node j.

    9) Pair Transition
    -> residual update on z
    Position-wise feed-forward transformation on the pair representation.

    Summary
    -------
    - m is the MSA representation.
    - z is the pair representation.
    - Outer Product Mean is the main MSA -> pair bridge.
    - Triangle modules refine z by reasoning over residue triples (i, j, k).
    - In triangle attention, attention is not applied to a single pair vector alone:
    z[i, j] is the target/query pair, and it attends over related pair vectors
    such as z[i, k] or z[k, j].

    Simplified Order
    ----------------
    1) MSA Row Attention with Pair Bias   -> residual on m
    2) MSA Column Attention               -> residual on m
    3) MSA Transition                     -> residual on m
    4) Outer Product Mean                 -> residual on z, and serves as the
                                            main bridge from m to z
    5) Triangle Multiplication Outgoing   -> residual on z
    6) Triangle Multiplication Incoming   -> residual on z
    7) Triangle Attention Starting Node   -> residual on z
    8) Triangle Attention Ending Node     -> residual on z
    9) Pair Transition                    -> residual on z
    """

    def __init__(
        self,
        c_m=256,
        c_z=128,
        c_hidden_opm=32,
        c_hidden_tri_mul=128,
        num_heads_msa=8,
        num_heads_pair=4,
        c_hidden_msa_att=32,
        c_hidden_pair_att=32,
        transition_expansion=4,
        dropout=0.15):

        super().__init__()


        self.msa_row_attn = MSARowAttentionWithPairBias(
            c_m=c_m,
            c_z=c_z,
            num_heads=num_heads_msa,
            c_hidden=c_hidden_msa_att)

        self.msa_col_attn = MSAColumnAttention(
            c_m=c_m,
            num_heads=num_heads_msa,
            c_hidden=c_hidden_msa_att)

        self.msa_transition = MSATransition(
            c_m=c_m,
            expansion=transition_expansion,)

        self.outer_product_mean = OuterProductMean(
            c_m=c_m,
            c_hidden=c_hidden_opm,
            c_z=c_z,)

        self.tri_mul_out = TriangleMultiplicationOutgoing(
            c_z=c_z,
            c_hidden=c_hidden_tri_mul,)

        self.tri_mul_in = TriangleMultiplicationIncoming(
            c_z=c_z,
            c_hidden=c_hidden_tri_mul,)

        self.tri_attn_start = TriangleAttentionStartingNode(
            c_z=c_z,
            num_heads=num_heads_pair,
            c_hidden=c_hidden_pair_att,)

        self.tri_attn_end = TriangleAttentionEndingNode(
            c_z=c_z,
            num_heads=num_heads_pair,
            c_hidden=c_hidden_pair_att,)

        self.pair_transition = PairTransition(
            c_z=c_z,
            expansion=transition_expansion,)

        self.dropout = nn.Dropout(dropout)

        self._zero_init_residual_projections()

    def _zero_init_residual_projections(self):
        # final residual projections in each submodule
        zero_init_linear(self.msa_row_attn.output_linear)
        zero_init_linear(self.msa_col_attn.output_linear)
        zero_init_linear(self.outer_product_mean.output_linear)
        zero_init_linear(self.tri_mul_out.output_linear)
        zero_init_linear(self.tri_mul_in.output_linear)
        zero_init_linear(self.tri_attn_start.output_linear)
        zero_init_linear(self.tri_attn_end.output_linear)
        # msa_transition.linear_2 and pair_transition.linear_2
        # already zero-initialized in their constructors

    def forward(self, m, z, msa_mask=None, pair_mask=None):
        # ----- MSA stack -----
        m = m + self.dropout(self.msa_row_attn(m, z, msa_mask))
        m = m + self.dropout(self.msa_col_attn(m, msa_mask))
        m = m + self.dropout(self.msa_transition(m, msa_mask))

        # ----- MSA -> Pair -----
        z = z + self.dropout(self.outer_product_mean(m, msa_mask))

        # ----- Pair stack -----
        z = z + self.dropout(self.tri_mul_out(z, pair_mask))
        z = z + self.dropout(self.tri_mul_in(z, pair_mask))
        z = z + self.dropout(self.tri_attn_start(z, pair_mask))
        z = z + self.dropout(self.tri_attn_end(z, pair_mask))
        z = z + self.dropout(self.pair_transition(z, pair_mask))

        return m, z
