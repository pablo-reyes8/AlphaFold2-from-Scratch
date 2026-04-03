"""Validate Evoformer gating flags without instantiating the full AlphaFold2 model."""

from __future__ import annotations

import torch
import torch.nn as nn

from model.evoformer_block import EvoformerBlock
from model.evoformer_stack import EvoformerStack


class _ConstantResidual(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = float(value)

    def forward(self, tensor, *args, **kwargs):
        return torch.full_like(tensor, self.value)


class _OuterProductMeanResidual(nn.Module):
    def __init__(self, value: float, c_z: int):
        super().__init__()
        self.value = float(value)
        self.c_z = int(c_z)

    def forward(self, m, *args, **kwargs):
        batch, _, seq_len, _ = m.shape
        return torch.full(
            (batch, seq_len, seq_len, self.c_z),
            self.value,
            dtype=m.dtype,
            device=m.device,
        )


def _stub_block(
    *,
    pair_stack_enabled=True,
    triangle_multiplication_enabled=True,
    triangle_attention_enabled=True,
    pair_transition_enabled=True,
):
    c_z = 4
    block = EvoformerBlock.__new__(EvoformerBlock)
    nn.Module.__init__(block)
    block.pair_stack_enabled = pair_stack_enabled
    block.triangle_multiplication_enabled = triangle_multiplication_enabled
    block.triangle_attention_enabled = triangle_attention_enabled
    block.pair_transition_enabled = pair_transition_enabled
    block.msa_row_attn = _ConstantResidual(1.0)
    block.msa_col_attn = _ConstantResidual(2.0)
    block.msa_transition = _ConstantResidual(3.0)
    block.outer_product_mean = _OuterProductMeanResidual(4.0, c_z=c_z)
    block.tri_mul_out = _ConstantResidual(5.0)
    block.tri_mul_in = _ConstantResidual(6.0)
    block.tri_attn_start = _ConstantResidual(7.0)
    block.tri_attn_end = _ConstantResidual(8.0)
    block.pair_transition = _ConstantResidual(9.0)
    block.dropout = nn.Identity()
    return block


def test_evoformer_block_can_skip_entire_pair_stack():
    block = _stub_block(pair_stack_enabled=False)
    m = torch.zeros(1, 2, 3, 4)
    z = torch.zeros(1, 3, 3, 4)

    m_out, z_out = block(m, z)

    assert torch.allclose(m_out, torch.full_like(m, 6.0))
    assert torch.allclose(z_out, torch.full_like(z, 4.0))


def test_evoformer_block_can_keep_triangle_multiplication_without_attention():
    block = _stub_block(
        pair_stack_enabled=True,
        triangle_multiplication_enabled=True,
        triangle_attention_enabled=False,
        pair_transition_enabled=True,
    )
    m = torch.zeros(1, 2, 3, 4)
    z = torch.zeros(1, 3, 3, 4)

    _, z_out = block(m, z)

    expected = 4.0 + 5.0 + 6.0 + 9.0
    assert torch.allclose(z_out, torch.full_like(z, expected))


def test_evoformer_stack_propagates_pair_stack_flags_to_blocks():
    stack = EvoformerStack(
        num_blocks=2,
        c_m=256,
        c_z=128,
        pair_stack_enabled=False,
        triangle_multiplication_enabled=False,
        triangle_attention_enabled=False,
        pair_transition_enabled=False,
    )

    assert len(stack.blocks) == 2
    assert all(block.pair_stack_enabled is False for block in stack.blocks)
    assert all(block.triangle_multiplication_enabled is False for block in stack.blocks)
    assert all(block.triangle_attention_enabled is False for block in stack.blocks)
    assert all(block.pair_transition_enabled is False for block in stack.blocks)
