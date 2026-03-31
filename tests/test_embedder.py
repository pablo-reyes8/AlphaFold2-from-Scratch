from __future__ import annotations

import torch

from model.input_embedder import InputEmbedder


def test_input_embedder_shapes_and_masks():
    batch_size = 2
    msa_depth = 8
    length = 24
    n_tokens = 23
    pad_idx = 0

    seq_tokens = torch.randint(1, n_tokens, (batch_size, length))
    msa_tokens = torch.randint(1, n_tokens, (batch_size, msa_depth, length))

    seq_mask = torch.ones(batch_size, length, dtype=torch.float32)
    msa_mask = torch.ones(batch_size, msa_depth, length, dtype=torch.float32)

    seq_tokens[0, -5:] = pad_idx
    seq_mask[0, -5:] = 0.0
    msa_tokens[0, :, -5:] = pad_idx
    msa_mask[0, :, -5:] = 0.0

    model = InputEmbedder(
        n_tokens=n_tokens,
        c_m=256,
        c_z=128,
        c_s=256,
        max_relpos=32,
        pad_idx=pad_idx,
    )

    msa_repr, pair_repr = model(
        seq_tokens=seq_tokens,
        msa_tokens=msa_tokens,
        seq_mask=seq_mask,
        msa_mask=msa_mask,
    )

    assert msa_repr.shape == (batch_size, msa_depth, length, 256)
    assert pair_repr.shape == (batch_size, length, length, 128)
    assert torch.isfinite(msa_repr).all()
    assert torch.isfinite(pair_repr).all()

    assert torch.allclose(
        msa_repr[0, :, -5:, :],
        torch.zeros_like(msa_repr[0, :, -5:, :]),
        atol=1e-5,
    )
    assert torch.allclose(
        pair_repr[0, -5:, :, :],
        torch.zeros_like(pair_repr[0, -5:, :, :]),
        atol=1e-5,
    )
    assert torch.allclose(
        pair_repr[0, :, -5:, :],
        torch.zeros_like(pair_repr[0, :, -5:, :]),
        atol=1e-5,
    )


def test_input_embedder_is_deterministic_for_fixed_inputs():
    model = InputEmbedder(
        n_tokens=27,
        c_m=256,
        c_z=128,
        c_s=256,
        max_relpos=32,
        pad_idx=0,
    )

    seq_tokens = torch.randint(1, 27, (1, 10))
    msa_tokens = torch.randint(1, 27, (1, 2, 10))

    msa_repr_1, pair_repr_1 = model(seq_tokens=seq_tokens, msa_tokens=msa_tokens)
    msa_repr_2, pair_repr_2 = model(seq_tokens=seq_tokens, msa_tokens=msa_tokens)

    assert torch.allclose(msa_repr_1, msa_repr_2, atol=1e-6)
    assert torch.allclose(pair_repr_1, pair_repr_2, atol=1e-6)
