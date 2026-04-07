"""Test MSA row and column attention modules with shared silent-runner infrastructure."""

import math

import torch
import torch.nn as nn


from model.msa_row_attention import *
from model.msa_column_attention import *
from tests.test_helpers import (
    assert_close,
    assert_finite_tensor,
    assert_scalar_finite,
    assert_shape,
    finalize_test_results,
    run_test_silent,
)


# =========================================================
# Fake MSA batch
# =========================================================
torch.manual_seed(11)


def make_fake_msa_pair_batch(
    B=2,
    N=128,
    L=250,
    c_m=256,
    c_z=128,
    device="cpu",
    dtype=torch.float32,
):
    m = torch.randn(B, N, L, c_m, device=device, dtype=dtype)
    z = torch.randn(B, L, L, c_z, device=device, dtype=dtype)

    msa_mask = torch.ones(B, N, L, device=device, dtype=dtype)

    for b in range(B):
        cut = torch.randint(low=int(0.7 * L), high=L + 1, size=(1,)).item()
        msa_mask[b, :, cut:] = 0.0

    return {
        "m": m,
        "z": z,
        "msa_mask": msa_mask,
    }


def permute_residue_axis_in_m(m, perm):
    return m[:, :, perm, :]


def permute_residue_axis_in_msa_mask(msa_mask, perm):
    return msa_mask[:, :, perm]


def permute_residue_axes_in_z(z, perm):
    return z[:, perm][:, :, perm, :]


def permute_msa_axis_in_m(m, perm):
    return m[:, perm, :, :]


def permute_msa_axis_in_mask(msa_mask, perm):
    return msa_mask[:, perm, :]


# =========================================================
# Generic MSA attention tests
# =========================================================
def test_msa_attention_output_shape(module, batch, uses_pair_bias=False):
    module.eval()
    with torch.no_grad():
        if uses_pair_bias:
            out = module(batch["m"], batch["z"], batch["msa_mask"])
        else:
            out = module(batch["m"], batch["msa_mask"])

    B, N, L, c_m = batch["m"].shape
    assert_shape(out, (B, N, L, c_m), "msa_attention_out")


def test_msa_attention_output_finite(module, batch, uses_pair_bias=False):
    module.eval()
    with torch.no_grad():
        if uses_pair_bias:
            out = module(batch["m"], batch["z"], batch["msa_mask"])
        else:
            out = module(batch["m"], batch["msa_mask"])
    assert_finite_tensor(out, "msa_attention_out")


def test_msa_attention_deterministic_eval(module, batch, uses_pair_bias=False):
    module.eval()
    with torch.no_grad():
        if uses_pair_bias:
            out1 = module(batch["m"], batch["z"], batch["msa_mask"])
            out2 = module(batch["m"], batch["z"], batch["msa_mask"])
        else:
            out1 = module(batch["m"], batch["msa_mask"])
            out2 = module(batch["m"], batch["msa_mask"])
    assert_close(out1, out2, name="deterministic_eval")


def test_msa_attention_mask_all_ones_matches_unmasked(module, batch, uses_pair_bias=False):
    module.eval()
    B, N, L, _ = batch["m"].shape
    all_ones = torch.ones(B, N, L, device=batch["m"].device, dtype=batch["m"].dtype)

    with torch.no_grad():
        if uses_pair_bias:
            out_masked = module(batch["m"], batch["z"], all_ones)
            out_unmasked = module(batch["m"], batch["z"], None)
        else:
            out_masked = module(batch["m"], all_ones)
            out_unmasked = module(batch["m"], None)

    assert_close(
        out_masked, out_unmasked,
        atol=1e-5, rtol=1e-5,
        name="mask_all_ones_matches_unmasked"
    )


def test_msa_attention_all_zero_mask_gives_zero_output(module, batch, uses_pair_bias=False):
    module.eval()
    B, N, L, _ = batch["m"].shape
    zero_mask = torch.zeros(B, N, L, device=batch["m"].device, dtype=batch["m"].dtype)

    with torch.no_grad():
        if uses_pair_bias:
            out = module(batch["m"], batch["z"], zero_mask)
        else:
            out = module(batch["m"], zero_mask)

    assert_close(
        out, torch.zeros_like(out),
        atol=1e-7, rtol=1e-6,
        name="all_zero_mask_zero_output"
    )


def test_msa_attention_output_zero_on_masked_positions(module, batch, uses_pair_bias=False):
    module.eval()
    with torch.no_grad():
        if uses_pair_bias:
            out = module(batch["m"], batch["z"], batch["msa_mask"])
        else:
            out = module(batch["m"], batch["msa_mask"])

    masked = (batch["msa_mask"] == 0).unsqueeze(-1)
    assert_close(
        out.masked_select(masked),
        torch.zeros_like(out.masked_select(masked)),
        atol=1e-7, rtol=1e-6,
        name="masked_positions_are_zero"
    )


def test_msa_attention_gradient_flow(module, batch, uses_pair_bias=False):
    module.train()

    for p in module.parameters():
        if p.grad is not None:
            p.grad.zero_()

    if uses_pair_bias:
        out = module(batch["m"], batch["z"], batch["msa_mask"])
    else:
        out = module(batch["m"], batch["msa_mask"])

    loss = out.mean()
    assert_scalar_finite(loss, "loss")
    loss.backward()

    got_grad = False
    for name, p in module.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"{name} did not receive gradient"
            assert torch.isfinite(p.grad).all(), f"{name} gradient has NaN/Inf"
            got_grad = True

    assert got_grad, "No parameter got gradients"


def test_msa_attention_input_sensitivity_to_m(module, batch, uses_pair_bias=False):
    module.eval()

    m2 = batch["m"].clone()
    valid_idx = (batch["msa_mask"] > 0).nonzero(as_tuple=False)
    assert valid_idx.numel() > 0, "No valid positions in msa_mask"

    for row in valid_idx[:16]:
        b, n, l = row.tolist()
        m2[b, n, l, :] += 0.5

    with torch.no_grad():
        if uses_pair_bias:
            out1 = module(batch["m"], batch["z"], batch["msa_mask"])
            out2 = module(m2, batch["z"], batch["msa_mask"])
        else:
            out1 = module(batch["m"], batch["msa_mask"])
            out2 = module(m2, batch["msa_mask"])

    diff = (out1 - out2).abs().max().item()
    assert diff > 1e-8, f"Output did not change enough, max diff={diff}"


def test_msa_attention_masked_input_perturbation_has_no_effect(module, batch, uses_pair_bias=False):
    module.eval()

    m2 = batch["m"].clone()
    masked_idx = (batch["msa_mask"] == 0).nonzero(as_tuple=False)

    if masked_idx.numel() == 0:
        return

    for row in masked_idx[:16]:
        b, n, l = row.tolist()
        m2[b, n, l, :] += 10.0

    with torch.no_grad():
        if uses_pair_bias:
            out1 = module(batch["m"], batch["z"], batch["msa_mask"])
            out2 = module(m2, batch["z"], batch["msa_mask"])
        else:
            out1 = module(batch["m"], batch["msa_mask"])
            out2 = module(m2, batch["msa_mask"])

    assert_close(
        out1, out2,
        atol=1e-6, rtol=1e-5,
        name="masked_input_perturbation_no_effect"
    )


# =========================================================
# Row attention specific tests
# =========================================================
def test_msa_row_attention_pair_bias_affects_output(module, batch):
    module.eval()

    z2 = batch["z"].clone()
    z2[:, :8, :8, :] += 0.75

    with torch.no_grad():
        out1 = module(batch["m"], batch["z"], batch["msa_mask"])
        out2 = module(batch["m"], z2, batch["msa_mask"])

    diff = (out1 - out2).abs().max().item()
    assert diff > 1e-8, f"Output did not change enough after changing z, max diff={diff}"


def test_msa_row_attention_residue_permutation_equivariance(module, batch):
    module.eval()

    L = batch["m"].shape[2]
    perm = torch.randperm(L, device=batch["m"].device)

    m_perm = permute_residue_axis_in_m(batch["m"], perm)
    mask_perm = permute_residue_axis_in_msa_mask(batch["msa_mask"], perm)
    z_perm = permute_residue_axes_in_z(batch["z"], perm)

    with torch.no_grad():
        out = module(batch["m"], batch["z"], batch["msa_mask"])
        out_perm = module(m_perm, z_perm, mask_perm)

    out_expected = permute_residue_axis_in_m(out, perm)

    assert_close(
        out_perm, out_expected,
        atol=1e-5, rtol=1e-5,
        name="row_residue_permutation_equivariance"
    )


# =========================================================
# Column attention specific tests
# =========================================================
def test_msa_column_attention_msa_permutation_equivariance(module, batch):
    module.eval()

    N = batch["m"].shape[1]
    perm = torch.randperm(N, device=batch["m"].device)

    m_perm = permute_msa_axis_in_m(batch["m"], perm)
    mask_perm = permute_msa_axis_in_mask(batch["msa_mask"], perm)

    with torch.no_grad():
        out = module(batch["m"], batch["msa_mask"])
        out_perm = module(m_perm, mask_perm)

    out_expected = permute_msa_axis_in_m(out, perm)

    assert_close(
        out_perm, out_expected,
        atol=1e-5, rtol=1e-5,
        name="column_msa_permutation_equivariance"
    )


def test_msa_column_attention_residue_permutation_equivariance(module, batch):
    module.eval()

    L = batch["m"].shape[2]
    perm = torch.randperm(L, device=batch["m"].device)

    m_perm = permute_residue_axis_in_m(batch["m"], perm)
    mask_perm = permute_residue_axis_in_msa_mask(batch["msa_mask"], perm)

    with torch.no_grad():
        out = module(batch["m"], batch["msa_mask"])
        out_perm = module(m_perm, mask_perm)

    out_expected = permute_residue_axis_in_m(out, perm)

    assert_close(
        out_perm, out_expected,
        atol=1e-5, rtol=1e-5,
        name="column_residue_permutation_equivariance"
    )


# =========================================================
# Silent orchestrators
# =========================================================
def run_msa_row_attention_with_pair_bias_test_suite(
    c_m=256,
    c_z=128,
    num_heads=8,
    c_hidden=32,
    B=2,
    N=128,
    L=250,
    device="cpu",
):
    batch = make_fake_msa_pair_batch(
        B=B, N=N, L=L, c_m=c_m, c_z=c_z, device=device
    )

    module = MSARowAttentionWithPairBias(
        c_m=c_m,
        c_z=c_z,
        num_heads=num_heads,
        c_hidden=c_hidden,
    ).to(device)

    tests = [
        ("output_shape", lambda: test_msa_attention_output_shape(module, batch, uses_pair_bias=True)),
        ("output_finite", lambda: test_msa_attention_output_finite(module, batch, uses_pair_bias=True)),
        ("deterministic_eval", lambda: test_msa_attention_deterministic_eval(module, batch, uses_pair_bias=True)),
        ("mask_all_ones_matches_unmasked", lambda: test_msa_attention_mask_all_ones_matches_unmasked(module, batch, uses_pair_bias=True)),
        ("all_zero_mask_gives_zero_output", lambda: test_msa_attention_all_zero_mask_gives_zero_output(module, batch, uses_pair_bias=True)),
        ("output_zero_on_masked_positions", lambda: test_msa_attention_output_zero_on_masked_positions(module, batch, uses_pair_bias=True)),
        ("gradient_flow", lambda: test_msa_attention_gradient_flow(module, batch, uses_pair_bias=True)),
        ("input_sensitivity_to_m", lambda: test_msa_attention_input_sensitivity_to_m(module, batch, uses_pair_bias=True)),
        ("masked_input_perturbation_has_no_effect", lambda: test_msa_attention_masked_input_perturbation_has_no_effect(module, batch, uses_pair_bias=True)),
        ("pair_bias_affects_output", lambda: test_msa_row_attention_pair_bias_affects_output(module, batch)),
        ("row_residue_permutation_equivariance", lambda: test_msa_row_attention_residue_permutation_equivariance(module, batch)),
    ]

    results = [run_test_silent(name, fn) for name, fn in tests]
    finalize_test_results(results, suite_name="MSARowAttentionWithPairBias")
    return module, batch, results


def run_msa_column_attention_test_suite(
    c_m=256,
    num_heads=8,
    c_hidden=32,
    B=2,
    N=128,
    L=250,
    device="cpu",
):
    batch = make_fake_msa_pair_batch(
        B=B, N=N, L=L, c_m=c_m, c_z=128, device=device
    )

    module = MSAColumnAttention(
        c_m=c_m,
        num_heads=num_heads,
        c_hidden=c_hidden,
    ).to(device)

    tests = [
        ("output_shape", lambda: test_msa_attention_output_shape(module, batch, uses_pair_bias=False)),
        ("output_finite", lambda: test_msa_attention_output_finite(module, batch, uses_pair_bias=False)),
        ("deterministic_eval", lambda: test_msa_attention_deterministic_eval(module, batch, uses_pair_bias=False)),
        ("mask_all_ones_matches_unmasked", lambda: test_msa_attention_mask_all_ones_matches_unmasked(module, batch, uses_pair_bias=False)),
        ("all_zero_mask_gives_zero_output", lambda: test_msa_attention_all_zero_mask_gives_zero_output(module, batch, uses_pair_bias=False)),
        ("output_zero_on_masked_positions", lambda: test_msa_attention_output_zero_on_masked_positions(module, batch, uses_pair_bias=False)),
        ("gradient_flow", lambda: test_msa_attention_gradient_flow(module, batch, uses_pair_bias=False)),
        ("input_sensitivity_to_m", lambda: test_msa_attention_input_sensitivity_to_m(module, batch, uses_pair_bias=False)),
        ("masked_input_perturbation_has_no_effect", lambda: test_msa_attention_masked_input_perturbation_has_no_effect(module, batch, uses_pair_bias=False)),
        ("column_msa_permutation_equivariance", lambda: test_msa_column_attention_msa_permutation_equivariance(module, batch)),
        ("column_residue_permutation_equivariance", lambda: test_msa_column_attention_residue_permutation_equivariance(module, batch)),
    ]

    results = [run_test_silent(name, fn) for name, fn in tests]
    finalize_test_results(results, suite_name="MSAColumnAttention")
    return module, batch, results
