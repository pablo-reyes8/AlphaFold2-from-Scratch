"""Test triangle attention blocks with shared silent-runner helpers and pairwise fixtures."""

import copy
import math

import torch
import torch.nn as nn

from model.triangle_attention import *
from tests.test_helpers import (
    assert_close,
    assert_finite_tensor,
    assert_scalar_finite,
    assert_shape,
    finalize_test_results,
    run_test_silent,
)


# =========================================================
# Fake pair batch
# =========================================================
torch.manual_seed(11)


def make_fake_pair_batch(
    B=2,
    L=250,
    c_z=128,
    device="cpu",
    dtype=torch.float32,
):
    z = torch.randn(B, L, L, c_z, device=device, dtype=dtype)

    pair_mask = torch.ones(B, L, L, device=device, dtype=dtype)

    for b in range(B):
        cut = torch.randint(low=int(0.7 * L), high=L + 1, size=(1,)).item()
        pair_mask[b, cut:, :] = 0.0
        pair_mask[b, :, cut:] = 0.0

    return {
        "z": z,
        "pair_mask": pair_mask,
    }


def permute_pair_tensor(z, perm):
    return z[:, perm][:, :, perm, :]


def permute_pair_mask(pair_mask, perm):
    return pair_mask[:, perm][:, :, perm]


# =========================================================
# Generic tests for triangle attention modules
# =========================================================
def test_triangle_attention_output_shape(module, batch):
    module.eval()
    with torch.no_grad():
        out = module(batch["z"], batch["pair_mask"])

    B, L, _, c_z = batch["z"].shape
    assert_shape(out, (B, L, L, c_z), "triangle_attention_out")


def test_triangle_attention_output_finite(module, batch):
    module.eval()
    with torch.no_grad():
        out = module(batch["z"], batch["pair_mask"])
    assert_finite_tensor(out, "triangle_attention_out")


def test_triangle_attention_deterministic_eval(module, batch):
    module.eval()
    with torch.no_grad():
        out1 = module(batch["z"], batch["pair_mask"])
        out2 = module(batch["z"], batch["pair_mask"])
    assert_close(out1, out2, name="deterministic_eval")


def test_triangle_attention_mask_all_ones_matches_unmasked(module, batch):
    module.eval()
    B, L, _, _ = batch["z"].shape
    all_ones = torch.ones(B, L, L, device=batch["z"].device, dtype=batch["z"].dtype)

    with torch.no_grad():
        out_masked = module(batch["z"], all_ones)
        out_unmasked = module(batch["z"], None)

    assert_close(
        out_masked, out_unmasked,
        atol=1e-5, rtol=1e-5,
        name="mask_all_ones_matches_unmasked"
    )


def test_triangle_attention_all_zero_mask_gives_zero_output(module, batch):
    module.eval()
    B, L, _, _ = batch["z"].shape
    zero_mask = torch.zeros(B, L, L, device=batch["z"].device, dtype=batch["z"].dtype)

    with torch.no_grad():
        out = module(batch["z"], zero_mask)

    assert_close(
        out, torch.zeros_like(out),
        atol=1e-7, rtol=1e-6,
        name="all_zero_mask_zero_output"
    )


def test_triangle_attention_output_zero_on_masked_positions(module, batch):
    module.eval()
    with torch.no_grad():
        out = module(batch["z"], batch["pair_mask"])

    masked = (batch["pair_mask"] == 0).unsqueeze(-1)
    assert_close(
        out.masked_select(masked),
        torch.zeros_like(out.masked_select(masked)),
        atol=1e-7, rtol=1e-6,
        name="masked_positions_are_zero"
    )


def test_triangle_attention_gradient_flow(module, batch):
    module.train()

    for p in module.parameters():
        if p.grad is not None:
            p.grad.zero_()

    out = module(batch["z"], batch["pair_mask"])
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


def test_triangle_attention_input_sensitivity(module, batch):
    module.eval()

    z2 = batch["z"].clone()
    valid_idx = (batch["pair_mask"] > 0).nonzero(as_tuple=False)
    assert valid_idx.numel() > 0, "No valid positions in pair_mask"

    for row in valid_idx[:16]:
        b, i, j = row.tolist()
        z2[b, i, j, :] += 0.5

    with torch.no_grad():
        out1 = module(batch["z"], batch["pair_mask"])
        out2 = module(z2, batch["pair_mask"])

    diff = (out1 - out2).abs().max().item()
    assert diff > 1e-8, f"Output did not change enough, max diff={diff}"


def test_triangle_attention_masked_input_perturbation_has_no_effect(module, batch):
    module.eval()

    z2 = batch["z"].clone()
    masked_idx = (batch["pair_mask"] == 0).nonzero(as_tuple=False)

    if masked_idx.numel() == 0:
        return

    for row in masked_idx[:16]:
        b, i, j = row.tolist()
        z2[b, i, j, :] += 10.0

    with torch.no_grad():
        out1 = module(batch["z"], batch["pair_mask"])
        out2 = module(z2, batch["pair_mask"])

    assert_close(
        out1, out2,
        atol=1e-6, rtol=1e-5,
        name="masked_input_perturbation_no_effect"
    )


def test_triangle_attention_permutation_equivariance(module, batch):
    module.eval()
    L = batch["z"].shape[1]
    perm = torch.randperm(L, device=batch["z"].device)

    z_perm = permute_pair_tensor(batch["z"], perm)
    mask_perm = permute_pair_mask(batch["pair_mask"], perm)

    with torch.no_grad():
        out = module(batch["z"], batch["pair_mask"])
        out_perm = module(z_perm, mask_perm)

    out_expected = permute_pair_tensor(out, perm)

    assert_close(
        out_perm, out_expected,
        atol=1e-5, rtol=1e-5,
        name="permutation_equivariance"
    )


# =========================================================
# Structural relation between StartingNode and EndingNode
# =========================================================
def test_triangle_attention_starting_ending_transpose_equivalence(
    batch,
    c_z=128,
    num_heads=4,
    c_hidden=32,
    device="cpu",
):
    """
    Propiedad estructural correcta:

        Ending(z, mask) == Starting(z^T, mask^T)^T

    si ambos módulos comparten exactamente los mismos pesos.
    """
    starting = TriangleAttentionStartingNode(
        c_z=c_z, num_heads=num_heads, c_hidden=c_hidden
    ).to(device)

    ending = TriangleAttentionEndingNode(
        c_z=c_z, num_heads=num_heads, c_hidden=c_hidden
    ).to(device)

    ending.load_state_dict(copy.deepcopy(starting.state_dict()))

    starting.eval()
    ending.eval()

    z = batch["z"]
    mask = batch["pair_mask"]

    z_t = z.transpose(1, 2)
    mask_t = mask.transpose(1, 2)

    with torch.no_grad():
        out_end = ending(z, mask)
        out_start_t = starting(z_t, mask_t).transpose(1, 2)

    assert_close(
        out_end, out_start_t,
        atol=1e-5, rtol=1e-5,
        name="starting_ending_transpose_equivalence"
    )


# =========================================================
# Silent orchestrators
# =========================================================
def run_triangle_attention_starting_test_suite(
    c_z=128,
    num_heads=4,
    c_hidden=32,
    B=2,
    L=250,
    device="cpu",
):
    batch = make_fake_pair_batch(B=B, L=L, c_z=c_z, device=device)
    module = TriangleAttentionStartingNode(
        c_z=c_z,
        num_heads=num_heads,
        c_hidden=c_hidden,
    ).to(device)

    tests = [
        ("output_shape", lambda: test_triangle_attention_output_shape(module, batch)),
        ("output_finite", lambda: test_triangle_attention_output_finite(module, batch)),
        ("deterministic_eval", lambda: test_triangle_attention_deterministic_eval(module, batch)),
        ("mask_all_ones_matches_unmasked", lambda: test_triangle_attention_mask_all_ones_matches_unmasked(module, batch)),
        ("all_zero_mask_gives_zero_output", lambda: test_triangle_attention_all_zero_mask_gives_zero_output(module, batch)),
        ("output_zero_on_masked_positions", lambda: test_triangle_attention_output_zero_on_masked_positions(module, batch)),
        ("gradient_flow", lambda: test_triangle_attention_gradient_flow(module, batch)),
        ("input_sensitivity", lambda: test_triangle_attention_input_sensitivity(module, batch)),
        ("masked_input_perturbation_has_no_effect", lambda: test_triangle_attention_masked_input_perturbation_has_no_effect(module, batch)),
        ("permutation_equivariance", lambda: test_triangle_attention_permutation_equivariance(module, batch)),
    ]

    results = [run_test_silent(name, fn) for name, fn in tests]
    finalize_test_results(results, suite_name="TriangleAttentionStartingNode")
    return module, batch, results


def run_triangle_attention_ending_test_suite(
    c_z=128,
    num_heads=4,
    c_hidden=32,
    B=2,
    L=250,
    device="cpu",
):
    batch = make_fake_pair_batch(B=B, L=L, c_z=c_z, device=device)
    module = TriangleAttentionEndingNode(
        c_z=c_z,
        num_heads=num_heads,
        c_hidden=c_hidden,
    ).to(device)

    tests = [
        ("output_shape", lambda: test_triangle_attention_output_shape(module, batch)),
        ("output_finite", lambda: test_triangle_attention_output_finite(module, batch)),
        ("deterministic_eval", lambda: test_triangle_attention_deterministic_eval(module, batch)),
        ("mask_all_ones_matches_unmasked", lambda: test_triangle_attention_mask_all_ones_matches_unmasked(module, batch)),
        ("all_zero_mask_gives_zero_output", lambda: test_triangle_attention_all_zero_mask_gives_zero_output(module, batch)),
        ("output_zero_on_masked_positions", lambda: test_triangle_attention_output_zero_on_masked_positions(module, batch)),
        ("gradient_flow", lambda: test_triangle_attention_gradient_flow(module, batch)),
        ("input_sensitivity", lambda: test_triangle_attention_input_sensitivity(module, batch)),
        ("masked_input_perturbation_has_no_effect", lambda: test_triangle_attention_masked_input_perturbation_has_no_effect(module, batch)),
        ("permutation_equivariance", lambda: test_triangle_attention_permutation_equivariance(module, batch)),
    ]

    results = [run_test_silent(name, fn) for name, fn in tests]
    finalize_test_results(results, suite_name="TriangleAttentionEndingNode")
    return module, batch, results


def run_triangle_attention_joint_structural_test_suite(
    c_z=128,
    num_heads=4,
    c_hidden=32,
    B=2,
    L=250,
    device="cpu",
):
    batch = make_fake_pair_batch(B=B, L=L, c_z=c_z, device=device)

    tests = [
        (
            "starting_ending_transpose_equivalence",
            lambda: test_triangle_attention_starting_ending_transpose_equivalence(
                batch=batch,
                c_z=c_z,
                num_heads=num_heads,
                c_hidden=c_hidden,
                device=device,
            ),
        ),
    ]

    results = [run_test_silent(name, fn) for name, fn in tests]
    finalize_test_results(results, suite_name="TriangleAttentionJoint")
    return batch, results
