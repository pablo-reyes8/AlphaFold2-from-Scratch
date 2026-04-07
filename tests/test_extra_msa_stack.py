"""Test the extra MSA stack with CPU-safe random tensors and structural assertions."""

import copy

import pytest
import torch

from model.extra_msa_stack import ExtraMsaStack
from tests.test_helpers import (
    assert_close,
    assert_finite_tensor,
    assert_not_close,
    assert_scalar_finite,
    assert_shape,
    finalize_test_results,
    run_test_silent,
)


torch.manual_seed(11)


@pytest.fixture
def batch():
    return make_fake_extra_msa_batch(device="cpu")


@pytest.fixture
def module(batch):
    return _build_module(
        c_m=batch["m"].shape[-1],
        c_z=batch["z"].shape[-1],
        extra_dim=batch["extra_msa_feat"].shape[-1],
        device="cpu",
    )


def make_fake_extra_msa_batch(
    B=2,
    N_msa=3,
    N_extra=6,
    L=24,
    c_m=48,
    c_z=64,
    extra_dim=25,
    device="cpu",
    dtype=torch.float32,
):
    m = torch.randn(B, N_msa, L, c_m, device=device, dtype=dtype)
    z = torch.randn(B, L, L, c_z, device=device, dtype=dtype)
    extra_msa_feat = torch.randn(B, N_extra, L, extra_dim, device=device, dtype=dtype)
    seq_mask = torch.ones(B, L, device=device, dtype=dtype)
    extra_msa_mask = torch.ones(B, N_extra, L, device=device, dtype=dtype)

    for batch_index in range(B):
        cut = torch.randint(low=int(0.65 * L), high=L + 1, size=(1,)).item()
        seq_mask[batch_index, cut:] = 0.0
        extra_msa_mask[batch_index, :, cut:] = 0.0
        extra_msa_mask[batch_index, -1, :] = 0.0

    return {
        "m": m,
        "z": z,
        "extra_msa_feat": extra_msa_feat,
        "seq_mask": seq_mask,
        "extra_msa_mask": extra_msa_mask,
    }


def _build_module(c_m=48, c_z=64, extra_dim=25, device="cpu"):
    return ExtraMsaStack(
        c_m=c_m,
        c_z=c_z,
        extra_dim=extra_dim,
        c_e=32,
        num_blocks=1,
        c_hidden_opm=16,
        c_hidden_tri_mul=32,
        num_heads_msa=4,
        num_heads_pair=4,
        c_hidden_msa_att=8,
        c_hidden_pair_att=16,
        transition_expansion=2,
        dropout_msa=0.0,
        dropout_pair=0.0,
    ).to(device)


def _build_sensitive_module(module):
    sensitive = copy.deepcopy(module)
    block = sensitive.blocks[0]
    with torch.no_grad():
        block.outer_product_mean.output_linear.weight.normal_(mean=0.0, std=0.02)
        block.outer_product_mean.output_linear.bias.normal_(mean=0.0, std=0.02)
    return sensitive


def test_extra_msa_stack_shapes_and_m_passthrough(module, batch):
    module.eval()
    with torch.no_grad():
        m_out, z_out = module(
            batch["m"],
            batch["z"],
            extra_msa_feat=batch["extra_msa_feat"],
            seq_mask=batch["seq_mask"],
            extra_msa_mask=batch["extra_msa_mask"],
        )

    assert_shape(m_out, batch["m"].shape, "m_out")
    assert_shape(z_out, batch["z"].shape, "z_out")
    assert_close(m_out, batch["m"], name="m_passthrough")


def test_extra_msa_stack_output_finite(module, batch):
    module.eval()
    with torch.no_grad():
        _, z_out = module(
            batch["m"],
            batch["z"],
            extra_msa_feat=batch["extra_msa_feat"],
            seq_mask=batch["seq_mask"],
            extra_msa_mask=batch["extra_msa_mask"],
        )
    assert_finite_tensor(z_out, "z_out")


def test_extra_msa_stack_deterministic_eval(module, batch):
    module.eval()
    with torch.no_grad():
        _, z_first = module(
            batch["m"],
            batch["z"],
            extra_msa_feat=batch["extra_msa_feat"],
            seq_mask=batch["seq_mask"],
            extra_msa_mask=batch["extra_msa_mask"],
        )
        _, z_second = module(
            batch["m"],
            batch["z"],
            extra_msa_feat=batch["extra_msa_feat"],
            seq_mask=batch["seq_mask"],
            extra_msa_mask=batch["extra_msa_mask"],
        )
    assert_close(z_first, z_second, name="deterministic_eval")


def test_extra_msa_stack_none_extra_is_identity(module, batch):
    module.eval()
    with torch.no_grad():
        m_out, z_out = module(batch["m"], batch["z"], extra_msa_feat=None, seq_mask=batch["seq_mask"])
    assert_close(m_out, batch["m"], name="identity_m")
    assert_close(z_out, batch["z"], name="identity_z")


def test_extra_msa_stack_ignores_fully_masked_extra_sequences(module, batch):
    module.eval()
    B, _, L, extra_dim = batch["extra_msa_feat"].shape
    extra_feat = torch.randn(B, 2, L, extra_dim, device=batch["m"].device, dtype=batch["m"].dtype)
    extra_mask = torch.zeros(B, 2, L, device=batch["m"].device, dtype=batch["m"].dtype)

    feat_aug = torch.cat([batch["extra_msa_feat"], extra_feat], dim=1)
    mask_aug = torch.cat([batch["extra_msa_mask"], extra_mask], dim=1)

    with torch.no_grad():
        _, z_base = module(
            batch["m"],
            batch["z"],
            extra_msa_feat=batch["extra_msa_feat"],
            seq_mask=batch["seq_mask"],
            extra_msa_mask=batch["extra_msa_mask"],
        )
        _, z_aug = module(
            batch["m"],
            batch["z"],
            extra_msa_feat=feat_aug,
            seq_mask=batch["seq_mask"],
            extra_msa_mask=mask_aug,
        )

    assert_close(z_base, z_aug, atol=1e-5, rtol=1e-5, name="ignore_masked_extra_sequences")


def test_extra_msa_stack_masked_perturbation_has_no_effect(module, batch):
    module = _build_sensitive_module(module).eval()
    extra_msa_feat = batch["extra_msa_feat"].clone()
    masked_index = (batch["extra_msa_mask"] == 0).nonzero(as_tuple=False)
    assert masked_index.numel() > 0, "Expected masked positions in extra_msa_mask"

    for row in masked_index[:24]:
        b, n, l = row.tolist()
        extra_msa_feat[b, n, l, :] += 10.0

    with torch.no_grad():
        _, z_base = module(
            batch["m"],
            batch["z"],
            extra_msa_feat=batch["extra_msa_feat"],
            seq_mask=batch["seq_mask"],
            extra_msa_mask=batch["extra_msa_mask"],
        )
        _, z_perturbed = module(
            batch["m"],
            batch["z"],
            extra_msa_feat=extra_msa_feat,
            seq_mask=batch["seq_mask"],
            extra_msa_mask=batch["extra_msa_mask"],
        )

    assert_close(z_base, z_perturbed, atol=1e-5, rtol=1e-5, name="masked_perturbation_no_effect")


def test_extra_msa_stack_sensitive_when_valid_inputs_change(module, batch):
    module = _build_sensitive_module(module).eval()
    extra_msa_feat = batch["extra_msa_feat"].clone()
    valid_index = (batch["extra_msa_mask"] > 0).nonzero(as_tuple=False)
    b, n, l = valid_index[0].tolist()
    extra_msa_feat[b, n, l, :] += 1.0

    with torch.no_grad():
        _, z_base = module(
            batch["m"],
            batch["z"],
            extra_msa_feat=batch["extra_msa_feat"],
            seq_mask=batch["seq_mask"],
            extra_msa_mask=batch["extra_msa_mask"],
        )
        _, z_changed = module(
            batch["m"],
            batch["z"],
            extra_msa_feat=extra_msa_feat,
            seq_mask=batch["seq_mask"],
            extra_msa_mask=batch["extra_msa_mask"],
        )

    assert_not_close(z_base, z_changed, atol=1e-7, rtol=1e-6, name="valid_input_sensitivity")


def test_extra_msa_stack_gradient_flow(module, batch):
    module.train()
    _, z_out = module(
        batch["m"],
        batch["z"],
        extra_msa_feat=batch["extra_msa_feat"],
        seq_mask=batch["seq_mask"],
        extra_msa_mask=batch["extra_msa_mask"],
    )
    loss = z_out.mean()
    assert_scalar_finite(loss, "loss")
    loss.backward()

    got_grad = False
    for name, parameter in module.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, f"{name} did not receive gradient"
            assert torch.isfinite(parameter.grad).all(), f"{name} gradient has NaN/Inf"
            got_grad = True
    assert got_grad, "No parameter got gradients"


def run_extra_msa_stack_test_suite(device="cpu"):
    batch = make_fake_extra_msa_batch(device=device)
    module = _build_module(
        c_m=batch["m"].shape[-1],
        c_z=batch["z"].shape[-1],
        extra_dim=batch["extra_msa_feat"].shape[-1],
        device=device,
    )

    tests = [
        ("shapes_and_m_passthrough", lambda: test_extra_msa_stack_shapes_and_m_passthrough(module, batch)),
        ("output_finite", lambda: test_extra_msa_stack_output_finite(module, batch)),
        ("deterministic_eval", lambda: test_extra_msa_stack_deterministic_eval(module, batch)),
        ("none_extra_is_identity", lambda: test_extra_msa_stack_none_extra_is_identity(module, batch)),
        ("ignores_fully_masked_extra_sequences", lambda: test_extra_msa_stack_ignores_fully_masked_extra_sequences(module, batch)),
        ("masked_perturbation_has_no_effect", lambda: test_extra_msa_stack_masked_perturbation_has_no_effect(module, batch)),
        ("sensitive_when_valid_inputs_change", lambda: test_extra_msa_stack_sensitive_when_valid_inputs_change(module, batch)),
        ("gradient_flow", lambda: test_extra_msa_stack_gradient_flow(module, batch)),
    ]

    results = [run_test_silent(name, fn) for name, fn in tests]
    finalize_test_results(results, suite_name="ExtraMsaStack")


def test_extra_msa_stack_suite():
    run_extra_msa_stack_test_suite(device="cpu")
