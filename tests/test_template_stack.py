"""Test the template stack with CPU-safe random tensors and masking invariants."""

import copy

import pytest
import torch

from model.template_stack import TemplateStack, augment_msa_mask_with_template_mask, normalize_template_mask
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
    return make_fake_template_batch(device="cpu")


@pytest.fixture
def module(batch):
    return _build_module(
        c_m=batch["m"].shape[-1],
        c_z=batch["z"].shape[-1],
        angle_dim=batch["template_angle_feat"].shape[-1],
        pair_dim=batch["template_pair_feat"].shape[-1],
        device="cpu",
    )


def make_fake_template_batch(
    B=2,
    N_msa=3,
    T=2,
    L=18,
    c_m=32,
    c_z=64,
    angle_dim=51,
    pair_dim=88,
    device="cpu",
    dtype=torch.float32,
):
    m = torch.randn(B, N_msa, L, c_m, device=device, dtype=dtype)
    z = torch.randn(B, L, L, c_z, device=device, dtype=dtype)
    template_angle_feat = torch.randn(B, T, L, angle_dim, device=device, dtype=dtype)
    template_pair_feat = torch.randn(B, T, L, L, pair_dim, device=device, dtype=dtype)
    template_mask = torch.ones(B, T, L, device=device, dtype=dtype)

    for batch_index in range(B):
        cut = torch.randint(low=int(0.65 * L), high=L + 1, size=(1,)).item()
        template_mask[batch_index, :, cut:] = 0.0
        template_mask[batch_index, -1, :] = 0.0

    return {
        "m": m,
        "z": z,
        "template_angle_feat": template_angle_feat,
        "template_pair_feat": template_pair_feat,
        "template_mask": template_mask,
    }


def _build_module(c_m=32, c_z=64, angle_dim=51, pair_dim=88, device="cpu"):
    return TemplateStack(
        c_m=c_m,
        c_z=c_z,
        template_angle_dim=angle_dim,
        template_pair_dim=pair_dim,
        c_t=32,
        num_blocks=1,
        num_heads=4,
        c_hidden_att=8,
        c_hidden_mul=32,
        transition_expansion=2,
        dropout=0.0,
    ).to(device)


def _build_sensitive_pair_module(module):
    sensitive = copy.deepcopy(module)
    with torch.no_grad():
        sensitive.template_pointwise_attention.output_linear.weight.normal_(mean=0.0, std=0.02)
        sensitive.template_pointwise_attention.output_linear.bias.normal_(mean=0.0, std=0.02)
    return sensitive


def test_template_stack_identity_without_templates(module, batch):
    module.eval()
    with torch.no_grad():
        m_out, z_out = module(batch["m"], batch["z"], template_angle_feat=None, template_pair_feat=None)
    assert_close(m_out, batch["m"], name="identity_m")
    assert_close(z_out, batch["z"], name="identity_z")


def test_template_stack_angle_path_appends_rows_and_respects_mask(module, batch):
    module.eval()
    with torch.no_grad():
        m_out, z_out = module(
            batch["m"],
            batch["z"],
            template_angle_feat=batch["template_angle_feat"],
            template_pair_feat=None,
            template_mask=batch["template_mask"],
        )

    B, N_msa, L, c_m = batch["m"].shape
    T = batch["template_angle_feat"].shape[1]
    assert_shape(m_out, (B, N_msa + T, L, c_m), "m_out")
    assert_close(m_out[:, :N_msa], batch["m"], name="msa_prefix_preserved")
    assert_close(z_out, batch["z"], name="z_unchanged_without_pair_templates")

    masked = (batch["template_mask"] == 0).unsqueeze(-1)
    appended = m_out[:, N_msa:]
    assert_close(
        appended.masked_select(masked),
        torch.zeros_like(appended.masked_select(masked)),
        atol=1e-6,
        rtol=1e-6,
        name="masked_template_angle_rows_zero",
    )


def test_template_stack_pair_path_output_finite_and_deterministic(module, batch):
    module.eval()
    with torch.no_grad():
        m_first, z_first = module(
            batch["m"],
            batch["z"],
            template_angle_feat=None,
            template_pair_feat=batch["template_pair_feat"],
            template_mask=batch["template_mask"],
        )
        m_second, z_second = module(
            batch["m"],
            batch["z"],
            template_angle_feat=None,
            template_pair_feat=batch["template_pair_feat"],
            template_mask=batch["template_mask"],
        )

    assert_close(m_first, batch["m"], name="m_passthrough_pair_only")
    assert_finite_tensor(z_first, "z_first")
    assert_close(z_first, z_second, name="deterministic_eval")


def test_template_stack_pair_mask_all_ones_matches_unmasked(module, batch):
    module = _build_sensitive_pair_module(module).eval()
    all_ones = torch.ones_like(batch["template_mask"])

    with torch.no_grad():
        _, z_masked = module(
            batch["m"],
            batch["z"],
            template_angle_feat=None,
            template_pair_feat=batch["template_pair_feat"],
            template_mask=all_ones,
        )
        _, z_unmasked = module(
            batch["m"],
            batch["z"],
            template_angle_feat=None,
            template_pair_feat=batch["template_pair_feat"],
            template_mask=None,
        )

    assert_close(z_masked, z_unmasked, atol=1e-5, rtol=1e-5, name="all_ones_matches_unmasked")


def test_template_stack_masked_pair_perturbation_has_no_effect(module, batch):
    module = _build_sensitive_pair_module(module).eval()
    template_pair_feat = batch["template_pair_feat"].clone()
    masked_index = (batch["template_mask"] == 0).nonzero(as_tuple=False)
    assert masked_index.numel() > 0, "Expected masked positions in template_mask"

    for row in masked_index[:16]:
        b, t, l = row.tolist()
        template_pair_feat[b, t, l, :, :] += 10.0
        template_pair_feat[b, t, :, l, :] += 10.0

    with torch.no_grad():
        _, z_base = module(
            batch["m"],
            batch["z"],
            template_angle_feat=None,
            template_pair_feat=batch["template_pair_feat"],
            template_mask=batch["template_mask"],
        )
        _, z_perturbed = module(
            batch["m"],
            batch["z"],
            template_angle_feat=None,
            template_pair_feat=template_pair_feat,
            template_mask=batch["template_mask"],
        )

    assert_close(z_base, z_perturbed, atol=1e-5, rtol=1e-5, name="masked_pair_perturbation_no_effect")


def test_template_stack_pair_path_sensitive_when_valid_inputs_change(module, batch):
    module = _build_sensitive_pair_module(module).eval()
    template_pair_feat = batch["template_pair_feat"].clone()
    valid_index = (batch["template_mask"] > 0).nonzero(as_tuple=False)
    b, t, l = valid_index[0].tolist()
    template_pair_feat[b, t, l, :, :] += 1.0
    template_pair_feat[b, t, :, l, :] += 1.0

    with torch.no_grad():
        _, z_base = module(
            batch["m"],
            batch["z"],
            template_angle_feat=None,
            template_pair_feat=batch["template_pair_feat"],
            template_mask=batch["template_mask"],
        )
        _, z_changed = module(
            batch["m"],
            batch["z"],
            template_angle_feat=None,
            template_pair_feat=template_pair_feat,
            template_mask=batch["template_mask"],
        )

    assert_not_close(z_base, z_changed, atol=1e-7, rtol=1e-6, name="valid_pair_input_sensitivity")


def test_template_stack_helper_masks_behave_as_expected(batch):
    expanded = normalize_template_mask(batch["template_mask"][:, :, 0], length=batch["template_mask"].shape[-1])
    assert_shape(expanded, batch["template_mask"].shape, "expanded_mask")

    msa_mask = torch.ones(
        batch["m"].shape[0],
        batch["m"].shape[1],
        batch["m"].shape[2],
        device=batch["m"].device,
        dtype=batch["m"].dtype,
    )
    combined = augment_msa_mask_with_template_mask(msa_mask, batch["template_mask"])
    assert_shape(
        combined,
        (batch["m"].shape[0], batch["m"].shape[1] + batch["template_mask"].shape[1], batch["m"].shape[2]),
        "combined_msa_mask",
    )
    assert_close(combined[:, : batch["m"].shape[1]], msa_mask, name="msa_mask_prefix")


def test_template_stack_gradient_flow(module, batch):
    module = _build_sensitive_pair_module(module)
    module.train()
    m_out, z_out = module(
        batch["m"],
        batch["z"],
        template_angle_feat=batch["template_angle_feat"],
        template_pair_feat=batch["template_pair_feat"],
        template_mask=batch["template_mask"],
    )
    loss = z_out.mean() + m_out.mean()
    assert_scalar_finite(loss, "loss")
    loss.backward()

    got_grad = False
    for name, parameter in module.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, f"{name} did not receive gradient"
            assert torch.isfinite(parameter.grad).all(), f"{name} gradient has NaN/Inf"
            got_grad = True
    assert got_grad, "No parameter got gradients"


def run_template_stack_test_suite(device="cpu"):
    batch = make_fake_template_batch(device=device)
    module = _build_module(
        c_m=batch["m"].shape[-1],
        c_z=batch["z"].shape[-1],
        angle_dim=batch["template_angle_feat"].shape[-1],
        pair_dim=batch["template_pair_feat"].shape[-1],
        device=device,
    )

    tests = [
        ("identity_without_templates", lambda: test_template_stack_identity_without_templates(module, batch)),
        ("angle_path_appends_rows_and_respects_mask", lambda: test_template_stack_angle_path_appends_rows_and_respects_mask(module, batch)),
        ("pair_path_output_finite_and_deterministic", lambda: test_template_stack_pair_path_output_finite_and_deterministic(module, batch)),
        ("pair_mask_all_ones_matches_unmasked", lambda: test_template_stack_pair_mask_all_ones_matches_unmasked(module, batch)),
        ("masked_pair_perturbation_has_no_effect", lambda: test_template_stack_masked_pair_perturbation_has_no_effect(module, batch)),
        ("pair_path_sensitive_when_valid_inputs_change", lambda: test_template_stack_pair_path_sensitive_when_valid_inputs_change(module, batch)),
        ("helper_masks_behave_as_expected", lambda: test_template_stack_helper_masks_behave_as_expected(batch)),
        ("gradient_flow", lambda: test_template_stack_gradient_flow(module, batch)),
    ]

    results = [run_test_silent(name, fn) for name, fn in tests]
    finalize_test_results(results, suite_name="TemplateStack")


def test_template_stack_suite():
    run_template_stack_test_suite(device="cpu")
