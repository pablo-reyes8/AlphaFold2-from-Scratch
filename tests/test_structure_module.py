"""Targeted tests for the AlphaFold-style structure module wrapper."""

from __future__ import annotations

import copy

import torch

from model.structure_block import StructureModule


def assert_close(x, y, atol=1e-5, rtol=1e-5, msg=""):
    if not torch.allclose(x, y, atol=atol, rtol=rtol):
        max_err = (x - y).abs().max().item()
        raise AssertionError(f"{msg} | max_err={max_err}")


def test_structure_module_optional_outputs_are_backward_compatible():
    batch_size, length, c_s, c_z = 2, 11, 32, 16
    s = torch.randn(batch_size, length, c_s)
    z = torch.randn(batch_size, length, length, c_z)
    mask = torch.ones(batch_size, length)

    coords_ca = torch.randn(batch_size, length, 3)
    coords_n = coords_ca + torch.tensor([-1.2, 0.4, 0.1], dtype=torch.float32)
    coords_c = coords_ca + torch.tensor([1.3, 0.5, -0.1], dtype=torch.float32)

    module = StructureModule(
        c_s=c_s,
        c_z=c_z,
        num_blocks=3,
        ipa_heads=4,
        ipa_scalar_dim=8,
        dropout=0.0,
    )
    module.eval()

    with torch.no_grad():
        s_out, R_out, t_out = module(s, z, mask=mask)
        s_aux, R_aux, t_aux, aux_loss, intermediates = module(
            s,
            z,
            mask=mask,
            coords_n=coords_n,
            coords_ca=coords_ca,
            coords_c=coords_c,
            backbone_mask=mask,
            return_aux=True,
            return_intermediates=True,
        )

    assert s_out.shape == (batch_size, length, c_s)
    assert R_out.shape == (batch_size, length, 3, 3)
    assert t_out.shape == (batch_size, length, 3)
    assert_close(s_out, s_aux, atol=1e-6, rtol=1e-6, msg="s output changed after enabling aux")
    assert_close(R_out, R_aux, atol=1e-6, rtol=1e-6, msg="R output changed after enabling aux")
    assert_close(t_out, t_aux, atol=1e-6, rtol=1e-6, msg="t output changed after enabling aux")
    assert aux_loss.ndim == 0
    assert torch.isfinite(aux_loss)
    assert module.last_aux_loss is not None
    assert torch.isfinite(module.last_aux_loss)
    assert module.last_aux_per_block.shape == (module.num_blocks,)
    assert intermediates["single"].shape == (module.num_blocks, batch_size, length, c_s)
    assert intermediates["R"].shape == (module.num_blocks, batch_size, length, 3, 3)
    assert intermediates["t"].shape == (module.num_blocks, batch_size, length, 3)


def test_structure_module_stop_rotation_gradients_cuts_inter_block_rotation_path():
    batch_size, length, c_s, c_z = 1, 4, 16, 8
    s = torch.randn(batch_size, length, c_s)
    z = torch.randn(batch_size, length, length, c_z)
    mask = torch.ones(batch_size, length)

    base = StructureModule(
        c_s=c_s,
        c_z=c_z,
        num_blocks=2,
        ipa_heads=4,
        ipa_scalar_dim=4,
        dropout=0.0,
        use_block_specific_params=True,
        stop_rotation_gradients=False,
    )
    with_stopgrad = copy.deepcopy(base)
    with_stopgrad.stop_rotation_gradients = True

    for module in (base, with_stopgrad):
        with torch.no_grad():
            for head in module.translation_heads:
                head.weight.zero_()
                head.bias.zero_()
            for update in module.backbone_updates:
                update.linear.weight.zero_()
                update.linear.bias.zero_()

            module.backbone_updates[0].linear.bias[3:] = torch.tensor([0.4, -0.2, 0.3])
            module.translation_heads[1].bias[:] = torch.tensor([1.0, 0.5, -0.25])

    _, _, t_no_stop = base(s, z, mask=mask)
    loss_no_stop = t_no_stop.sum()
    loss_no_stop.backward()

    _, _, t_stop = with_stopgrad(s, z, mask=mask)
    loss_stop = t_stop.sum()
    loss_stop.backward()

    grad_no_stop = base.backbone_updates[0].linear.bias.grad[3:].abs().sum()
    stop_grad_tensor = with_stopgrad.backbone_updates[0].linear.bias.grad
    if stop_grad_tensor is None:
        grad_stop = torch.zeros((), dtype=grad_no_stop.dtype)
    else:
        grad_stop = stop_grad_tensor[3:].abs().sum()

    assert torch.isfinite(grad_no_stop)
    assert torch.isfinite(grad_stop)
    assert grad_no_stop.item() > 1e-8, "Expected rotation gradient without stopgrad"
    assert grad_stop.item() < 1e-10, "Rotation gradient should be cut by stopgrad"
