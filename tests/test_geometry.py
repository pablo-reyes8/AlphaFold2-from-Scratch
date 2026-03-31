from __future__ import annotations

import torch

from model.quaterion_to_matrix import compose_frames, quaternion_to_rotation_matrix
from model.structure_transation import BackboneUpdate, StructureTransition


torch.manual_seed(42)


def assert_close(x, y, atol=1e-5, rtol=1e-5, msg=""):
    if not torch.allclose(x, y, atol=atol, rtol=rtol):
        max_err = (x - y).abs().max().item()
        raise AssertionError(f"{msg} | max_err={max_err}")


def random_unit_quaternions(*shape, device="cpu", dtype=torch.float32):
    q = torch.randn(*shape, 4, device=device, dtype=dtype)
    return q / torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(1e-8)


def test_quaternion_to_rotation_matrix_shapes():
    q = random_unit_quaternions(2, 5, 7)
    rotation = quaternion_to_rotation_matrix(q)
    assert rotation.shape == (2, 5, 7, 3, 3)
    assert torch.isfinite(rotation).all()


def test_quaternion_identity_gives_identity():
    q = torch.tensor([1.0, 0.0, 0.0, 0.0])
    rotation = quaternion_to_rotation_matrix(q)
    assert_close(rotation, torch.eye(3), atol=1e-6)


def test_quaternion_sign_invariance():
    q = random_unit_quaternions(3, 8)
    assert_close(
        quaternion_to_rotation_matrix(q),
        quaternion_to_rotation_matrix(-q),
        atol=1e-6,
    )


def test_rotation_matrix_orthogonality_and_det():
    q = random_unit_quaternions(4, 11)
    rotation = quaternion_to_rotation_matrix(q)
    rt_r = torch.matmul(rotation.transpose(-1, -2), rotation)
    identity = torch.eye(3, device=rotation.device, dtype=rotation.dtype).expand_as(rt_r)
    assert_close(rt_r, identity, atol=1e-5)
    assert_close(torch.det(rotation), torch.ones_like(torch.det(rotation)), atol=1e-5)


def test_compose_frames_matches_manual_formula():
    batch_size, length = 2, 5
    rotation = quaternion_to_rotation_matrix(random_unit_quaternions(batch_size, length))
    delta_rotation = quaternion_to_rotation_matrix(random_unit_quaternions(batch_size, length))
    translation = torch.randn(batch_size, length, 3)
    delta_translation = torch.randn(batch_size, length, 3)

    new_rotation, new_translation = compose_frames(rotation, translation, delta_rotation, delta_translation)

    assert_close(new_rotation, torch.matmul(rotation, delta_rotation), atol=1e-6)
    assert_close(
        new_translation,
        torch.matmul(rotation, delta_translation.unsqueeze(-1)).squeeze(-1) + translation,
        atol=1e-6,
    )


def test_structure_transition_shape_and_mask():
    s = torch.randn(2, 9, 256)
    mask = torch.ones(2, 9)
    mask[0, -3:] = 0.0

    module = StructureTransition(c_s=256, expansion=4, dropout=0.0).eval()
    outputs = module(s, mask=mask)

    assert outputs.shape == (2, 9, 256)
    assert torch.isfinite(outputs).all()
    assert_close(outputs[0, -3:, :], torch.zeros_like(outputs[0, -3:, :]), atol=1e-6)


def test_backbone_update_zero_init_behavior():
    s = torch.randn(2, 10, 256)
    module = BackboneUpdate(c_s=256).eval()
    delta_rotation, delta_translation = module(s, mask=None)

    identity = torch.eye(3).view(1, 1, 3, 3).repeat(2, 10, 1, 1)
    zeros = torch.zeros(2, 10, 3)

    assert delta_rotation.shape == (2, 10, 3, 3)
    assert delta_translation.shape == (2, 10, 3)
    assert_close(delta_rotation, identity, atol=1e-6)
    assert_close(delta_translation, zeros, atol=1e-6)


def test_backbone_update_mask_behavior():
    s = torch.randn(2, 8, 256)
    mask = torch.ones(2, 8)
    mask[0, -2:] = 0.0

    module = BackboneUpdate(c_s=256).eval()
    delta_rotation, delta_translation = module(s, mask=mask)

    assert_close(delta_translation[0, -2:, :], torch.zeros_like(delta_translation[0, -2:, :]), atol=1e-6)
    assert_close(
        delta_rotation[0, -2:, :, :],
        torch.eye(3).view(1, 3, 3).expand(2, 3, 3),
        atol=1e-6,
    )
