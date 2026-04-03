from __future__ import annotations

import torch

from model.quaterion_to_matrix import compose_frames, quaternion_to_rotation_matrix
from model.structure_transation import BackboneUpdate, StructureTransition


# ============================================================
# TESTS for quaternion_to_rotation_matrix / compose_frames
# / StructureTransition / BackboneUpdate
# ============================================================
torch.manual_seed(42)

def assert_close(x, y, atol=1e-5, rtol=1e-5, msg=""):
    if not torch.allclose(x, y, atol=atol, rtol=rtol):
        max_err = (x - y).abs().max().item()
        raise AssertionError(f"{msg} | max_err={max_err}")


def random_unit_quaternions(*shape, device="cpu", dtype=torch.float32):
    q = torch.randn(*shape, 4, device=device, dtype=dtype)
    q = q / torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(1e-8)
    return q


def test_quaternion_to_rotation_matrix_shapes():
    q = random_unit_quaternions(2, 5, 7)
    R = quaternion_to_rotation_matrix(q)
    assert R.shape == (2, 5, 7, 3, 3), f"Shape incorrect: {R.shape}"
    assert torch.isfinite(R).all(), "R has non-finite values"
    print("OK: shape + finiteness")


def test_quaternion_identity_gives_identity():
    q = torch.zeros(4)
    q[0] = 1.0  # [1,0,0,0]
    R = quaternion_to_rotation_matrix(q)
    I = torch.eye(3)
    assert_close(R, I, atol=1e-6, msg="Identity quaternion did not give identity matrix")
    print("OK: identity quaternion -> identity matrix")


def test_quaternion_known_180deg_x():
    # 180° around x-axis:
    # q = [cos(pi/2), sin(pi/2), 0, 0] = [0,1,0,0]
    q = torch.tensor([0.0, 1.0, 0.0, 0.0])
    R = quaternion_to_rotation_matrix(q)

    R_expected = torch.tensor([
        [1.0,  0.0,  0.0],
        [0.0, -1.0,  0.0],
        [0.0,  0.0, -1.0]
    ])

    assert_close(R, R_expected, atol=1e-6, msg="180deg rotation around x failed")
    print("OK: known rotation check")


def test_rotation_matrix_orthogonality_and_det():
    q = random_unit_quaternions(4, 11)
    R = quaternion_to_rotation_matrix(q)   # [4,11,3,3]

    RtR = torch.matmul(R.transpose(-1, -2), R)
    I = torch.eye(3, device=R.device, dtype=R.dtype).expand_as(RtR)
    assert_close(RtR, I, atol=1e-5, msg="R^T R != I")

    det = torch.det(R)
    ones = torch.ones_like(det)
    assert_close(det, ones, atol=1e-5, msg="det(R) != 1")

    print("OK: orthogonality + determinant")


def test_quaternion_sign_invariance():
    q = random_unit_quaternions(3, 8)
    R1 = quaternion_to_rotation_matrix(q)
    R2 = quaternion_to_rotation_matrix(-q)
    assert_close(R1, R2, atol=1e-6, msg="q and -q should define same rotation")
    print("OK: q and -q invariance")


def test_compose_frames_identity_case():
    B, L = 2, 6
    R = torch.eye(3).view(1,1,3,3).repeat(B, L, 1, 1)
    t = torch.zeros(B, L, 3)

    dR = torch.eye(3).view(1,1,3,3).repeat(B, L, 1, 1)
    dt = torch.zeros(B, L, 3)

    R_new, t_new = compose_frames(R, t, dR, dt)

    assert_close(R_new, R, atol=1e-6, msg="compose_frames identity R failed")
    assert_close(t_new, t, atol=1e-6, msg="compose_frames identity t failed")
    print("OK: compose_frames identity case")


def test_compose_frames_matches_manual_formula():
    B, L = 2, 5
    q1 = random_unit_quaternions(B, L)
    q2 = random_unit_quaternions(B, L)

    R = quaternion_to_rotation_matrix(q1)
    dR = quaternion_to_rotation_matrix(q2)

    t = torch.randn(B, L, 3)
    dt = torch.randn(B, L, 3)

    R_new, t_new = compose_frames(R, t, dR, dt)

    R_manual = torch.matmul(R, dR)
    t_manual = torch.matmul(R, dt.unsqueeze(-1)).squeeze(-1) + t

    assert_close(R_new, R_manual, atol=1e-6, msg="R composition mismatch")
    assert_close(t_new, t_manual, atol=1e-6, msg="t composition mismatch")
    print("OK: compose_frames matches manual formula")


def test_compose_frames_preserves_valid_rotations():
    B, L = 2, 5
    q1 = random_unit_quaternions(B, L)
    q2 = random_unit_quaternions(B, L)

    R = quaternion_to_rotation_matrix(q1)
    dR = quaternion_to_rotation_matrix(q2)

    R_new, _ = compose_frames(R, torch.zeros(B, L, 3), dR, torch.zeros(B, L, 3))

    RtR = torch.matmul(R_new.transpose(-1, -2), R_new)
    I = torch.eye(3).view(1,1,3,3).expand_as(RtR)
    det = torch.det(R_new)

    assert_close(RtR, I, atol=1e-5, msg="Composed rotation is not orthogonal")
    assert_close(det, torch.ones_like(det), atol=1e-5, msg="Composed rotation det != 1")
    print("OK: compose_frames preserves SO(3)")


def test_structure_transition_shape_and_mask():
    B, L, c_s = 2, 9, 256
    s = torch.randn(B, L, c_s)
    mask = torch.ones(B, L)
    mask[0, -3:] = 0.0

    mod = StructureTransition(c_s=c_s, expansion=4, dropout=0.0)
    mod.eval()

    out = mod(s, mask=mask)

    assert out.shape == (B, L, c_s), f"Unexpected shape: {out.shape}"
    assert torch.isfinite(out).all(), "Non-finite values in StructureTransition"

    assert_close(
        out[0, -3:, :],
        torch.zeros_like(out[0, -3:, :]),
        atol=1e-6,
        msg="Masked positions in StructureTransition are not zero"
    )
    print("OK: StructureTransition shape + mask")


def test_backbone_update_zero_init_behavior():
    B, L, c_s = 2, 10, 256
    s = torch.randn(B, L, c_s)

    mod = BackboneUpdate(c_s=c_s)
    mod.eval()

    dR, dt = mod(s, mask=None)

    I = torch.eye(3).view(1,1,3,3).repeat(B, L, 1, 1)
    Z = torch.zeros(B, L, 3)

    assert dR.shape == (B, L, 3, 3), f"dR shape incorrect: {dR.shape}"
    assert dt.shape == (B, L, 3), f"dt shape incorrect: {dt.shape}"

    assert_close(dR, I, atol=1e-6, msg="BackboneUpdate should init to identity rotation")
    assert_close(dt, Z, atol=1e-6, msg="BackboneUpdate should init to zero translation")
    print("OK: BackboneUpdate zero-init behavior")


def test_backbone_update_mask_behavior():
    B, L, c_s = 2, 8, 256
    s = torch.randn(B, L, c_s)
    mask = torch.ones(B, L)
    mask[0, -2:] = 0.0

    mod = BackboneUpdate(c_s=c_s)
    mod.eval()

    dR, dt = mod(s, mask=mask)

    I = torch.eye(3, dtype=s.dtype, device=s.device)

    # masked dt should be zero
    assert_close(
        dt[0, -2:, :],
        torch.zeros_like(dt[0, -2:, :]),
        atol=1e-6,
        msg="Masked dt not zero"
    )

    # masked dR should be identity
    assert_close(
        dR[0, -2:, :, :],
        I.view(1, 3, 3).expand(2, 3, 3),
        atol=1e-6,
        msg="Masked dR not identity"
    )

    print("OK: BackboneUpdate mask behavior")


def test_backbone_update_rotation_validity():
    B, L, c_s = 2, 12, 256
    s = torch.randn(B, L, c_s)

    mod = BackboneUpdate(c_s=c_s)

    # break zero-init a little to test general case
    with torch.no_grad():
        mod.linear.weight.normal_(mean=0.0, std=0.02)
        mod.linear.bias.normal_(mean=0.0, std=0.02)

    mod.eval()
    dR, dt = mod(s)

    RtR = torch.matmul(dR.transpose(-1, -2), dR)
    I = torch.eye(3, device=dR.device, dtype=dR.dtype).view(1,1,3,3).expand_as(RtR)
    det = torch.det(dR)

    assert_close(RtR, I, atol=1e-4, msg="dR is not orthogonal")
    assert_close(det, torch.ones_like(det), atol=1e-4, msg="det(dR) != 1")
    assert torch.isfinite(dt).all(), "dt has non-finite values"

    print("OK: BackboneUpdate produces valid rotations")


def run_all_geometry_tests():
    print("Running geometry / frame tests...\n")

    test_quaternion_to_rotation_matrix_shapes()
    test_quaternion_identity_gives_identity()
    test_quaternion_known_180deg_x()
    test_rotation_matrix_orthogonality_and_det()
    test_quaternion_sign_invariance()

    test_compose_frames_identity_case()
    test_compose_frames_matches_manual_formula()
    test_compose_frames_preserves_valid_rotations()

    test_structure_transition_shape_and_mask()

    test_backbone_update_zero_init_behavior()
    test_backbone_update_mask_behavior()
    test_backbone_update_rotation_validity()

    print("\nAll tests passed.")

