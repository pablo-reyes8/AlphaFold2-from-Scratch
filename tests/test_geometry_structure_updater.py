"""Validate geometry utilities and structure-updater modules with silent pytest-style assertions."""

from __future__ import annotations

import torch

from model.quaternion_to_matrix import compose_frames, quaternion_to_rotation_matrix
from model.structure_transition import BackboneUpdate, StructureTransition

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

def test_quaternion_identity_gives_identity():
    q = torch.zeros(4)
    q[0] = 1.0  # [1,0,0,0]
    R = quaternion_to_rotation_matrix(q)
    I = torch.eye(3)
    assert_close(R, I, atol=1e-6, msg="Identity quaternion did not give identity matrix")

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

def test_rotation_matrix_orthogonality_and_det():
    q = random_unit_quaternions(4, 11)
    R = quaternion_to_rotation_matrix(q)   # [4,11,3,3]

    RtR = torch.matmul(R.transpose(-1, -2), R)
    I = torch.eye(3, device=R.device, dtype=R.dtype).expand_as(RtR)
    assert_close(RtR, I, atol=1e-5, msg="R^T R != I")

    det = torch.det(R)
    ones = torch.ones_like(det)
    assert_close(det, ones, atol=1e-5, msg="det(R) != 1")

def test_quaternion_sign_invariance():
    q = random_unit_quaternions(3, 8)
    R1 = quaternion_to_rotation_matrix(q)
    R2 = quaternion_to_rotation_matrix(-q)
    assert_close(R1, R2, atol=1e-6, msg="q and -q should define same rotation")

def test_compose_frames_identity_case():
    B, L = 2, 6
    R = torch.eye(3).view(1,1,3,3).repeat(B, L, 1, 1)
    t = torch.zeros(B, L, 3)

    dR = torch.eye(3).view(1,1,3,3).repeat(B, L, 1, 1)
    dt = torch.zeros(B, L, 3)

    R_new, t_new = compose_frames(R, t, dR, dt)

    assert_close(R_new, R, atol=1e-6, msg="compose_frames identity R failed")
    assert_close(t_new, t, atol=1e-6, msg="compose_frames identity t failed")

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

def test_structure_transition_shape_and_mask():
    B, L, c_s = 2, 9, 256
    s = torch.randn(B, L, c_s)
    mask = torch.ones(B, L)
    mask[0, -3:] = 0.0

    mod = StructureTransition(c_s=c_s, dropout=0.0)
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

def assert_scalar_finite(x, msg=""):
    assert torch.is_tensor(x), f"{msg} | expected tensor scalar"
    assert x.ndim == 0, f"{msg} | expected scalar tensor, got shape {tuple(x.shape)}"
    assert torch.isfinite(x), f"{msg} | scalar is not finite"

def test_quaternion_batch_consistency():
    q = random_unit_quaternions(2, 3, 5)   # [2,3,5,4]
    R = quaternion_to_rotation_matrix(q)

    for i in range(2):
        for j in range(3):
            for k in range(5):
                R_ijk = quaternion_to_rotation_matrix(q[i, j, k])
                assert_close(
                    R[i, j, k],
                    R_ijk,
                    atol=1e-6,
                    msg=f"Batch consistency failed at {(i,j,k)}"
                )

def test_quaternion_rotation_preserves_vector_norm():
    q = random_unit_quaternions(4, 7)
    R = quaternion_to_rotation_matrix(q)   # [4,7,3,3]
    x = torch.randn(4, 7, 3)

    x_rot = torch.matmul(R, x.unsqueeze(-1)).squeeze(-1)

    x_norm = torch.linalg.norm(x, dim=-1)
    x_rot_norm = torch.linalg.norm(x_rot, dim=-1)

    assert_close(
        x_norm,
        x_rot_norm,
        atol=1e-5,
        msg="Rotation did not preserve vector norm"
    )

def test_quaternion_gradients_finite():
    q_raw = torch.randn(3, 4, requires_grad=True)
    q = q_raw / torch.linalg.norm(q_raw, dim=-1, keepdim=True).clamp_min(1e-8)

    R = quaternion_to_rotation_matrix(q)
    loss = (R ** 2).mean()

    assert_scalar_finite(loss, msg="Quaternion loss not finite")
    loss.backward()

    assert q_raw.grad is not None, "Quaternion raw input got no gradient"
    assert torch.isfinite(q_raw.grad).all(), "Quaternion raw gradient has NaN/Inf"

def test_compose_frames_point_consistency():
    """
    Check that composing frames is geometrically consistent on points:
      x_global_new = R_new x + t_new
                   = R (dR x + dt) + t
    """
    B, L = 2, 5
    q1 = random_unit_quaternions(B, L)
    q2 = random_unit_quaternions(B, L)

    R = quaternion_to_rotation_matrix(q1)
    dR = quaternion_to_rotation_matrix(q2)
    t = torch.randn(B, L, 3)
    dt = torch.randn(B, L, 3)

    x_local = torch.randn(B, L, 3)

    R_new, t_new = compose_frames(R, t, dR, dt)

    lhs = torch.matmul(R_new, x_local.unsqueeze(-1)).squeeze(-1) + t_new

    inner = torch.matmul(dR, x_local.unsqueeze(-1)).squeeze(-1) + dt
    rhs = torch.matmul(R, inner.unsqueeze(-1)).squeeze(-1) + t

    assert_close(lhs, rhs, atol=1e-5, msg="compose_frames point consistency failed")

def test_structure_transition_deterministic_eval():
    B, L, c_s = 2, 9, 256
    s = torch.randn(B, L, c_s)
    mask = torch.ones(B, L)

    mod = StructureTransition(c_s=c_s, dropout=0.1)
    mod.eval()

    with torch.no_grad():
        out1 = mod(s, mask=mask)
        out2 = mod(s, mask=mask)

    assert_close(out1, out2, atol=1e-6, msg="StructureTransition eval is not deterministic")

def test_structure_transition_zero_init_outputs_zero_without_mask():
    """
    Because lin3 is zero-initialized, the whole module should output exactly zero
    at initialization, regardless of s, as long as mask=None.
    """
    B, L, c_s = 2, 11, 256
    s = torch.randn(B, L, c_s)

    mod = StructureTransition(c_s=c_s, dropout=0.0)
    mod.eval()

    out = mod(s, mask=None)

    assert_close(
        out,
        torch.zeros_like(out),
        atol=1e-7,
        msg="StructureTransition zero-init should produce zero output"
    )

def test_structure_transition_all_zero_mask_gives_zero():
    B, L, c_s = 2, 9, 256
    s = torch.randn(B, L, c_s)
    mask = torch.zeros(B, L)

    mod = StructureTransition(c_s=c_s, dropout=0.0)
    mod.eval()

    out = mod(s, mask=mask)

    assert_close(
        out,
        torch.zeros_like(out),
        atol=1e-7,
        msg="StructureTransition all-zero mask did not give zero output"
    )

def test_structure_transition_gradients_finite():
    B, L, c_s = 2, 8, 256
    s = torch.randn(B, L, c_s)
    mask = torch.ones(B, L)

    mod = StructureTransition(c_s=c_s, dropout=0.0)
    mod.train()

    out = mod(s, mask=mask)
    loss = (out ** 2).mean()

    assert_scalar_finite(loss, msg="StructureTransition loss not finite")
    loss.backward()

    got_grad = False
    for name, p in mod.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"{name} got no gradient"
            assert torch.isfinite(p.grad).all(), f"{name} gradient has NaN/Inf"
            got_grad = True

    assert got_grad, "No StructureTransition parameter received gradient"

def test_backbone_update_deterministic_eval():
    B, L, c_s = 2, 10, 256
    s = torch.randn(B, L, c_s)
    mask = torch.ones(B, L)

    mod = BackboneUpdate(c_s=c_s)
    mod.eval()

    with torch.no_grad():
        dR1, dt1 = mod(s, mask=mask)
        dR2, dt2 = mod(s, mask=mask)

    assert_close(dR1, dR2, atol=1e-6, msg="BackboneUpdate dR eval not deterministic")
    assert_close(dt1, dt2, atol=1e-6, msg="BackboneUpdate dt eval not deterministic")

def test_backbone_update_all_zero_mask_gives_identity_and_zero():
    B, L, c_s = 2, 7, 256
    s = torch.randn(B, L, c_s)
    mask = torch.zeros(B, L)

    mod = BackboneUpdate(c_s=c_s)
    mod.eval()

    dR, dt = mod(s, mask=mask)

    I = torch.eye(3, device=s.device, dtype=s.dtype).view(1, 1, 3, 3).expand(B, L, 3, 3)
    Z = torch.zeros(B, L, 3, device=s.device, dtype=s.dtype)

    assert_close(dR, I, atol=1e-6, msg="All-zero mask should give identity dR")
    assert_close(dt, Z, atol=1e-6, msg="All-zero mask should give zero dt")

def test_backbone_update_input_sensitivity_after_perturbing_weights():
    """
    With exact zero-init, output is constant by design.
    So we perturb weights slightly and verify output responds to s.
    """
    B, L, c_s = 2, 9, 256
    s1 = torch.randn(B, L, c_s)
    s2 = s1.clone()
    s2[:, :3, :] += 0.5

    mod = BackboneUpdate(c_s=c_s)
    with torch.no_grad():
        mod.linear.weight.normal_(mean=0.0, std=0.02)
        mod.linear.bias.normal_(mean=0.0, std=0.02)

    mod.eval()
    dR1, dt1 = mod(s1)
    dR2, dt2 = mod(s2)

    diff_R = (dR1 - dR2).abs().max().item()
    diff_t = (dt1 - dt2).abs().max().item()

    assert diff_R > 1e-8 or diff_t > 1e-8, (
        f"BackboneUpdate output did not change enough after changing input; "
        f"diff_R={diff_R}, diff_t={diff_t}"
    )

def test_backbone_update_gradients_finite():
    B, L, c_s = 2, 8, 256
    s = torch.randn(B, L, c_s)
    mask = torch.ones(B, L)

    mod = BackboneUpdate(c_s=c_s)
    with torch.no_grad():
        mod.linear.weight.normal_(mean=0.0, std=0.02)
        mod.linear.bias.normal_(mean=0.0, std=0.02)

    mod.train()
    dR, dt = mod(s, mask=mask)
    loss = dR.mean() + dt.mean()

    assert_scalar_finite(loss, msg="BackboneUpdate loss not finite")
    loss.backward()

    got_grad = False
    for name, p in mod.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"{name} got no gradient"
            assert torch.isfinite(p.grad).all(), f"{name} gradient has NaN/Inf"
            got_grad = True

    assert got_grad, "No BackboneUpdate parameter received gradient"

def test_backbone_update_zero_init_matches_identity_compose():
    """
    Since zero-init BackboneUpdate returns dR=I, dt=0,
    composing any frame with it should leave the frame unchanged.
    """
    B, L, c_s = 2, 6, 256
    s = torch.randn(B, L, c_s)

    q = random_unit_quaternions(B, L)
    R = quaternion_to_rotation_matrix(q)
    t = torch.randn(B, L, 3)

    mod = BackboneUpdate(c_s=c_s)
    mod.eval()

    dR, dt = mod(s, mask=None)
    R_new, t_new = compose_frames(R, t, dR, dt)

    assert_close(R_new, R, atol=1e-6, msg="Zero-init BackboneUpdate should preserve R under compose")
    assert_close(t_new, t, atol=1e-6, msg="Zero-init BackboneUpdate should preserve t under compose")

def run_all_geometry_tests():

    # quaternion_to_rotation_matrix
    test_quaternion_to_rotation_matrix_shapes()
    test_quaternion_identity_gives_identity()
    test_quaternion_known_180deg_x()
    test_rotation_matrix_orthogonality_and_det()
    test_quaternion_sign_invariance()
    test_quaternion_batch_consistency()
    test_quaternion_rotation_preserves_vector_norm()
    test_quaternion_gradients_finite()

    # compose_frames
    test_compose_frames_identity_case()
    test_compose_frames_matches_manual_formula()
    test_compose_frames_preserves_valid_rotations()
    test_compose_frames_point_consistency()

    # StructureTransition
    test_structure_transition_shape_and_mask()
    test_structure_transition_deterministic_eval()
    test_structure_transition_zero_init_outputs_zero_without_mask()
    test_structure_transition_all_zero_mask_gives_zero()
    test_structure_transition_gradients_finite()

    # BackboneUpdate
    test_backbone_update_zero_init_behavior()
    test_backbone_update_mask_behavior()
    test_backbone_update_rotation_validity()
    test_backbone_update_deterministic_eval()
    test_backbone_update_all_zero_mask_gives_identity_and_zero()
    test_backbone_update_input_sensitivity_after_perturbing_weights()
    test_backbone_update_gradients_finite()
    test_backbone_update_zero_init_matches_identity_compose()
