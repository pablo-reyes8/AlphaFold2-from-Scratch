import torch
import pytest

from model.alphafold2_full_loss import *

def move_batch_to_device(batch, device: str):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


# =========================================================
# Helpers
# =========================================================
def _get_real_batch(loader, device="cpu"):
    batch = next(iter(loader))
    batch = move_batch_to_device(batch, device)
    return batch


def _assert_scalar_finite(x, name="tensor"):
    assert torch.is_tensor(x), f"{name} must be a tensor"
    assert x.ndim == 0, f"{name} must be scalar, got shape {tuple(x.shape)}"
    assert torch.isfinite(x), f"{name} is not finite"


def _make_fake_loss_outputs(batch, device="cpu", num_dist_bins=64, num_plddt_bins=50):
    """
    Construye outputs fake pero consistentes con el batch real,
    pensados para probar AlphaFoldLoss sin depender del forward real.
    """
    B, L = batch["seq_tokens"].shape

    out = {
        "R": torch.eye(3, device=device).view(1, 1, 3, 3).repeat(B, L, 1, 1),
        "t": batch["coords_ca"].clone(),  # pred fácil: usar CA real
        "distogram_logits": torch.randn(B, L, L, num_dist_bins, device=device),
        "plddt_logits": torch.randn(B, L, num_plddt_bins, device=device),
        "torsions": batch["torsion_true"].clone(),  # predicción perfecta
    }
    return out


# =========================================================
# Fixtures
# =========================================================
@pytest.fixture
def device():
    return "cpu"


@pytest.fixture
def real_batch(loader, device):
    torch.manual_seed(11)
    return _get_real_batch(loader, device=device)


# =========================================================
# Tests: individual losses on real batch
# =========================================================
def test_plddt_loss_real_batch_smoke(real_batch, device):
    """
    Evalúa PlddtLoss directamente sobre un batch real.
    No depende del forward del modelo.
    """
    batch = real_batch
    B, L, _ = batch["coords_ca"].shape
    num_bins = 50

    x_true = batch["coords_ca"]
    x_pred = x_true.clone()  # caso perfecto-ish geométricamente
    mask = batch["valid_res_mask"]

    logits = torch.randn(B, L, num_bins, device=device)

    loss_fn = PlddtLoss(
        num_bins=num_bins,
        inclusion_radius=15.0,
    )
    loss = loss_fn(logits, x_pred, x_true, mask=mask)

    print(f"\nPlddtLoss (real batch): {loss.item():.6f}")
    _assert_scalar_finite(loss, "plddt_loss")


def test_torsion_loss_real_batch_perfect_and_perturbed(real_batch):
    """
    Evalúa TorsionLoss en dos escenarios:
    1) predicción perfecta -> pérdida ~ 0
    2) predicción perturbada -> pérdida mayor
    """
    batch = real_batch
    torsion_true = batch["torsion_true"]   # [B, L, T, 2]
    torsion_mask = batch["torsion_mask"]   # [B, L, T]

    loss_fn = TorsionLoss()

    # caso perfecto
    torsion_pred = torsion_true.clone()
    loss = loss_fn(torsion_pred, torsion_true, torsion_mask)

    print(f"\nTorsionLoss perfect prediction (real batch): {loss.item():.10f}")
    _assert_scalar_finite(loss, "torsion_loss_perfect")
    assert loss.item() < 1e-7, "TorsionLoss should be ~0 for perfect prediction"

    # caso perturbado
    torsion_pred_pert = torsion_true + 0.1 * torch.randn_like(torsion_true)
    torsion_pred_pert = torsion_pred_pert / torch.linalg.norm(
        torsion_pred_pert, dim=-1, keepdim=True
    ).clamp_min(1e-8)

    loss_pert = loss_fn(torsion_pred_pert, torsion_true, torsion_mask)

    print(f"TorsionLoss perturbed (real batch): {loss_pert.item():.10f}")
    _assert_scalar_finite(loss_pert, "torsion_loss_perturbed")
    assert loss_pert.item() > loss.item(), (
        "Perturbed torsion loss should be larger than perfect torsion loss"
    )


# =========================================================
# Test: full AlphaFoldLoss orchestrator on real batch
# =========================================================
def test_alphafold_loss_orchestrator_real_batch(real_batch, device):
    """
    Prueba AlphaFoldLoss completo sobre un batch real, pero con outputs fake
    consistentes. Esto valida el contrato del criterion sin depender del model forward.
    """
    batch = real_batch

    num_dist_bins = 64
    num_plddt_bins = 50
    T = batch["torsion_true"].shape[2]

    out = _make_fake_loss_outputs(
        batch=batch,
        device=device,
        num_dist_bins=num_dist_bins,
        num_plddt_bins=num_plddt_bins,
    )

    loss_fn = AlphaFoldLoss(
        fape_length_scale=10.0,
        fape_clamp_distance=10.0,
        dist_num_bins=num_dist_bins,
        dist_min_bin=2.0,
        dist_max_bin=22.0,
        plddt_num_bins=num_plddt_bins,
        plddt_inclusion_radius=15.0,
        w_fape=0.5,
        w_dist=0.3,
        w_plddt=0.01,
        w_torsion=0.01,
    )

    losses = loss_fn(out, batch)

    # prints útiles de debugging
    print("")
    for k, v in losses.items():
        if torch.is_tensor(v) and v.ndim == 0:
            print(f"{k}: {v.item():.8f}")
        else:
            print(f"{k}: {v}")

    # keys esperadas
    expected_keys = {"loss", "fape_loss", "dist_loss", "plddt_loss", "torsion_loss"}
    assert expected_keys.issubset(losses.keys()), (
        f"Missing keys in losses. Expected at least {expected_keys}, got {set(losses.keys())}"
    )

    # todos escalares y finitos
    for name in expected_keys:
        _assert_scalar_finite(losses[name], name)

    # torsiones perfectas -> pérdida ~0
    assert losses["torsion_loss"].item() < 1e-7, (
        "torsion_loss should be ~0 for perfect torsion prediction"
    )

    # chequeo de composición ponderada
    expected_total = (
        loss_fn.w_fape * losses["fape_loss"]
        + loss_fn.w_dist * losses["dist_loss"]
        + loss_fn.w_plddt * losses["plddt_loss"]
        + loss_fn.w_torsion * losses["torsion_loss"]
    )
    assert torch.allclose(losses["loss"], expected_total, atol=1e-6), (
        "Total loss does not match weighted sum of components"
    )

    # chequeos extra de shape
    B, L = batch["seq_tokens"].shape
    assert out["R"].shape == (B, L, 3, 3)
    assert out["t"].shape == (B, L, 3)
    assert out["distogram_logits"].shape == (B, L, L, num_dist_bins)
    assert out["plddt_logits"].shape == (B, L, num_plddt_bins)
    assert out["torsions"].shape == batch["torsion_true"].shape
    assert T == 3, f"Expected 3 torsions, got {T}"


# =========================================================
# Optional stricter test
# =========================================================
def test_alphafold_loss_orchestrator_real_batch_uses_backbone_coords_if_present(
    real_batch, device
):
    """
    Test opcional:
    si tu AlphaFoldLoss usa backbone_coords cuando está presente en `out`,
    este test verifica al menos que esa ruta corre y produce loss finita.
    """
    batch = real_batch
    B, L = batch["seq_tokens"].shape

    out = {
        "R": torch.eye(3, device=device).view(1, 1, 3, 3).repeat(B, L, 1, 1),
        "t": batch["coords_ca"].clone(),
        "backbone_coords": batch["coords_backbone"].clone() if "coords_backbone" in batch else None,
        "distogram_logits": torch.randn(B, L, L, 64, device=device),
        "plddt_logits": torch.randn(B, L, 50, device=device),
        "torsions": batch["torsion_true"].clone(),
    }

    if out["backbone_coords"] is None:
        pytest.skip("Batch does not contain `coords_backbone`; skipping backbone_coords path test.")

    loss_fn = AlphaFoldLoss(
        fape_length_scale=10.0,
        fape_clamp_distance=10.0,
        dist_num_bins=64,
        dist_min_bin=2.0,
        dist_max_bin=22.0,
        plddt_num_bins=50,
        plddt_inclusion_radius=15.0,
        w_fape=0.5,
        w_dist=0.3,
        w_plddt=0.01,
        w_torsion=0.01,
    )

    losses = loss_fn(out, batch)
    _assert_scalar_finite(losses["loss"], "loss")
    assert losses["torsion_loss"].item() < 1e-7