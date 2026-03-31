from __future__ import annotations

import torch

from model.losses.distogram_loss import DistogramLoss
from model.losses.fape_loss import FAPELoss
from model.losses.pLDDT_loss import PlddtLoss
from model.losses.torsion_loss import TorsionLoss


def test_fape_loss_perfect_prediction_is_near_zero():
    batch_size, length = 2, 10
    rotation_true = torch.eye(3).view(1, 1, 3, 3).repeat(batch_size, length, 1, 1)
    translation_true = torch.randn(batch_size, length, 3)
    coords_true = translation_true.clone()

    mask = torch.ones(batch_size, length)
    mask[0, -2:] = 0.0

    loss_fn = FAPELoss(length_scale=10.0, clamp_distance=10.0)
    loss = loss_fn(
        rotation_true,
        translation_true,
        coords_true,
        rotation_true,
        translation_true,
        coords_true,
        mask=mask,
    )

    assert torch.isfinite(loss)
    assert loss.item() < 2e-5


def test_fape_loss_increases_with_perturbation():
    batch_size, length = 2, 10
    rotation_true = torch.eye(3).view(1, 1, 3, 3).repeat(batch_size, length, 1, 1)
    translation_true = torch.randn(batch_size, length, 3)
    coords_true = translation_true.clone()
    mask = torch.ones(batch_size, length)

    loss_fn = FAPELoss(length_scale=10.0, clamp_distance=10.0)
    perfect = loss_fn(
        rotation_true,
        translation_true,
        coords_true,
        rotation_true,
        translation_true,
        coords_true,
        mask=mask,
    )

    translation_pred = translation_true + 0.5 * torch.randn(batch_size, length, 3)
    coords_pred = translation_pred.clone()
    perturbed = loss_fn(
        rotation_true,
        translation_pred,
        coords_pred,
        rotation_true,
        translation_true,
        coords_true,
        mask=mask,
    )

    assert perturbed.item() > perfect.item()


def test_distogram_loss_returns_finite_scalar():
    logits = torch.randn(2, 12, 12, 64)
    coords_true = torch.randn(2, 12, 3)
    mask = torch.ones(2, 12)
    mask[1, -3:] = 0.0

    loss = DistogramLoss(num_bins=64, min_bin=2.0, max_bin=22.0)(
        logits,
        coords_true,
        mask=mask,
    )

    assert torch.isfinite(loss)
    assert loss.ndim == 0


def test_plddt_loss_returns_finite_scalar():
    batch_size, length, num_bins = 2, 8, 50
    coords_true = torch.randn(batch_size, length, 3)
    coords_pred = coords_true + 0.1 * torch.randn_like(coords_true)
    logits = torch.randn(batch_size, length, num_bins)
    mask = torch.ones(batch_size, length)

    loss = PlddtLoss(num_bins=num_bins, inclusion_radius=15.0)(
        logits,
        coords_pred,
        coords_true,
        mask=mask,
    )

    assert torch.isfinite(loss)
    assert loss.ndim == 0


def test_torsion_loss_is_zero_for_identical_normalized_vectors():
    torsion_true = torch.randn(2, 6, 3, 2)
    torsion_true = torsion_true / torch.linalg.norm(torsion_true, dim=-1, keepdim=True).clamp_min(1e-8)
    torsion_mask = torch.ones(2, 6, 3)

    loss = TorsionLoss()(torsion_true, torsion_true, torsion_mask)
    assert torch.isfinite(loss)
    assert loss.item() < 1e-7
