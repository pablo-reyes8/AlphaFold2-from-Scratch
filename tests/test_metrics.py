from __future__ import annotations

import torch

from training.efficent_metrics import compute_structure_metrics


torch.manual_seed(0)


def test_structure_metrics_for_perfect_prediction():
    batch_size, length = 2, 20
    coords_true = torch.randn(batch_size, length, 3)
    mask = torch.ones(batch_size, length)

    metrics = compute_structure_metrics(coords_true, coords_true, mask, align=True)

    assert metrics["rmsd"].item() < 1e-3
    assert metrics["tm_score"].item() > 0.999
    assert metrics["gdt_ts"].item() > 0.999


def test_structure_metrics_degrade_for_noisy_prediction():
    batch_size, length = 2, 20
    coords_true = torch.randn(batch_size, length, 3)
    mask = torch.ones(batch_size, length)

    perfect = compute_structure_metrics(coords_true, coords_true, mask, align=True)
    noisy_coords = coords_true + 0.8 * torch.randn_like(coords_true)
    noisy = compute_structure_metrics(noisy_coords, coords_true, mask, align=True)

    assert noisy["rmsd"].item() > perfect["rmsd"].item()
    assert noisy["tm_score"].item() < perfect["tm_score"].item()
    assert noisy["gdt_ts"].item() < perfect["gdt_ts"].item()
