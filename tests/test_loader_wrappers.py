"""Validate dataloader wrapper helpers on the bundled showcase subset."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("Bio")

from data.dataloaders import FoldbenchProteinDataset
from data.loader_wrappers import build_train_eval_protein_dataloaders, resolve_train_eval_indices


ROOT_DIR = Path(__file__).resolve().parents[1]
SHOWCASE_MANIFEST = ROOT_DIR / "data" / "showcase_manifest.csv"


def test_resolve_train_eval_indices_is_deterministic_when_shuffled():
    first = resolve_train_eval_indices(6, eval_size=2, split_seed=7, shuffle_before_split=True)
    second = resolve_train_eval_indices(6, eval_size=2, split_seed=7, shuffle_before_split=True)
    assert first == second
    assert len(first[0]) == 4
    assert len(first[1]) == 2
    assert set(first[0]).isdisjoint(first[1])


def test_build_train_eval_protein_dataloaders_splits_showcase_subset():
    dataset = FoldbenchProteinDataset(
        manifest_csv=str(SHOWCASE_MANIFEST),
        max_msa_seqs=8,
        max_extra_msa_seqs=16,
        max_templates=4,
        verbose=False,
    )

    split = build_train_eval_protein_dataloaders(
        dataset,
        batch_size=2,
        shuffle=False,
        eval_size=1,
        eval_shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    assert len(split.train_indices) == 1
    assert len(split.eval_indices) == 1
    assert set(split.train_indices).isdisjoint(split.eval_indices)

    train_batch = next(iter(split.train_loader))
    eval_batch = next(iter(split.eval_loader))

    assert train_batch["seq_tokens"].shape[0] == 1
    assert eval_batch["seq_tokens"].shape[0] == 1
    assert set(train_batch["id"]).isdisjoint(eval_batch["id"])