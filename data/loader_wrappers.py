"""Convenience wrappers around DataLoader for plain and train/eval workflows.

These helpers preserve the simple DataLoader API used across the repo while
adding a tiny deterministic train/eval split utility for the showcase dataset.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Subset

from data.collate_proteins import collate_proteins


@dataclass(frozen=True)
class TrainEvalDataLoaders:
    """Bundle the train/eval loaders together with the resolved split indices."""

    train_loader: DataLoader
    eval_loader: DataLoader | None
    train_indices: tuple[int, ...]
    eval_indices: tuple[int, ...]


def build_protein_dataloader(
    dataset,
    batch_size: int = 1,
    shuffle: bool = True,
    collate_fn=collate_proteins,
    **dataloader_kwargs,
) -> DataLoader:
    """Build a regular protein dataloader with the repo collate function by default."""
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        collate_fn=collate_fn,
        **dataloader_kwargs,
    )


def resolve_train_eval_indices(
    dataset_size: int,
    *,
    eval_size: int = 1,
    split_seed: int = 42,
    shuffle_before_split: bool = False,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Resolve a deterministic train/eval split over dataset indices."""
    dataset_size = int(dataset_size)
    eval_size = int(eval_size)

    if dataset_size <= 0:
        raise ValueError("dataset_size must be > 0.")
    if eval_size < 0:
        raise ValueError("eval_size must be >= 0.")
    if eval_size >= dataset_size:
        raise ValueError("eval_size must be smaller than dataset_size.")
    if eval_size == 0:
        return tuple(range(dataset_size)), tuple()

    indices = list(range(dataset_size))
    if shuffle_before_split:
        generator = torch.Generator().manual_seed(int(split_seed))
        indices = torch.randperm(dataset_size, generator=generator).tolist()

    return tuple(indices[:-eval_size]), tuple(indices[-eval_size:])


def build_train_eval_protein_dataloaders(
    dataset,
    batch_size: int = 1,
    shuffle: bool = True,
    collate_fn=collate_proteins,
    *,
    eval_size: int = 1,
    eval_shuffle: bool = False,
    split_seed: int = 42,
    shuffle_before_split: bool = False,
    **dataloader_kwargs,
) -> TrainEvalDataLoaders:
    """Build train and eval loaders from one dataset using a deterministic split."""
    train_indices, eval_indices = resolve_train_eval_indices(
        len(dataset),
        eval_size=eval_size,
        split_seed=split_seed,
        shuffle_before_split=shuffle_before_split,
    )

    train_loader = build_protein_dataloader(
        Subset(dataset, list(train_indices)),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        **dataloader_kwargs,
    )
    eval_loader = None
    if eval_indices:
        eval_loader = build_protein_dataloader(
            Subset(dataset, list(eval_indices)),
            batch_size=batch_size,
            shuffle=eval_shuffle,
            collate_fn=collate_fn,
            **dataloader_kwargs,
        )

    return TrainEvalDataLoaders(
        train_loader=train_loader,
        eval_loader=eval_loader,
        train_indices=train_indices,
        eval_indices=eval_indices,
    )