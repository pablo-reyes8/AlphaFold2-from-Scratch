"""Validate lightweight data-ablation helpers without requiring full training."""

from __future__ import annotations

from data.dataloaders import select_msa_sequences


def test_select_msa_sequences_preserves_full_msa_by_default():
    selected = select_msa_sequences(
        ["AAAA", "BBBB", "CCCC"],
        target_sequence="WXYZ",
        target_len=4,
        max_msa_seqs=2,
        single_sequence_mode=False,
    )

    assert selected == ["AAAA", "BBBB"]


def test_select_msa_sequences_collapses_to_target_in_single_sequence_mode():
    selected = select_msa_sequences(
        ["AAAA", "BBBB", "CCCC"],
        target_sequence="WXYZ",
        target_len=4,
        max_msa_seqs=128,
        single_sequence_mode=True,
    )

    assert selected == ["WXYZ"]
