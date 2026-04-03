"""Cover CLI parsing for the ablation training launchers without building full models."""

from __future__ import annotations

from scripts.ablations.run_suite import parse_args as parse_run_suite_args
from scripts.train_ablation import parse_args as parse_train_ablation_args
from scripts.train_ablation_parallel import parse_args as parse_train_ablation_parallel_args


def test_train_ablation_cli_parses_listing_and_ablation_overrides():
    args = parse_train_ablation_args(
        [
            "--ablation",
            "AF2_1",
            "--dry-run",
            "--max-recycles",
            "3",
            "--single-sequence-msa",
        ]
    )
    assert args.ablation == "AF2_1"
    assert args.dry_run is True
    assert args.max_recycles == 3
    assert args.single_sequence_msa is True


def test_train_ablation_parallel_cli_parses_parallel_overrides():
    args = parse_train_ablation_parallel_args(
        [
            "--ablation",
            "AF2_4",
            "--parallel-mode",
            "model",
            "--model-devices",
            "cuda:0,cuda:1",
            "--single-sequence-msa",
        ]
    )
    assert args.ablation == "AF2_4"
    assert args.parallel_mode == "model"
    assert args.model_devices == "cuda:0,cuda:1"
    assert args.single_sequence_msa is True


def test_run_suite_cli_parses_baseline_and_output_controls():
    args = parse_run_suite_args(
        [
            "--include-baseline",
            "--ablation",
            "AF2_2",
            "--output-dir",
            "artifacts/ablation_suite",
            "--single-sequence-msa",
        ]
    )
    assert args.include_baseline is True
    assert args.ablation == ["AF2_2"]
    assert args.output_dir == "artifacts/ablation_suite"
    assert args.single_sequence_msa is True
