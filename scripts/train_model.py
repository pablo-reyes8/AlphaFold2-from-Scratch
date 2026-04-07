"""CLI entry point for running configured training jobs.

This launcher builds the dataset, dataloader, model, loss, optimizer,
scheduler, EMA, AMP runtime, and geometric constants from the YAML experiment
config, then dispatches the repository training loop with a small set of
practical command-line overrides.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.common import (
    build_amp_runtime,
    build_dataset_from_config,
    build_ema_from_config,
    build_ideal_backbone_local,
    build_loss_from_config,
    build_model_from_config,
    build_optimizer_scheduler_from_config,
    build_train_eval_dataloaders_from_config,
    choose_device,
    count_trainable_parameters,
    load_yaml_config,
)
from training.seeds import seed_everything
from training.train_alphafold2 import train_alphafold2


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the AlphaFold2-like model from a YAML config.")
    parser.add_argument("--config", type=str, default="config/experiments/af2_poc.yaml")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Force a one-epoch, one-batch smoke training run.")
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--amp-dtype", type=str, default=None)
    parser.add_argument("--num-recycles", type=int, default=None)
    parser.add_argument("--eval-size", type=int, default=None)
    parser.add_argument("--stochastic-recycling", action="store_true")
    parser.add_argument("--max-recycles", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_yaml_config(args.config)
    trainer_cfg = dict(config.get("trainer", {}))
    data_loader_cfg = dict(config.get("data", {}).get("loader", {}))

    device = choose_device(args.device)
    seed_everything(
        seed=int(config.get("seed", 42) if args.seed is None else args.seed),
        deterministic=args.deterministic,
    )

    dataset = build_dataset_from_config(
        config,
        max_samples=1 if args.dry_run and args.max_samples is None else args.max_samples,
        verbose=True,
    )
    if len(dataset) == 0:
        raise ValueError("Dataset resolved zero valid examples. Check the manifest and local data paths.")

    loader, eval_loader, split_info = build_train_eval_dataloaders_from_config(
        dataset,
        config,
        batch_size=1 if args.dry_run and args.batch_size is None else args.batch_size,
        shuffle=False if args.dry_run else None,
        eval_size=0 if args.dry_run else args.eval_size,
    )

    model = build_model_from_config(config, device=device)
    criterion = build_loss_from_config(config, device=device)
    ideal_backbone_local = build_ideal_backbone_local(config, device=device)

    epochs = 1 if args.dry_run else int(trainer_cfg.get("epochs", 1) if args.epochs is None else args.epochs)
    grad_accum_steps = int(trainer_cfg.get("grad_accum_steps", 1))
    max_batches = 1 if args.dry_run else args.max_batches

    optimizer, scheduler = build_optimizer_scheduler_from_config(
        model,
        config,
        num_batches=len(loader),
        epochs=epochs,
        grad_accum_steps=grad_accum_steps,
        max_batches=max_batches,
    )

    ema = None if args.no_ema else build_ema_from_config(model, config)
    amp_runtime = build_amp_runtime(
        config,
        device=device,
        amp_enabled=False if args.no_amp else None,
        amp_dtype=args.amp_dtype,
    )

    num_recycles = int(trainer_cfg.get("num_recycles", 0) if args.num_recycles is None else args.num_recycles)
    stochastic_recycling = bool(trainer_cfg.get("stochastic_recycling", False) or args.stochastic_recycling)
    max_recycles = args.max_recycles
    if max_recycles is None and "max_recycles" in trainer_cfg:
        max_recycles = int(trainer_cfg["max_recycles"])

    print(
        {
            "device": device,
            "dataset_examples": len(dataset),
            "train_examples": len(split_info["train_indices"]),
            "eval_examples": len(split_info["eval_indices"]),
            "loader_batch_size": int(args.batch_size or data_loader_cfg.get("batch_size", 1)),
            "epochs": epochs,
            "max_batches": max_batches,
            "trainable_parameters": count_trainable_parameters(model),
            "num_recycles": num_recycles,
            "stochastic_recycling": stochastic_recycling,
            "max_recycles": max_recycles,
            "amp_enabled_effective": amp_runtime["amp_enabled"],
            "amp_dtype_effective": str(amp_runtime["amp_dtype_effective"]),
        }
    )

    train_alphafold2(
        model=model,
        train_loader=loader,
        eval_loader=eval_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        ema=ema,
        scaler=amp_runtime["scaler"],
        device=device,
        epochs=epochs,
        amp_enabled=bool(amp_runtime["amp_enabled"]),
        amp_dtype=str(args.amp_dtype or trainer_cfg.get("amp_dtype", "bf16")),
        grad_clip=trainer_cfg.get("grad_clip", 1.0),
        grad_accum_steps=grad_accum_steps,
        log_every=int(trainer_cfg.get("log_every", 10)),
        log_grad_norm=bool(trainer_cfg.get("log_grad_norm", True)),
        log_mem=bool(trainer_cfg.get("log_mem", False)),
        max_batches=max_batches,
        on_oom=str(trainer_cfg.get("on_oom", "skip")),
        ideal_backbone_local=ideal_backbone_local,
        num_recycles=num_recycles,
        stochastic_recycling=stochastic_recycling,
        max_recycles=max_recycles,
        ckpt_dir=str(trainer_cfg.get("ckpt_dir", "checkpoints")),
        run_name=str(trainer_cfg.get("run_name", "alphafold2")),
        save_every=int(trainer_cfg.get("save_every", 1)),
        save_last=bool(trainer_cfg.get("save_last", True)),
        eval_every=int(trainer_cfg.get("eval_every", 1)),
        monitor_name=str(trainer_cfg.get("monitor_name", "loss")),
        monitor_mode=str(trainer_cfg.get("monitor_mode", "min")),
        config=config,
        resume_path=args.resume_path,
    )


if __name__ == "__main__":
    main()
