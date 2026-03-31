import math
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn


def build_alphafold_param_groups(
    model: nn.Module,
    weight_decay: float = 1e-4):

    """
    Build optimizer parameter groups for AlphaFold2-like model.

    No weight decay for:
      - biases
      - normalization params
      - embeddings
      - 1D parameters

    Returns
    -------
    List[dict]
    """
    model = model.module if hasattr(model, "module") else model

    decay_params = []
    no_decay_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        name_l = name.lower()

        is_bias = name.endswith(".bias")
        is_norm = (
            "norm" in name_l or
            "layernorm" in name_l or
            ".ln" in name_l or
            "batchnorm" in name_l or
            ".bn" in name_l or
            "groupnorm" in name_l)

        is_embedding = "embedding" in name_l or "embed" in name_l
        is_1d = p.ndim <= 1

        if is_bias or is_norm or is_embedding or is_1d:
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0}]


class WarmupCosineLR:
    """
    Linear warmup, then cosine decay to min_lr. Step-based scheduler.

    Behavior
    --------
    - steps 1..warmup_steps:
        lr increases linearly from 0 to base_lr
    - after warmup:
        cosine decay from base_lr to min_lr
    - resume-safe through state_dict / load_state_dict
    """

    def __init__(
        self,
        optimizer,
        total_steps: int,
        warmup_steps: int,
        min_lr: float = 0.0):

        if total_steps <= 0:
            raise ValueError(f"total_steps must be > 0, got {total_steps}")
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
        if min_lr < 0:
            raise ValueError(f"min_lr must be >= 0, got {min_lr}")

        self.optimizer = optimizer
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.min_lr = float(min_lr)

        self.base_lrs = [float(g["lr"]) for g in optimizer.param_groups]
        self.step_num = 0

    def _compute_lr(self, base_lr: float, t: int) -> float:
        # Warmup
        if self.warmup_steps > 0 and t <= self.warmup_steps:
            return base_lr * (t / max(1, self.warmup_steps))

        # Cosine phase
        if self.total_steps <= self.warmup_steps:
            # Degenerate case: just stay at min_lr after warmup region
            return self.min_lr

        tt = min(max(t, self.warmup_steps), self.total_steps)
        denom = max(1, self.total_steps - self.warmup_steps)
        progress = (tt - self.warmup_steps) / denom
        progress = min(1.0, max(0.0, progress))

        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = self.min_lr + (base_lr - self.min_lr) * cosine
        return lr

    def _set_lr(self, t: int):
        for i, group in enumerate(self.optimizer.param_groups):
            base_lr = self.base_lrs[i]
            group["lr"] = self._compute_lr(base_lr, t)

    def step(self):
        self.step_num += 1
        self._set_lr(self.step_num)

    def get_last_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self):
        return {
            "step_num": int(self.step_num),
            "base_lrs": list(self.base_lrs),
            "min_lr": float(self.min_lr),
            "total_steps": int(self.total_steps),
            "warmup_steps": int(self.warmup_steps)}

    def load_state_dict(self, state_dict):
        if not isinstance(state_dict, dict):
            return

        self.step_num = int(state_dict.get("step_num", 0))

        loaded_base_lrs = state_dict.get("base_lrs", None)
        if isinstance(loaded_base_lrs, (list, tuple)) and len(loaded_base_lrs) == len(self.optimizer.param_groups):
            self.base_lrs = [float(x) for x in loaded_base_lrs]

        self.min_lr = float(state_dict.get("min_lr", self.min_lr))
        self.total_steps = int(state_dict.get("total_steps", self.total_steps))
        self.warmup_steps = int(state_dict.get("warmup_steps", self.warmup_steps))

        # restore LR exactly to the resumed step
        self._set_lr(self.step_num)

def build_optimizer_and_scheduler(
    model: nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    betas=(0.9, 0.95),
    eps: float = 1e-8,
    total_steps: int = 100000,
    warmup_steps: int = 5000,
    min_lr: float = 1e-6):

    """
    Build AdamW optimizer + warmup cosine scheduler for AlphaFold2-like training.
    """
    param_groups = build_alphafold_param_groups(
        model=model,
        weight_decay=weight_decay)

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=lr,
        betas=betas,
        eps=eps)

    scheduler = WarmupCosineLR(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr=min_lr)

    return optimizer, scheduler