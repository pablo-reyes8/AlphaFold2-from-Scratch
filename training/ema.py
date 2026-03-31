"""Exponential moving average utilities for model parameters.

This file keeps a shadow copy of trainable weights, supports swapping EMA
weights into the live model for evaluation, and provides health checks and
state-dict helpers used by the checkpoint and training code.
"""

import copy
from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


class EMA:
    """
    Exponential Moving Average over trainable model parameters.

    Features
    --------
    - stores shadow weights in fp32
    - maps by parameter name (robust across checkpointing / requires_grad changes)
    - supports optional offloading to cpu
    - can temporarily swap EMA weights into the model for evaluation

    Parameters
    ----------
    model : nn.Module
    decay : float
        Base EMA decay, e.g. 0.999 or 0.9999
    device : str | torch.device | None
        Where to store the shadow weights. Use "cpu" to save GPU memory.
    use_num_updates : bool
        If True, adapt effective decay at early steps.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        device: Optional[str | torch.device] = None,
        use_num_updates: bool = True):

        self.decay = float(decay)
        self.device = device
        self.use_num_updates = bool(use_num_updates)
        self.num_updates = 0

        model = unwrap_model(model)

        self.shadow = {}
        self.backup = {}

        for name, p in model.named_parameters():
            if p.requires_grad:
                s = p.detach().to(dtype=torch.float32).clone()
                if self.device is not None:
                    s = s.to(self.device)
                self.shadow[name] = s

    def _get_decay(self) -> float:
        """
        Optionally use a lower EMA decay early in training, then asymptote to self.decay.
        This is common and helps EMA start tracking sensibly.
        """
        if not self.use_num_updates:
            return self.decay

        # simple warmup-style schedule
        # starts lower, approaches target decay as updates grow
        self.num_updates += 1
        d = min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))
        return float(d)

    @torch.no_grad()
    def update(self, model: nn.Module):
        """
        Update EMA shadow from current model parameters.
        """
        model = unwrap_model(model)
        decay = self._get_decay()

        for name, p in model.named_parameters():
            if name in self.shadow and p.requires_grad:
                s = self.shadow[name]
                p32 = p.detach().to(dtype=torch.float32)

                if s.device != p32.device:
                    p32 = p32.to(s.device)

                s.mul_(decay).add_(p32, alpha=1.0 - decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        """
        Copy EMA weights into model parameters.
        """
        model = unwrap_model(model)
        for name, p in model.named_parameters():
            if name in self.shadow:
                s = self.shadow[name]
                p.data.copy_(s.to(device=p.device, dtype=p.dtype))

    @torch.no_grad()
    def store(self, model: nn.Module):
        """
        Store current model params so we can later restore them after EMA eval.
        """
        model = unwrap_model(model)
        self.backup = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.backup[name] = p.detach().clone()

    @torch.no_grad()
    def restore(self, model: nn.Module):
        """
        Restore model params previously saved by store().
        """
        model = unwrap_model(model)
        for name, p in model.named_parameters():
            if name in self.backup:
                p.data.copy_(self.backup[name].to(device=p.device, dtype=p.dtype))
        self.backup = {}

    @contextmanager
    def average_parameters(self, model: nn.Module):
        """
        Temporarily swap EMA weights into the model for evaluation.

        Usage:
            with ema.average_parameters(model):
                val_metrics = validate(...)
        """
        self.store(model)
        self.copy_to(model)
        try:
            yield
        finally:
            self.restore(model)

    @torch.no_grad()
    def to(self, device: str | torch.device):
        """
        Move EMA shadow weights to a new device.
        """
        self.device = device
        for name in self.shadow:
            self.shadow[name] = self.shadow[name].to(device)

    @torch.no_grad()
    def state_dict(self):
        """
        Safe checkpoint state.
        """
        return {
            "decay": self.decay,
            "device": str(self.device) if self.device is not None else None,
            "use_num_updates": self.use_num_updates,
            "num_updates": self.num_updates,
            "shadow": {name: s.detach().cpu() for name, s in self.shadow.items()}}

    @torch.no_grad()
    def load_state_dict(self, state_dict):
        """
        Restore EMA state from checkpoint.
        """
        self.decay = float(state_dict.get("decay", self.decay))
        self.use_num_updates = bool(state_dict.get("use_num_updates", self.use_num_updates))
        self.num_updates = int(state_dict.get("num_updates", self.num_updates))

        loaded_shadow = state_dict.get("shadow", {})

        for name, s in self.shadow.items():
            if name in loaded_shadow:
                s.data.copy_(loaded_shadow[name].to(device=s.device, dtype=s.dtype))
            else:
                print(f"[EMA] Warning: parameter '{name}' missing in checkpoint.")

    def __len__(self):
        return len(self.shadow)

@torch.no_grad()
def ema_health(ema: EMA, model: nn.Module, rel_tol: float = 5.0):
    """
    Basic sanity check comparing EMA weights against current model weights.

    Returns
    -------
    (ok: bool, status: str, rel_diff: float)
    """
    model = unwrap_model(model)

    def _flat(t):
        return t.detach().float().cpu().reshape(-1)

    m_params = []
    e_params = []

    for name, p in model.named_parameters():
        if name in ema.shadow:
            m_params.append(p)
            e_params.append(ema.shadow[name])

    if not m_params:
        return (False, "empty_ema", float("inf"))

    m_flat = torch.cat([_flat(p) for p in m_params], dim=0)
    e_flat = torch.cat([_flat(s) for s in e_params], dim=0)

    if not torch.isfinite(e_flat).all():
        return (False, "nan_or_inf_in_ema", float("inf"))

    m_norm = m_flat.norm().item()
    e_norm = e_flat.norm().item()

    if e_norm < 1e-12:
        return (False, "ema_zero_norm", float("inf"))
    if m_norm < 1e-12:
        return (False, "model_zero_norm", float("inf"))

    rel = (m_flat - e_flat).norm().item() / (m_norm + 1e-8)

    if rel > rel_tol:
        return (False, "large_rel_diff", rel)

    return (True, "ok", rel)


@torch.no_grad()
def ema_reinit_from_model(ema: EMA, model: nn.Module):
    """
    Hard reset EMA weights from current model weights.
    """
    model = unwrap_model(model)
    for name, p in model.named_parameters():
        if name in ema.shadow:
            s = ema.shadow[name]
            s.data.copy_(p.detach().to(dtype=torch.float32, device=s.device))


def ema_set_decay(ema: EMA, new_decay: float):
    ema.decay = float(new_decay)
