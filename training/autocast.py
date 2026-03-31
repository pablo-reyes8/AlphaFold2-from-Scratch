"""Automatic mixed precision helpers for training.

This module centralizes dtype resolution, autocast context creation, and
GradScaler handling so the rest of the training stack can stay device-agnostic
and robust across CPU, CUDA, bf16, fp16, and fp32 execution modes.
"""

import inspect
from contextlib import contextmanager, nullcontext
from typing import Optional

import torch


DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32}


def resolve_amp_dtype(
    amp_dtype: str = "bf16",
    device: str = "cuda") -> torch.dtype:

    """
    Resolve the requested AMP dtype to a torch.dtype.
    """
    amp_dtype = amp_dtype.lower()
    if amp_dtype not in DTYPE_MAP:
        raise ValueError(f"Unsupported amp dtype: {amp_dtype}")
    return DTYPE_MAP[amp_dtype]


def cuda_supports_bf16() -> bool:
    """
    Check whether current CUDA device supports bfloat16 autocast.
    """
    if not torch.cuda.is_available():
        return False

    # PyTorch exposes this helper in modern versions
    if hasattr(torch.cuda, "is_bf16_supported"):
        try:
            return torch.cuda.is_bf16_supported()
        except Exception:
            pass

    # Conservative fallback
    major, _ = torch.cuda.get_device_capability()
    return major >= 8


def get_effective_amp_dtype(
    amp_dtype: str = "bf16",
    device: str = "cuda") -> Optional[torch.dtype]:
    """
    Decide the actually usable AMP dtype on the current device.

    Returns:
        torch.dtype if AMP should be used
        None if AMP should be disabled / no-op
    """
    want = resolve_amp_dtype(amp_dtype, device=device)

    if device == "cuda":
        if not torch.cuda.is_available():
            return None

        if want == torch.bfloat16:
            return torch.bfloat16 if cuda_supports_bf16() else torch.float16

        if want == torch.float16:
            return torch.float16

        if want == torch.float32:
            return None

        return None

    if device == "cpu":
        # CPU autocast is mostly meaningful in bf16
        if want == torch.bfloat16:
            return torch.bfloat16
        return None

    return None


def should_use_grad_scaler(
    device: str = "cuda",
    amp_enabled: bool = True,
    amp_dtype: str = "bf16") -> bool:

    """
    GradScaler is useful for fp16, but usually not needed for bf16.
    """
    if not amp_enabled:
        return False

    effective_dtype = get_effective_amp_dtype(amp_dtype=amp_dtype, device=device)

    if device == "cuda" and effective_dtype == torch.float16:
        return True

    return False


def make_grad_scaler(
    device: str = "cuda",
    amp_enabled: bool = True,
    amp_dtype: str = "bf16"):
    """
    Build a GradScaler only when it is actually useful.
    """
    enabled = should_use_grad_scaler(
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype)

    if not enabled:
        return None

    # Newer API
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            sig = inspect.signature(torch.amp.GradScaler)
            if len(sig.parameters) >= 1:
                return torch.amp.GradScaler(device_type=device)
            return torch.amp.GradScaler()
        except Exception:
            pass

    # Older CUDA API
    if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "GradScaler"):
        return torch.cuda.amp.GradScaler()

    return None


@contextmanager
def autocast_ctx(
    device: str = "cuda",
    enabled: bool = True,
    amp_dtype: str = "bf16",
    cache_enabled: bool = True):
    """
    Robust autocast context for AlphaFold2-like training.

    Behavior:
    - CUDA + bf16 requested:
        uses bf16 if supported, else falls back to fp16
    - CUDA + fp16 requested:
        uses fp16
    - CUDA + fp32 requested:
        disables autocast
    - CPU + bf16 requested:
        uses cpu autocast bf16 if available
    - otherwise:
        no-op
    """
    if not enabled:
        with nullcontext():
            yield
        return

    effective_dtype = get_effective_amp_dtype(
        amp_dtype=amp_dtype,
        device=device)

    if effective_dtype is None:
        with nullcontext():
            yield
        return

    if device == "cuda":
        with torch.amp.autocast(
            device_type="cuda",
            dtype=effective_dtype,
            cache_enabled=cache_enabled):

            yield
        return

    if device == "cpu":
        try:
            with torch.amp.autocast(
                device_type="cpu",
                dtype=effective_dtype,
                cache_enabled=cache_enabled):

                yield
        except Exception:
            with nullcontext():
                yield
        return

    with nullcontext():
        yield

def build_amp_config(
    device: str = "cuda",
    amp_enabled: bool = True,
    amp_dtype: str = "bf16"):

    effective_dtype = get_effective_amp_dtype(
        amp_dtype=amp_dtype,
        device=device)

    scaler = make_grad_scaler(
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype)

    return {
        "amp_enabled": amp_enabled and (effective_dtype is not None),
        "amp_dtype_requested": amp_dtype,
        "amp_dtype_effective": effective_dtype,
        "use_grad_scaler": scaler is not None,
        "scaler": scaler}
