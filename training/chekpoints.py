import os
import random
from pathlib import Path
from typing import Optional, Dict, Any

import torch


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _safe_state_dict(obj):
    return None if obj is None else obj.state_dict()


def get_rng_state():
    state = {
        "torch": torch.get_rng_state(),
        "python": random.getstate()}

    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state):
    if not isinstance(state, dict):
        return
    if "torch" in state and state["torch"] is not None:
        torch.set_rng_state(state["torch"])
    if "python" in state and state["python"] is not None:
        random.setstate(state["python"])
    if torch.cuda.is_available() and "cuda" in state and state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path: str,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    ema=None,
    epoch: int = 0,
    global_step: int = 0,
    best_metric: Optional[float] = None,
    monitor_name: str = "val_loss",
    metrics: Optional[Dict[str, float]] = None,
    config: Optional[Dict[str, Any]] = None,
    save_optimizer_state: bool = True,
    save_rng_state: bool = True):

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    model_to_save = unwrap_model(model)

    ckpt = {
        "model": model_to_save.state_dict(),
        "ema": _safe_state_dict(ema),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_metric": None if best_metric is None else float(best_metric),
        "monitor_name": monitor_name,
        "metrics": metrics or {},
        "config": config or {},
        "rng_state": get_rng_state() if save_rng_state else None}

    if save_optimizer_state:
        ckpt["optimizer"] = _safe_state_dict(optimizer)
        ckpt["scheduler"] = _safe_state_dict(scheduler)
        ckpt["scaler"] = _safe_state_dict(scaler)
    else:
        ckpt["optimizer"] = None
        ckpt["scheduler"] = None
        ckpt["scaler"] = None

    torch.save(ckpt, str(path))


def save_weights_only_checkpoint(
    path: str,
    model,
    ema=None,
    epoch: int = 0,
    global_step: int = 0,
    metrics: Optional[Dict[str, float]] = None,
    monitor_name: str = "val_loss"):

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    model_to_save = unwrap_model(model)

    ckpt = {
        "model": model_to_save.state_dict(),
        "ema": _safe_state_dict(ema),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "metrics": metrics or {},
        "monitor_name": monitor_name}

    torch.save(ckpt, str(path))


def load_checkpoint(
    path: str,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    ema=None,
    map_location="cpu",
    strict: bool = True,
    load_optimizer_state: bool = True,
    restore_rng_state: bool = False):

    ckpt = torch.load(path, map_location=map_location, weights_only=False)

    model_to_load = unwrap_model(model)
    model_to_load.load_state_dict(ckpt["model"], strict=strict)

    if ema is not None and ckpt.get("ema") is not None:
        ema.load_state_dict(ckpt["ema"])

    if load_optimizer_state:
        if optimizer is not None and ckpt.get("optimizer") is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler is not None and ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        if scaler is not None and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])

    if restore_rng_state and ckpt.get("rng_state") is not None:
        set_rng_state(ckpt["rng_state"])

    return ckpt



def get_resume_state(ckpt: Dict[str, Any]):
    """
    Extract standard resume info.
    """
    return {
        "epoch": int(ckpt.get("epoch", 0)),
        "global_step": int(ckpt.get("global_step", 0)),
        "best_metric": ckpt.get("best_metric", None),
        "monitor_name": ckpt.get("monitor_name", "val_loss"),
        "metrics": ckpt.get("metrics", {}),
        "config": ckpt.get("config", {})}


def is_better_metric(current: float, best: Optional[float], mode: str = "min") -> bool:
    if best is None:
        return True
    if mode == "min":
        return current < best
    elif mode == "max":
        return current > best
    else:
        raise ValueError(f"mode must be 'min' or 'max', got {mode}")


def maybe_save_best_and_last(
    save_dir: str,
    model,
    optimizer,
    scheduler,
    scaler,
    ema,
    epoch: int,
    global_step: int,
    current_metric: float,
    best_metric: Optional[float],
    metric_name: str,
    mode: str,
    val_metrics: Dict[str, float],
    config: Optional[Dict[str, Any]] = None):

    os.makedirs(save_dir, exist_ok=True)

    improved = is_better_metric(current_metric, best_metric, mode=mode)
    new_best_metric = current_metric if improved else best_metric

    # save LAST with the UPDATED best metric
    save_checkpoint(
        path=os.path.join(save_dir, "last.pt"),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        ema=ema,
        epoch=epoch,
        global_step=global_step,
        best_metric=new_best_metric,
        monitor_name=metric_name,
        metrics=val_metrics,
        config=config,
        save_optimizer_state=True,
        save_rng_state=True)

    if improved:
        save_weights_only_checkpoint(
            path=os.path.join(save_dir, "best.pt"),
            model=model,
            ema=ema,
            epoch=epoch,
            global_step=global_step,
            metrics=val_metrics,
            monitor_name=metric_name)

    return new_best_metric, improved