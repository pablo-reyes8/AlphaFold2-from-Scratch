from __future__ import annotations

import torch
import torch.nn as nn

from training.autocast import get_effective_amp_dtype, resolve_amp_dtype
from training.chekpoints import get_resume_state, load_checkpoint, maybe_save_best_and_last, save_checkpoint
from training.ema import EMA, ema_health
from training.scheduler_warmup import WarmupCosineLR, build_alphafold_param_groups
from training.train_one_epoch import compute_grad_norm, move_batch_to_device


def test_resolve_amp_dtype_aliases():
    assert resolve_amp_dtype("bf16") == torch.bfloat16
    assert resolve_amp_dtype("float16") == torch.float16
    assert resolve_amp_dtype("fp32") == torch.float32


def test_get_effective_amp_dtype_cpu_behavior():
    assert get_effective_amp_dtype(device="cpu", amp_dtype="bf16") == torch.bfloat16
    assert get_effective_amp_dtype(device="cpu", amp_dtype="fp16") is None
    assert get_effective_amp_dtype(device="cpu", amp_dtype="fp32") is None


def test_move_batch_to_device_preserves_non_tensors():
    batch = {"x": torch.ones(2, 3), "meta": ["a", "b"], "name": "toy"}
    moved = move_batch_to_device(batch, "cpu")
    assert moved["x"].device.type == "cpu"
    assert moved["meta"] == ["a", "b"]
    assert moved["name"] == "toy"


def test_compute_grad_norm_is_positive_after_backward():
    model = nn.Linear(4, 2)
    inputs = torch.randn(3, 4)
    target = torch.randn(3, 2)
    loss = torch.nn.functional.mse_loss(model(inputs), target)
    loss.backward()
    assert compute_grad_norm(model) > 0.0


def test_build_alphafold_param_groups_separates_decay_and_no_decay():
    class TinyModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_embedding = nn.Embedding(8, 4)
            self.proj = nn.Linear(4, 4)
            self.norm = nn.LayerNorm(4)

    module = TinyModule()
    param_groups = build_alphafold_param_groups(module, weight_decay=1e-4)

    assert len(param_groups) == 2
    decay_group, no_decay_group = param_groups
    assert decay_group["weight_decay"] == 1e-4
    assert no_decay_group["weight_decay"] == 0.0

    no_decay_ids = {id(param) for param in no_decay_group["params"]}
    assert id(module.input_embedding.weight) in no_decay_ids
    assert id(module.proj.bias) in no_decay_ids
    assert id(module.norm.weight) in no_decay_ids
    assert id(module.norm.bias) in no_decay_ids
    assert id(module.proj.weight) not in no_decay_ids


def test_ema_update_store_restore_and_health():
    model = nn.Linear(4, 2)
    ema = EMA(model, decay=0.9, device="cpu", use_num_updates=False)

    with torch.no_grad():
        for param in model.parameters():
            param.add_(1.0)

    ema.update(model)
    ok, status, rel = ema_health(ema, model, rel_tol=10.0)
    assert ok
    assert status == "ok"
    assert rel >= 0.0

    ema.store(model)
    original_weight = model.weight.detach().clone()
    with torch.no_grad():
        model.weight.zero_()
    ema.restore(model)
    assert torch.allclose(model.weight, original_weight)


def test_checkpoint_roundtrip_and_best_last_saving(tmp_path):
    model = nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = WarmupCosineLR(optimizer, total_steps=10, warmup_steps=2, min_lr=1e-5)
    ema = EMA(model, decay=0.999, device="cpu", use_num_updates=False)

    ckpt_path = tmp_path / "manual.pt"
    save_checkpoint(
        path=str(ckpt_path),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        epoch=3,
        global_step=17,
        best_metric=0.25,
        monitor_name="loss",
        metrics={"loss": 0.25},
        config={"run_name": "toy"},
    )

    cloned_model = nn.Linear(4, 2)
    cloned_optimizer = torch.optim.AdamW(cloned_model.parameters(), lr=1e-3)
    cloned_scheduler = WarmupCosineLR(cloned_optimizer, total_steps=10, warmup_steps=2, min_lr=1e-5)
    cloned_ema = EMA(cloned_model, decay=0.999, device="cpu", use_num_updates=False)

    checkpoint = load_checkpoint(
        path=str(ckpt_path),
        model=cloned_model,
        optimizer=cloned_optimizer,
        scheduler=cloned_scheduler,
        ema=cloned_ema,
        map_location="cpu",
    )
    resume = get_resume_state(checkpoint)

    assert resume["epoch"] == 3
    assert resume["global_step"] == 17
    assert resume["best_metric"] == 0.25

    save_dir = tmp_path / "managed"
    best_metric, improved = maybe_save_best_and_last(
        save_dir=str(save_dir),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        ema=ema,
        epoch=4,
        global_step=18,
        current_metric=0.2,
        best_metric=0.25,
        metric_name="loss",
        mode="min",
        val_metrics={"loss": 0.2},
        config={"run_name": "toy"},
    )

    assert improved
    assert best_metric == 0.2
    assert (save_dir / "last.pt").exists()
    assert (save_dir / "best.pt").exists()
