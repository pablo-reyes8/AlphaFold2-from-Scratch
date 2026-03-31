from __future__ import annotations

import torch
import torch.nn as nn

from training.scheduler_warmup import WarmupCosineLR


def test_warmup_cosine_lr_resume():
    model = nn.Linear(8, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    scheduler = WarmupCosineLR(
        optimizer=optimizer,
        total_steps=20,
        warmup_steps=5,
        min_lr=1e-5,
    )

    lrs = []
    for _ in range(10):
        scheduler.step()
        lrs.append(scheduler.get_last_lr()[0])

    assert lrs[0] < lrs[1] < lrs[4]
    assert lrs[5] > lrs[-1]

    state = scheduler.state_dict()
    scheduler2 = WarmupCosineLR(
        optimizer=optimizer,
        total_steps=20,
        warmup_steps=5,
        min_lr=1e-5,
    )
    scheduler2.load_state_dict(state)

    assert scheduler2.step_num == scheduler.step_num
    assert abs(scheduler2.get_last_lr()[0] - scheduler.get_last_lr()[0]) < 1e-12
