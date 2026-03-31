import torch 
import torch.nn as nn 
from training.scheduler_warmup import *

def test_warmup_cosine_lr():

    model = nn.Linear(8, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    scheduler = WarmupCosineLR(
        optimizer=optimizer,
        total_steps=20,
        warmup_steps=5,
        min_lr=1e-5)

    lrs = []
    for _ in range(10):
        scheduler.step()
        lrs.append(scheduler.get_last_lr()[0])

    print("First 10 LRs:", lrs)

    # save/load
    state = scheduler.state_dict()

    scheduler2 = WarmupCosineLR(
        optimizer=optimizer,
        total_steps=20,
        warmup_steps=5,
        min_lr=1e-5)

    scheduler2.load_state_dict(state)

    assert scheduler2.step_num == scheduler.step_num, "step_num mismatch after resume"
    assert abs(scheduler2.get_last_lr()[0] - scheduler.get_last_lr()[0]) < 1e-12, "LR mismatch after resume"

    print("Scheduler resume OK")


test_warmup_cosine_lr()