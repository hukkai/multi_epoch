from __future__ import annotations

import math


def cosine_lr(
    step: int,
    num_steps: int,
    warmup_steps: int,
    max_lr: float,
    min_lr: float,
    cosine_power: float = 1.0,
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    if num_steps <= warmup_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, num_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    cosine = cosine**cosine_power
    return min_lr + (max_lr - min_lr) * cosine
