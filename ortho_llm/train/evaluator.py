from __future__ import annotations

from contextlib import nullcontext

import torch

from ortho_llm.data import MemmapTokenDataset
from ortho_llm.train.metrics import perplexity


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataset: MemmapTokenDataset,
    *,
    num_batches: int,
    device: torch.device,
) -> dict[str, float | None]:
    if num_batches <= 0:
        return {"val_loss": None, "val_ppl": None}

    was_training = model.training
    model.eval()
    losses = []
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    state = dataset.state_dict()
    for _ in range(num_batches):
        input_ids, labels = dataset.next_batch()
        with autocast_ctx:
            output = model(input_ids=input_ids, labels=labels)
        losses.append(float(output["loss"].detach().item()))
    dataset.load_state_dict(state)
    if was_training:
        model.train()
    loss = sum(losses) / len(losses)
    return {"val_loss": loss, "val_ppl": perplexity(loss)}
