from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.distributed as dist

from ortho_llm.data import MemmapTokenDataset
from ortho_llm.train.metrics import perplexity


def _distributed_min(value: int, device: torch.device) -> int:
    if not dist.is_available() or not dist.is_initialized():
        return value
    tensor = torch.tensor(value, dtype=torch.long, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    return int(tensor.item())


def capped_eval_batches(dataset: MemmapTokenDataset, requested_batches: int, device: torch.device) -> int:
    if requested_batches <= 0:
        return 0
    local_batches = dataset.num_tokens // dataset.tokens_per_batch
    max_batches = _distributed_min(local_batches, device)
    return min(requested_batches, max_batches)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataset: MemmapTokenDataset,
    *,
    num_batches: int,
    device: torch.device,
) -> dict[str, float | int | None]:
    num_batches = capped_eval_batches(dataset, num_batches, device)
    if num_batches <= 0:
        return {"val_loss": None, "val_ppl": None, "val_batches": 0}

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
    loss = torch.tensor(sum(losses) / len(losses), device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        loss /= dist.get_world_size()
    loss = float(loss.item())
    return {"val_loss": loss, "val_ppl": perplexity(loss), "val_batches": num_batches}
