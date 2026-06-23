from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

try:
    from prepare_tokens import (
        _load_dataset,
        _load_tokenizer,
        _require_eos_token_id,
        tokenize_example,
        validate_shard_settings,
    )
except ModuleNotFoundError:
    from data.prepare_tokens import (
        _load_dataset,
        _load_tokenizer,
        _require_eos_token_id,
        tokenize_example,
        validate_shard_settings,
    )


DEFAULT_TOTAL_TOKENS = 1_000_000
DEFAULT_NUM_RANKS = 8


@dataclass(frozen=True)
class ValShardResult:
    path: Path
    rank: int
    documents: int
    tokens: int


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("Generate fixed-size validation token shards")
    parser.add_argument("--tokenizer", type=str, default="lmsys/vicuna-7b-v1.5")
    parser.add_argument("--dataset-name", type=str, default="allenai/c4")
    parser.add_argument("--dataset-config", type=str, default="en")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--text-column", type=str, default="text")
    parser.add_argument("--output-dir", type=str, default="data/C4-val-1M")
    parser.add_argument("--total-tokens", type=int, default=DEFAULT_TOTAL_TOKENS)
    parser.add_argument("--num-ranks", type=int, default=DEFAULT_NUM_RANKS)
    parser.add_argument("--max-documents", type=int, default=0, help="0 means scan the full split")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.total_tokens <= 0:
        raise ValueError("--total-tokens must be positive")
    if args.num_ranks <= 0:
        raise ValueError("--num-ranks must be positive")
    if args.total_tokens % args.num_ranks != 0:
        raise ValueError("--total-tokens must be divisible by --num-ranks")
    if args.max_documents < 0:
        raise ValueError("--max-documents must be non-negative")


def _document_limit(dataset_length: int, requested_documents: int) -> int:
    if requested_documents <= 0:
        return dataset_length
    return min(dataset_length, requested_documents)


def _iter_rank_token_arrays(
    dataset,
    tokenizer,
    *,
    rank: int,
    num_ranks: int,
    target_tokens: int,
    text_column: str,
    max_documents: int,
) -> Iterator[tuple[np.ndarray, bool]]:
    validate_shard_settings(rank, num_ranks)
    total_documents = _document_limit(len(dataset), max_documents)
    eos_token_id = _require_eos_token_id(tokenizer)
    written = 0
    for doc_idx in range(rank, total_documents, num_ranks):
        token_array = tokenize_example(
            dataset[doc_idx],
            tokenizer,
            text_column=text_column,
            eos_token_id=eos_token_id,
        )
        written += int(token_array.shape[0])
        yield token_array, written >= target_tokens
        if written >= target_tokens:
            return
    raise ValueError(
        f"Rank {rank} only produced {written} tokens, but {target_tokens} were requested; "
        "increase --max-documents or use a larger validation split"
    )


def write_val_token_shard(
    dataset,
    tokenizer,
    *,
    output_dir: str | Path,
    rank: int,
    num_ranks: int,
    target_tokens: int,
    text_column: str = "text",
    max_documents: int = 0,
    overwrite: bool = False,
) -> ValShardResult:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"tokens_{rank}.bin"
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists; pass --overwrite to replace it")

    temp_path = output_root / f".tokens_{rank}.bin.tmp.{os.getpid()}"
    documents = 0
    tokens = 0
    try:
        with temp_path.open("wb") as handle:
            for token_array, done in _iter_rank_token_arrays(
                dataset,
                tokenizer,
                rank=rank,
                num_ranks=num_ranks,
                target_tokens=target_tokens,
                text_column=text_column,
                max_documents=max_documents,
            ):
                if token_array.dtype != np.uint32:
                    token_array = token_array.astype(np.uint32, copy=False)
                token_array.tofile(handle)
                documents += 1
                tokens += int(token_array.shape[0])
                if done:
                    break
        os.replace(temp_path, output_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    return ValShardResult(path=output_path, rank=rank, documents=documents, tokens=tokens)


def main() -> None:
    args = parse_args()
    validate_args(args)
    target_tokens = args.total_tokens // args.num_ranks
    tokenizer = _load_tokenizer(args.tokenizer)
    dataset = _load_dataset(args.dataset_name, args.dataset_config, args.split)

    total_written = 0
    for rank in range(args.num_ranks):
        result = write_val_token_shard(
            dataset,
            tokenizer,
            output_dir=args.output_dir,
            rank=rank,
            num_ranks=args.num_ranks,
            target_tokens=target_tokens,
            text_column=args.text_column,
            max_documents=args.max_documents,
            overwrite=args.overwrite,
        )
        total_written += result.tokens
        print(f"Wrote {result.tokens} validation tokens from {result.documents} documents to {result.path}")

    print(f"Wrote {total_written} total validation tokens across {args.num_ranks} ranks")


if __name__ == "__main__":
    main()
