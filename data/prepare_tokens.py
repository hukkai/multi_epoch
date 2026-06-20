from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np


UINT32_MAX = int(np.iinfo(np.uint32).max)

WORKER_TOKENIZER = None
WORKER_DATASET = None
WORKER_TEXT_COLUMN = None
WORKER_EOS_TOKEN_ID = None


@dataclass(frozen=True)
class TokenWriteResult:
    split: str
    path: Path
    documents: int
    tokens: int


@dataclass(frozen=True)
class WorkerConfig:
    tokenizer: str
    dataset_name: str
    dataset_config: str | None
    split: str
    text_column: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("Prepare token shards for LLaMA pretraining")
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--dataset-config", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="validation")
    parser.add_argument("--text-column", type=str, default="text")
    parser.add_argument("--shard-rank", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--num-documents", type=int, default=0, help="0 means use the full train split")
    parser.add_argument("--val-num-documents", type=int, default=0, help="0 means use the full validation split")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--val-output-dir", type=str, default=None)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    validate_shard_settings(args.shard_rank, args.num_shards, cli=True)
    if args.num_workers <= 0:
        raise ValueError("--num-workers must be positive")
    if args.num_documents < 0:
        raise ValueError("--num-documents must be non-negative")
    if args.val_num_documents < 0:
        raise ValueError("--val-num-documents must be non-negative")


def validate_shard_settings(shard_rank: int, num_shards: int, *, cli: bool = False) -> None:
    num_shards_name = "--num-shards" if cli else "num_shards"
    shard_rank_name = "--shard-rank" if cli else "shard_rank"
    if num_shards <= 0:
        raise ValueError(f"{num_shards_name} must be positive")
    if shard_rank < 0 or shard_rank >= num_shards:
        raise ValueError(f"{shard_rank_name} must satisfy 0 <= shard_rank < {num_shards_name}")


def _load_dataset(dataset_name: str, dataset_config: str | None, split: str):
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The 'datasets' package is required to prepare token shards. "
            "Install HuggingFace datasets before running this command."
        ) from exc
    return load_dataset(dataset_name, dataset_config, split=split)


def _load_tokenizer(tokenizer_name: str):
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The 'transformers' package is required to prepare token shards. "
            "Install HuggingFace transformers before running this command."
        ) from exc
    return AutoTokenizer.from_pretrained(tokenizer_name)


def _progress(iterable: Iterable, *, total: int, desc: str, enabled: bool) -> Iterable:
    if not enabled:
        return iterable
    try:
        from tqdm import tqdm
    except ModuleNotFoundError:
        return iterable
    return tqdm(iterable, total=total, desc=desc)


def _document_limit(dataset_length: int, requested_documents: int) -> int:
    if requested_documents <= 0:
        return dataset_length
    return min(dataset_length, requested_documents)


def _shard_document_count(total_documents: int, shard_rank: int, num_shards: int) -> int:
    if total_documents <= shard_rank:
        return 0
    return ((total_documents - 1 - shard_rank) // num_shards) + 1


def _require_eos_token_id(tokenizer) -> int:
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        raise ValueError("Tokenizer must define an EOS token")
    if eos_token_id < 0 or eos_token_id > UINT32_MAX:
        raise ValueError(f"EOS token id {eos_token_id} is outside uint32 range")
    return int(eos_token_id)


def _extract_text(example, text_column: str) -> str:
    if text_column not in example:
        available = ", ".join(str(key) for key in example.keys())
        raise KeyError(f"Missing text column {text_column!r}; available columns: {available}")
    text = example[text_column]
    if not isinstance(text, str):
        raise TypeError(f"Text column {text_column!r} must contain strings, got {type(text).__name__}")
    return text


def _validate_token_ids(token_ids: Sequence[int], *, context: str) -> None:
    for token_id in token_ids:
        if token_id < 0 or token_id > UINT32_MAX:
            raise ValueError(f"Token id {token_id} in {context} is outside uint32 range")


def tokenize_example(example, tokenizer, *, text_column: str, eos_token_id: int) -> np.ndarray:
    text = _extract_text(example, text_column)
    token_ids = list(tokenizer.encode(text, add_special_tokens=False))
    token_ids.append(eos_token_id)
    _validate_token_ids(token_ids, context=f"column {text_column!r}")
    return np.asarray(token_ids, dtype=np.uint32)


def _iter_token_arrays(
    dataset,
    tokenizer,
    *,
    text_column: str,
    shard_rank: int,
    num_shards: int,
    num_documents: int,
    split: str,
    show_progress: bool,
) -> Iterator[np.ndarray]:
    validate_shard_settings(shard_rank, num_shards)
    if num_documents < 0:
        raise ValueError("num_documents must be non-negative")
    total_documents = _document_limit(len(dataset), num_documents)
    shard_documents = _shard_document_count(total_documents, shard_rank, num_shards)
    eos_token_id = _require_eos_token_id(tokenizer)
    doc_indices = range(shard_rank, total_documents, num_shards)
    for doc_idx in _progress(doc_indices, total=shard_documents, desc=f"{split} shard {shard_rank}", enabled=show_progress):
        yield tokenize_example(dataset[doc_idx], tokenizer, text_column=text_column, eos_token_id=eos_token_id)


def _write_token_arrays(
    arrays: Iterable[np.ndarray],
    *,
    output_dir: str | Path,
    shard_rank: int,
    split: str,
) -> TokenWriteResult:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"tokens_{shard_rank}.bin"
    temp_path = output_root / f".tokens_{shard_rank}.bin.tmp.{os.getpid()}"

    documents = 0
    tokens = 0
    try:
        with temp_path.open("wb") as handle:
            for token_array in arrays:
                if token_array.dtype != np.uint32:
                    token_array = token_array.astype(np.uint32, copy=False)
                token_array.tofile(handle)
                documents += 1
                tokens += int(token_array.shape[0])
        if tokens == 0:
            raise ValueError(
                f"No tokens were written for {split} shard {shard_rank}; "
                "check split size, num_documents, shard_rank, and num_shards"
            )
        os.replace(temp_path, output_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    return TokenWriteResult(split=split, path=output_path, documents=documents, tokens=tokens)


def write_token_shard(
    dataset,
    tokenizer,
    *,
    output_dir: str | Path,
    text_column: str,
    shard_rank: int,
    num_shards: int,
    num_documents: int = 0,
    split: str = "train",
    show_progress: bool = True,
) -> TokenWriteResult:
    arrays = _iter_token_arrays(
        dataset,
        tokenizer,
        text_column=text_column,
        shard_rank=shard_rank,
        num_shards=num_shards,
        num_documents=num_documents,
        split=split,
        show_progress=show_progress,
    )
    return _write_token_arrays(arrays, output_dir=output_dir, shard_rank=shard_rank, split=split)


def init_worker(config: WorkerConfig) -> None:
    global WORKER_TOKENIZER, WORKER_DATASET, WORKER_TEXT_COLUMN, WORKER_EOS_TOKEN_ID
    WORKER_TOKENIZER = _load_tokenizer(config.tokenizer)
    WORKER_DATASET = _load_dataset(config.dataset_name, config.dataset_config, config.split)
    WORKER_TEXT_COLUMN = config.text_column
    WORKER_EOS_TOKEN_ID = _require_eos_token_id(WORKER_TOKENIZER)


def tokenize_worker(doc_idx: int) -> np.ndarray:
    return tokenize_example(
        WORKER_DATASET[doc_idx],
        WORKER_TOKENIZER,
        text_column=WORKER_TEXT_COLUMN,
        eos_token_id=WORKER_EOS_TOKEN_ID,
    )


def write_hf_token_shard(
    *,
    tokenizer_name: str,
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    output_dir: str | Path,
    text_column: str,
    shard_rank: int,
    num_shards: int,
    num_documents: int,
    num_workers: int,
) -> TokenWriteResult:
    validate_shard_settings(shard_rank, num_shards)
    if num_documents < 0:
        raise ValueError("num_documents must be non-negative")
    if num_workers <= 0:
        raise ValueError("num_workers must be positive")
    dataset = _load_dataset(dataset_name, dataset_config, split)
    total_documents = _document_limit(len(dataset), num_documents)
    shard_documents = _shard_document_count(total_documents, shard_rank, num_shards)
    if shard_documents == 0:
        raise ValueError(
            f"No documents selected for {split} shard {shard_rank}; "
            "check split size, num_documents, shard_rank, and num_shards"
        )

    if num_workers == 1:
        tokenizer = _load_tokenizer(tokenizer_name)
        return write_token_shard(
            dataset,
            tokenizer,
            output_dir=output_dir,
            text_column=text_column,
            shard_rank=shard_rank,
            num_shards=num_shards,
            num_documents=num_documents,
            split=split,
            show_progress=True,
        )

    config = WorkerConfig(
        tokenizer=tokenizer_name,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split=split,
        text_column=text_column,
    )
    doc_indices = range(shard_rank, total_documents, num_shards)
    ctx = mp.get_context("spawn")
    with ctx.Pool(num_workers, initializer=init_worker, initargs=(config,)) as pool:
        arrays = pool.imap(tokenize_worker, doc_indices, chunksize=16)
        arrays = _progress(arrays, total=shard_documents, desc=f"{split} shard {shard_rank}", enabled=True)
        return _write_token_arrays(arrays, output_dir=output_dir, shard_rank=shard_rank, split=split)


def main() -> None:
    args = parse_args()
    validate_args(args)

    results = [
        write_hf_token_shard(
            tokenizer_name=args.tokenizer,
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config,
            split=args.split,
            output_dir=args.output_dir,
            text_column=args.text_column,
            shard_rank=args.shard_rank,
            num_shards=args.num_shards,
            num_documents=args.num_documents,
            num_workers=args.num_workers,
        )
    ]
    if args.val_output_dir:
        results.append(
            write_hf_token_shard(
                tokenizer_name=args.tokenizer,
                dataset_name=args.dataset_name,
                dataset_config=args.dataset_config,
                split=args.val_split,
                output_dir=args.val_output_dir,
                text_column=args.text_column,
                shard_rank=args.shard_rank,
                num_shards=args.num_shards,
                num_documents=args.val_num_documents,
                num_workers=args.num_workers,
            )
        )

    for result in results:
        print(f"Wrote {result.tokens} {result.split} tokens from {result.documents} documents to {result.path}")


if __name__ == "__main__":
    main()
