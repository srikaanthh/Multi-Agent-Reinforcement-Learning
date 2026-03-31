from __future__ import annotations

from itertools import islice
from pathlib import Path
import re
from typing import Dict, Iterable, Iterator, List, Optional


_FINAL_ANSWER_RE = re.compile(r"####\s*([^\n]+)")
_NUMERIC_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def extract_gsm8k_final_answer(answer: str) -> str:
    """Extract the final GSM8K answer, preferring the `####` suffix when present."""

    match = _FINAL_ANSWER_RE.search(answer)
    candidate = match.group(1).strip() if match else answer.strip().splitlines()[-1]
    numeric = _NUMERIC_RE.search(candidate.replace("$", ""))
    if numeric:
        return numeric.group(0).replace(",", "")
    return candidate.strip()


def build_gsm8k_tasks_from_rows(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    tasks = []
    for row in rows:
        tasks.append(
            {
                "question": row["question"].strip(),
                "ground_truth": extract_gsm8k_final_answer(row["answer"]),
                "source": "openai/gsm8k",
            }
        )
    return tasks


def _resolve_cache_dir(cache_dir: Optional[str]) -> Path:
    resolved_cache_dir = Path(cache_dir) if cache_dir else Path(__file__).resolve().parents[1] / ".cache" / "huggingface"
    resolved_cache_dir.mkdir(parents=True, exist_ok=True)
    return resolved_cache_dir


def iter_gsm8k_tasks(
    split: str = "test",
    limit: Optional[int] = 1,
    *,
    start_index: int = 0,
    shuffle: bool = False,
    seed: int = 42,
    cache_dir: Optional[str] = None,
    streaming: bool = False,
) -> Iterator[Dict[str, str]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The `datasets` package is required to load GSM8K. Install project dependencies first."
        ) from exc

    resolved_cache_dir = _resolve_cache_dir(cache_dir)

    if streaming:
        try:
            dataset = load_dataset(
                "openai/gsm8k",
                "main",
                split=split,
                cache_dir=str(resolved_cache_dir),
                streaming=True,
            )
        except ImportError:
            # Some local Python environments have partially broken torch installs that
            # interfere with HF iterable datasets. Fall back to select-based loading.
            streaming = False

    if not streaming:
        dataset = load_dataset(
            "openai/gsm8k",
            "main",
            split=split,
            cache_dir=str(resolved_cache_dir),
            streaming=False,
        )

    if shuffle:
        if streaming:
            dataset = dataset.shuffle(buffer_size=10_000, seed=seed)
        else:
            dataset = dataset.shuffle(seed=seed)

    if streaming:
        iterator = iter(dataset)
        if start_index > 0:
            iterator = islice(iterator, start_index, None)
        if limit is not None:
            iterator = islice(iterator, limit)
        for row in iterator:
            yield {
                "question": row["question"].strip(),
                "ground_truth": extract_gsm8k_final_answer(row["answer"]),
                "source": "openai/gsm8k",
            }
        return

    dataset_length = len(dataset)
    start = min(max(start_index, 0), dataset_length)
    stop = dataset_length if limit is None else min(start + max(limit, 0), dataset_length)
    if start >= stop:
        return

    selected = dataset.select(range(start, stop))
    for row in selected:
        yield {
            "question": row["question"].strip(),
            "ground_truth": extract_gsm8k_final_answer(row["answer"]),
            "source": "openai/gsm8k",
        }


def load_gsm8k_tasks(
    split: str = "test",
    limit: Optional[int] = 1,
    *,
    start_index: int = 0,
    shuffle: bool = False,
    seed: int = 42,
    cache_dir: Optional[str] = None,
    streaming: bool = False,
) -> List[Dict[str, str]]:
    return list(
        iter_gsm8k_tasks(
            split=split,
            limit=limit,
            start_index=start_index,
            shuffle=shuffle,
            seed=seed,
            cache_dir=cache_dir,
            streaming=streaming,
        )
    )
