from __future__ import annotations

from difflib import SequenceMatcher
from typing import Iterable, List
import math
import re


_WHITESPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"\w+")


def normalize_text(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text.strip().lower())


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = _TOKEN_RE.findall(normalize_text(prediction))
    ref_tokens = _TOKEN_RE.findall(normalize_text(reference))
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counts = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1

    ref_counts = {}
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1

    overlap = 0
    for token, pred_count in pred_counts.items():
        overlap += min(pred_count, ref_counts.get(token, 0))

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def similarity_reward(answer: str, ground_truth: str) -> float:
    answer_norm = normalize_text(answer)
    ground_truth_norm = normalize_text(ground_truth)
    exact = 1.0 if answer_norm == ground_truth_norm else 0.0
    sequence = SequenceMatcher(a=answer_norm, b=ground_truth_norm).ratio()
    token = token_f1(answer_norm, ground_truth_norm)
    return max(exact, 0.5 * sequence + 0.5 * token)


def compute_ground_truth_reward(answer: str, ground_truth: str, task_type: str = "exact") -> float:
    if task_type not in {"exact", "flexible"}:
        raise ValueError(f"Unsupported task_type: {task_type}")
    if task_type == "exact":
        return 1.0 if normalize_text(answer) == normalize_text(ground_truth) else 0.0
    return similarity_reward(answer, ground_truth)


def normalize_scores(scores: List[float]) -> List[float]:
    if not scores:
        return []
    minimum = min(scores)
    maximum = max(scores)
    if math.isclose(minimum, maximum):
        return [0.5 for _ in scores]
    return [(score - minimum) / (maximum - minimum) for score in scores]


def safe_mean(values: Iterable[float], default: float = 0.0) -> float:
    values_list = list(values)
    if not values_list:
        return default
    return sum(values_list) / len(values_list)
