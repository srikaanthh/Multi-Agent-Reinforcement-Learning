from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import math

import numpy as np


@dataclass
class AttentionResult:
    weights: np.ndarray
    base_attention: np.ndarray
    entropy: float


class AttentionModule:
    """Trust-weighted attention over peer responses."""

    def __init__(self, temperature: float = 1.0, entropy_floor: float = 1e-8) -> None:
        self.temperature = max(1e-6, float(temperature))
        self.entropy_floor = max(1e-12, float(entropy_floor))

    def compute_weights(
        self,
        query: Sequence[float],
        keys: Sequence[Sequence[float]],
        trust_scores: Sequence[float],
    ) -> np.ndarray:
        return self.compute_attention(query, keys, trust_scores).weights

    def compute_attention(
        self,
        query: Sequence[float],
        keys: Sequence[Sequence[float]],
        trust_scores: Sequence[float],
    ) -> AttentionResult:
        if len(keys) != len(trust_scores):
            raise ValueError("keys and trust_scores must have the same length.")
        if not keys:
            return AttentionResult(
                weights=np.zeros(0, dtype=float),
                base_attention=np.zeros(0, dtype=float),
                entropy=0.0,
            )

        base_attention = self.compute_base_attention(query, keys)
        trust = np.clip(np.asarray(trust_scores, dtype=float), 0.0, None)

        weighted = base_attention * trust
        weighted_sum = float(weighted.sum())
        if math.isclose(weighted_sum, 0.0):
            final_weights = np.full(len(keys), 1.0 / len(keys))
        else:
            final_weights = weighted / weighted_sum

        entropy = float(-(final_weights * np.log(final_weights + self.entropy_floor)).sum())
        return AttentionResult(
            weights=final_weights,
            base_attention=base_attention,
            entropy=entropy,
        )

    def compute_base_attention(
        self,
        query: Sequence[float],
        keys: Sequence[Sequence[float]],
    ) -> np.ndarray:
        if not keys:
            return np.zeros(0, dtype=float)

        query_vec = np.asarray(query, dtype=float)
        key_matrix = np.asarray(keys, dtype=float)
        raw_scores = np.array(
            [self._cosine_similarity(query_vec, key) for key in key_matrix],
            dtype=float,
        )
        scaled = raw_scores / self.temperature
        scaled -= np.max(scaled)
        exp_scores = np.exp(scaled)
        denom = float(exp_scores.sum())
        if denom <= 0.0:
            return np.full(len(keys), 1.0 / len(keys))
        return exp_scores / denom

    @staticmethod
    def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
        left_norm = math.sqrt(float(np.dot(left, left)))
        right_norm = math.sqrt(float(np.dot(right, right)))
        if math.isclose(left_norm, 0.0) or math.isclose(right_norm, 0.0):
            return 0.0
        return float(np.dot(left, right) / (left_norm * right_norm))
