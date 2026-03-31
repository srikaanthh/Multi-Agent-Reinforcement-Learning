from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence
import math
import random

import numpy as np

try:  # pragma: no cover - local torch installs may be broken.
    import torch
except Exception:  # pragma: no cover
    torch = None


def make_log_prob_tensor(values: Sequence[float]):
    if torch is not None:
        try:
            return torch.tensor(list(values), dtype=torch.float32)
        except Exception:
            pass
    return np.asarray(list(values), dtype=float)


@dataclass
class PolicySample:
    action_name: str
    action_index: int
    probabilities: np.ndarray
    log_prob: float
    log_probs: object
    state_features: np.ndarray


class ContextualBanditPolicy:
    """A lightweight REINFORCE-ready policy head for prompt/revision actions."""

    def __init__(
        self,
        action_names: Sequence[str],
        *,
        feature_dim: int = 32,
        learning_rate: float = 0.05,
        seed: Optional[int] = None,
    ) -> None:
        if not action_names:
            raise ValueError("action_names must be non-empty.")
        self.action_names = list(action_names)
        self.feature_dim = int(feature_dim)
        self.learning_rate = float(learning_rate)
        self.rng = random.Random(seed)
        self.weights = np.zeros((self.feature_dim, len(self.action_names)), dtype=float)
        self.bias = np.zeros(len(self.action_names), dtype=float)

    def sample(self, state_features: Sequence[float]) -> PolicySample:
        features = np.asarray(state_features, dtype=float)
        if features.shape[0] != self.feature_dim:
            raise ValueError(f"Expected state_features of size {self.feature_dim}, got {features.shape[0]}.")
        probabilities = self.probabilities(features)
        action_index = self.rng.choices(range(len(self.action_names)), weights=probabilities, k=1)[0]
        prob = max(1e-8, float(probabilities[action_index]))
        return PolicySample(
            action_name=self.action_names[action_index],
            action_index=action_index,
            probabilities=probabilities,
            log_prob=math.log(prob),
            log_probs=make_log_prob_tensor([math.log(prob)]),
            state_features=features,
        )

    def probabilities(self, state_features: Sequence[float]) -> np.ndarray:
        features = np.asarray(state_features, dtype=float)
        logits = features @ self.weights + self.bias
        logits -= np.max(logits)
        exp_logits = np.exp(logits)
        total = float(exp_logits.sum())
        if math.isclose(total, 0.0):
            return np.full(len(self.action_names), 1.0 / len(self.action_names))
        return exp_logits / total

    def update(
        self,
        state_features: Sequence[float],
        action_index: int,
        advantage: float,
    ) -> float:
        features = np.asarray(state_features, dtype=float)
        probs = self.probabilities(features)
        one_hot = np.zeros_like(probs)
        one_hot[int(action_index)] = 1.0
        grad_logits = one_hot - probs
        self.weights += self.learning_rate * advantage * np.outer(features, grad_logits)
        self.bias += self.learning_rate * advantage * grad_logits
        return float(probs[int(action_index)])


def summarize_log_probs(log_probs: object) -> float:
    if torch is not None and isinstance(log_probs, torch.Tensor):
        return float(log_probs.sum().item())
    array = np.asarray(log_probs, dtype=float)
    return float(array.sum())
