from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import random

import numpy as np

from utils import ContextualBanditPolicy, embed_text, make_log_prob_tensor


GENERATION_ACTIONS = ["direct", "deliberate", "skeptical", "concise"]
REVISION_ACTIONS = ["preserve", "peer_weighted", "consistency_check"]


class Agent(ABC):
    """Base interface for pluggable MARL agents."""

    def __init__(
        self,
        name: str,
        seed: Optional[int] = None,
        *,
        use_rl: bool = False,
        rl_learning_rate: float = 0.05,
        reward_clip: float = 1.0,
        baseline_momentum: float = 0.9,
        feature_dim: int = 32,
    ) -> None:
        self.name = name
        self.seed = seed
        self.rng = random.Random(seed)
        self.reward_history: List[float] = []
        self.generation_memory: List[Dict[str, Any]] = []
        self.eval_memory: List[Dict[str, Any]] = []
        self.evaluator_alignment_history: List[float] = []
        self.failure_motifs: List[str] = []

        self.use_rl = use_rl
        self.reward_clip = max(0.0, float(reward_clip))
        self.baseline_momentum = min(max(float(baseline_momentum), 0.0), 0.9999)
        self.running_mean_reward = 0.0
        self.running_reward_var = 1.0
        self.update_calls = 0
        self.last_advantage = 0.0
        self.last_reward = 0.0

        self.feature_dim = int(feature_dim)
        self.generation_policy = ContextualBanditPolicy(
            GENERATION_ACTIONS,
            feature_dim=self.feature_dim,
            learning_rate=rl_learning_rate,
            seed=seed,
        )
        self.revision_policy = ContextualBanditPolicy(
            REVISION_ACTIONS,
            feature_dim=self.feature_dim,
            learning_rate=rl_learning_rate,
            seed=None if seed is None else seed + 10_000,
        )

    @abstractmethod
    def generate(self, question: str) -> Dict[str, Any]:
        """Return answer/reasoning plus optional RL metadata."""

    @abstractmethod
    def evaluate(self, question: str, responses: List[Dict[str, Any]]) -> List[float]:
        """Return one score per response in the same order as `responses`."""

    def revise(
        self,
        question: str,
        own_response: Dict[str, Any],
        responses: List[Dict[str, Any]],
        peer_scores: Optional[List[float]] = None,
        attention_context: Optional[List[Dict[str, Any]]] = None,
        attention_weights: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Optional revision hook for multi-round interaction. Defaults to keeping the current answer."""
        del question, responses, peer_scores, attention_context, attention_weights
        return {
            "answer": str(own_response.get("answer", "")).strip(),
            "reasoning": str(own_response.get("reasoning", "")).strip(),
            "log_probs": own_response.get("log_probs", make_log_prob_tensor([0.0])),
            "embedding": self._coerce_embedding(own_response.get("embedding")),
            "_trajectory": dict(own_response.get("_trajectory") or {}),
            "policy_action": own_response.get("policy_action", "preserve"),
        }

    def update(self, trajectory: Dict[str, Any]) -> None:
        """Contextual-bandit REINFORCE update over prompt/revision actions."""
        reward = float(trajectory.get("reward", 0.0))
        self.reward_history.append(reward)
        self.last_reward = reward

        feedback = trajectory.get("feedback")
        if isinstance(feedback, dict):
            align = feedback.get("evaluator_alignment")
            if isinstance(align, (int, float)):
                self.evaluator_alignment_history.append(float(align))
            motifs = feedback.get("new_failure_motifs")
            if isinstance(motifs, list):
                for m in motifs:
                    if isinstance(m, str) and m and m not in self.failure_motifs:
                        self.failure_motifs.append(m)
                        if len(self.failure_motifs) > 64:
                            self.failure_motifs = self.failure_motifs[-64:]

        if not self.use_rl:
            return

        trace = trajectory.get("policy_trace")
        if not isinstance(trace, dict):
            return
        policy_name = str(trace.get("policy", "generation"))
        state_features = trace.get("state_features")
        action_index = trace.get("action_index")
        if not isinstance(state_features, list) or not isinstance(action_index, int):
            return

        clipped_reward = max(-self.reward_clip, min(self.reward_clip, reward)) if self.reward_clip > 0 else reward
        advantage = clipped_reward - self.running_mean_reward
        reward_delta = clipped_reward - self.running_mean_reward
        self.running_mean_reward = (
            (self.baseline_momentum * self.running_mean_reward)
            + ((1.0 - self.baseline_momentum) * clipped_reward)
        )
        self.running_reward_var = (
            (self.baseline_momentum * self.running_reward_var)
            + ((1.0 - self.baseline_momentum) * (reward_delta ** 2))
        )
        reward_std = max(1e-6, float(self.running_reward_var) ** 0.5)
        normalized_advantage = advantage / reward_std
        self.last_advantage = normalized_advantage
        self.update_calls += 1

        policy = self.generation_policy if policy_name == "generation" else self.revision_policy
        policy.update(state_features=state_features, action_index=action_index, advantage=normalized_advantage)

    def _sample_generation_policy(self, question: str, memory_text: str = "") -> Dict[str, Any]:
        state_text = "\n".join(part for part in [question, memory_text] if part)
        return self._sample_policy(self.generation_policy, state_text, policy_name="generation")

    def _sample_revision_policy(
        self,
        question: str,
        own_response: Dict[str, Any],
        attention_context: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        context_summary = " ".join(
            str(item.get("answer", ""))
            for item in (attention_context or [])
        )
        state_text = "\n".join(
            part
            for part in [question, str(own_response.get("answer", "")), context_summary]
            if part
        )
        return self._sample_policy(self.revision_policy, state_text, policy_name="revision")

    def _sample_policy(
        self,
        policy: ContextualBanditPolicy,
        state_text: str,
        *,
        policy_name: str,
    ) -> Dict[str, Any]:
        sample = policy.sample(embed_text(state_text, dim=self.feature_dim))
        return {
            "policy": policy_name,
            "action": sample.action_name,
            "action_index": sample.action_index,
            "state_features": sample.state_features.tolist(),
            "probabilities": sample.probabilities.tolist(),
            "log_prob": sample.log_prob,
            "log_probs": sample.log_probs,
        }

    def _build_response_payload(
        self,
        answer: str,
        reasoning: str,
        *,
        policy_trace: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        embedding = embed_text(f"{answer}\n{reasoning}", dim=self.feature_dim)
        payload: Dict[str, Any] = {
            "answer": str(answer).strip(),
            "reasoning": str(reasoning).strip(),
            "log_probs": (policy_trace or {}).get("log_probs", make_log_prob_tensor([0.0])),
            "embedding": embedding,
            "_trajectory": policy_trace or {},
            "policy_action": (policy_trace or {}).get("action", "direct"),
        }
        if extra:
            payload.update(extra)
        return payload

    def _coerce_embedding(self, value: Any) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        if isinstance(value, list):
            return np.asarray(value, dtype=float)
        if value is None:
            return np.zeros(self.feature_dim, dtype=float)
        return np.asarray(value, dtype=float)
