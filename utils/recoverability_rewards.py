"""
Heuristic recoverability-aware reward: dense signal from prefix recoverability deltas,
belief-style scores, failure-motif penalties, and peer salvageability aggregation.

Extension point: replace SimpleRecoverabilityEstimator.estimate with verifier rollouts later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence
import re

from .rewards import compute_ground_truth_reward, safe_mean


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    return numerator / denominator if abs(denominator) > 1e-12 else default


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def extract_final_number(text: str) -> Optional[float]:
    matches = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def extract_reasoning_steps(reasoning: str) -> List[str]:
    """Split reasoning into steps: prefer lines, else sentences, else whole text."""
    text = (reasoning or "").strip()
    if not text:
        return []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) >= 2:
        return lines
    if len(lines) == 1 and len(lines[0]) > 200:
        parts = re.split(r"(?<=[.!?])\s+", lines[0])
        out = [p.strip() for p in parts if p.strip()]
        if len(out) >= 2:
            return out
        return [lines[0]]
    if lines:
        return lines
    parts = re.split(r"(?<=[.!?])\s+", text)
    out = [p.strip() for p in parts if p.strip()]
    return out if out else [text]


def extract_hypothesis_count(text: str) -> int:
    """Soft proxy for maintained branches: OR-clauses, 'alternatively', numbered options."""
    t = text.lower()
    n = 0
    if " alternatively " in t or t.startswith("alternatively"):
        n += 1
    n += t.count(" or ")
    n += len(re.findall(r"\b(?:either|option)\b", t))
    n += len(re.findall(r"\n\s*\d+[\).]\s", text))
    return max(1, min(5, 1 + n // 2))


@dataclass
class StepState:
    step_index: int
    text: str
    recoverability: float
    belief_score: float
    failure_flag: float
    delta_recoverability: float


@dataclass
class RewardBreakdown:
    final_correctness: float
    step_progress_reward: float
    belief_reward: float
    failure_penalty: float
    peer_recoverability_reward: float
    branch_bonus: float
    total_reward: float
    step_states: List[StepState] = field(default_factory=list)


class FailureMotifMemory:
    """Stores recurring failure patterns (lexical v1; replace with embeddings later)."""

    def __init__(self, motifs: Optional[Sequence[str]] = None, *, max_motifs: int = 128):
        self.motifs: List[str] = list(motifs or [])
        self.max_motifs = max(1, int(max_motifs))

    def add_motif(self, motif: str) -> None:
        m = motif.strip().lower()
        if not m or len(m) < 3:
            return
        if m in self.motifs:
            return
        self.motifs.append(m)
        if len(self.motifs) > self.max_motifs:
            self.motifs = self.motifs[-self.max_motifs :]

    def match_score(self, step_text: str) -> float:
        text = step_text.lower()
        if not self.motifs:
            return 0.0
        hits = sum(1 for motif in self.motifs if motif and motif in text)
        return clamp01(hits / max(1, len(self.motifs)))


class SimpleRecoverabilityEstimator:
    """Lightweight heuristic recoverability u_t in [0, 1]."""

    def __init__(
        self,
        base_recoverability: float = 0.5,
        progress_bonus: float = 0.10,
        error_penalty_scale: float = 0.15,
    ):
        self.base_recoverability = base_recoverability
        self.progress_bonus = progress_bonus
        self.error_penalty_scale = error_penalty_scale

    def estimate(
        self,
        prefix_steps: Sequence[str],
        ground_truth: str,
        failure_memory: Optional[FailureMotifMemory] = None,
    ) -> float:
        if not prefix_steps:
            return self.base_recoverability

        joined = " ".join(prefix_steps).lower()
        gt_val = extract_final_number(ground_truth)
        pred_val = extract_final_number(joined)

        score = self.base_recoverability

        reasoning_terms = ["therefore", "so", "thus", "because", "then", "hence"]
        found_terms = sum(1 for term in reasoning_terms if term in joined)
        score += self.progress_bonus * min(found_terms, 3)

        if gt_val is not None and pred_val is not None:
            diff = abs(pred_val - gt_val)
            score -= self.error_penalty_scale * min(diff, 3.0)

        if failure_memory is not None:
            fail_score = failure_memory.match_score(joined)
            score -= 0.25 * fail_score

        return clamp01(score)


class SimpleBeliefSetScorer:
    """Soft proxy for whether plausible correct paths remain."""

    def __init__(self, branch_coef: float = 0.04):
        self.branch_coef = branch_coef

    def score(
        self,
        prefix_steps: Sequence[str],
        ground_truth: str,
        failure_memory: Optional[FailureMotifMemory] = None,
    ) -> float:
        if not prefix_steps:
            return 1.0

        text = " ".join(prefix_steps).lower()
        numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
        has_ops = any(op in text for op in ["+", "-", "*", "/", "="])

        score = 0.5
        if has_ops:
            score += 0.2
        if len(numbers) >= 2:
            score += 0.2

        contradictions = text.count("final answer")
        if contradictions > 1:
            score -= 0.25

        if failure_memory is not None:
            score -= 0.2 * failure_memory.match_score(text)

        branches = extract_hypothesis_count(" ".join(prefix_steps))
        score += self.branch_coef * min(branches - 1, 3)

        return clamp01(score)


class RecoverabilityReward:
    """
    R = alpha * R_final
        + beta * sum_t (u_t - u_{t-1})
        + gamma * mean_t(b_t)
        - delta * mean_t(f_t)
        + eta * peer_recoverability
        + zeta * branch_bonus
    """

    def __init__(
        self,
        alpha: float = 0.60,
        beta: float = 0.25,
        gamma: float = 0.10,
        delta: float = 0.05,
        eta: float = 0.10,
        zeta: float = 0.05,
        recoverability_estimator: Optional[SimpleRecoverabilityEstimator] = None,
        belief_scorer: Optional[SimpleBeliefSetScorer] = None,
    ):
        assert alpha >= 0 and beta >= 0 and gamma >= 0 and delta >= 0 and eta >= 0 and zeta >= 0
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.eta = eta
        self.zeta = zeta
        self.recoverability_estimator = recoverability_estimator or SimpleRecoverabilityEstimator()
        self.belief_scorer = belief_scorer or SimpleBeliefSetScorer()

    @staticmethod
    def final_correctness(answer_text: str, ground_truth: str, task_type: str = "exact") -> float:
        return float(compute_ground_truth_reward(answer_text, ground_truth, task_type))

    def compute_step_states(
        self,
        steps: Sequence[str],
        ground_truth: str,
        failure_memory: Optional[FailureMotifMemory] = None,
    ) -> List[StepState]:
        states: List[StepState] = []
        prev_u = self.recoverability_estimator.estimate([], ground_truth, failure_memory)

        prefix: List[str] = []
        for idx, step in enumerate(steps):
            prefix.append(step)

            u_t = self.recoverability_estimator.estimate(prefix, ground_truth, failure_memory)
            b_t = self.belief_scorer.score(prefix, ground_truth, failure_memory)
            f_t = failure_memory.match_score(step) if failure_memory is not None else 0.0
            delta_u = u_t - prev_u

            states.append(
                StepState(
                    step_index=idx,
                    text=step,
                    recoverability=u_t,
                    belief_score=b_t,
                    failure_flag=f_t,
                    delta_recoverability=delta_u,
                )
            )
            prev_u = u_t

        return states

    @staticmethod
    def aggregate_peer_recoverability(
        peer_recoverabilities: Optional[Sequence[float]] = None,
        peer_trusts: Optional[Sequence[float]] = None,
    ) -> float:
        if not peer_recoverabilities:
            return 0.0
        pr = list(peer_recoverabilities)
        if not peer_trusts or len(peer_trusts) != len(pr):
            return clamp01(float(safe_mean(pr, default=0.0)))
        weighted_sum = 0.0
        weight_total = 0.0
        for u_j, t_j in zip(pr, peer_trusts):
            t_j = clamp01(float(t_j))
            u_j = clamp01(float(u_j))
            weighted_sum += t_j * u_j
            weight_total += t_j
        return clamp01(safe_div(weighted_sum, weight_total, default=0.0))

    def branch_bonus_from_steps(self, steps: Sequence[str]) -> float:
        if not steps:
            return 0.0
        full = " ".join(steps)
        b = extract_hypothesis_count(full)
        return clamp01(0.2 * min(b - 1, 3))

    def compute_reward(
        self,
        final_answer: str,
        steps: Sequence[str],
        ground_truth: str,
        task_type: str = "exact",
        failure_memory: Optional[FailureMotifMemory] = None,
        peer_recoverabilities: Optional[Sequence[float]] = None,
        peer_trusts: Optional[Sequence[float]] = None,
        aggregated_peer_recoverability: Optional[float] = None,
    ) -> RewardBreakdown:
        step_states = self.compute_step_states(
            steps=steps,
            ground_truth=ground_truth,
            failure_memory=failure_memory,
        )

        final_correct = self.final_correctness(final_answer, ground_truth, task_type)

        step_progress_reward = sum(s.delta_recoverability for s in step_states)
        belief_reward = sum(s.belief_score for s in step_states) / max(1, len(step_states))
        failure_penalty = sum(s.failure_flag for s in step_states) / max(1, len(step_states))

        if aggregated_peer_recoverability is not None:
            peer_recoverability_reward = clamp01(float(aggregated_peer_recoverability))
        else:
            peer_recoverability_reward = self.aggregate_peer_recoverability(
                peer_recoverabilities=peer_recoverabilities,
                peer_trusts=peer_trusts,
            )

        branch_bonus = self.branch_bonus_from_steps(steps)

        total = (
            self.alpha * final_correct
            + self.beta * step_progress_reward
            + self.gamma * belief_reward
            - self.delta * failure_penalty
            + self.eta * peer_recoverability_reward
            + self.zeta * branch_bonus
        )

        total = clamp01(total)

        return RewardBreakdown(
            final_correctness=final_correct,
            step_progress_reward=step_progress_reward,
            belief_reward=belief_reward,
            failure_penalty=failure_penalty,
            peer_recoverability_reward=peer_recoverability_reward,
            branch_bonus=branch_bonus,
            total_reward=total,
            step_states=step_states,
        )


def extract_failure_motif_snippet(reasoning: str, max_len: int = 72) -> Optional[str]:
    """Derive a short motif string from failed reasoning for negative memory."""
    text = (reasoning or "").strip()
    if not text:
        return None
    line = text.splitlines()[0].strip()
    if len(line) > max_len:
        line = line[:max_len]
    return line.lower() if len(line) >= 8 else None


try:
    import torch

    class PolicyGradientLoss(torch.nn.Module):
        """REINFORCE-style loss: -advantage * sum(log_probs)."""

        def forward(
            self,
            log_probs: "torch.Tensor",
            reward: float,
            baseline: float = 0.0,
        ) -> "torch.Tensor":
            if log_probs.ndim != 1:
                raise ValueError("log_probs must be a 1D tensor of log probabilities.")
            advantage = reward - baseline
            return -advantage * log_probs.sum()

except ImportError:  # pragma: no cover
    PolicyGradientLoss = None  # type: ignore[misc, assignment]
