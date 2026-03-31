from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, TypeVar
import math
import random

import numpy as np

from agents import Agent
from utils import AttentionModule, EpisodeLogger, compute_ground_truth_reward, normalize_scores, safe_mean

T = TypeVar("T")


@dataclass
class EvaluatorDiagnostics:
    evaluator: str
    raw_scores: List[Optional[float]]
    normalized_scores: List[Optional[float]]
    weight: float
    trust_weight: float
    anti_collusion_weight: float
    historical_trust: float
    flat_penalty_applied: bool
    disagreement_penalty: float
    normalization_preserved_ranking: bool
    self_evaluation_blocked: bool


class MultiAgentEnvironment:
    """Core MARL environment for response generation, attention-weighted peer evaluation, and RL updates."""

    def __init__(
        self,
        agents: Sequence[Agent],
        alpha: float = 0.8,
        task_type: str = "exact",
        seed: Optional[int] = None,
        logger: Optional[EpisodeLogger] = None,
        flat_score_weight: float = 0.35,
        disagreement_penalty_scale: float = 0.6,
        minimum_evaluator_weight: float = 0.1,
        max_concurrency: Optional[int] = None,
        revision_rounds: int = 0,
        continue_on_agent_error: bool = True,
        use_trust_weighting: bool = True,
        historical_trust_blend: float = 0.5,
        trust_floor: float = 0.1,
        use_rl: bool = False,
        use_attention: bool = True,
        attention_top_k: int = 2,
        attention_temperature: float = 1.0,
        attention_entropy_coef: float = 0.05,
    ) -> None:
        if not agents:
            raise ValueError("At least one agent is required.")
        if not 0.7 <= alpha <= 0.9:
            raise ValueError("alpha must be within [0.7, 0.9].")

        self.agents = list(agents)
        self.alpha = alpha
        self.task_type = task_type
        self.seed = seed
        self.rng = random.Random(seed)
        self.logger = logger or EpisodeLogger()
        self.flat_score_weight = flat_score_weight
        self.disagreement_penalty_scale = disagreement_penalty_scale
        self.minimum_evaluator_weight = minimum_evaluator_weight
        self.max_concurrency = max_concurrency
        self.revision_rounds = max(0, revision_rounds)
        self.continue_on_agent_error = continue_on_agent_error
        self.use_trust_weighting = use_trust_weighting
        self.historical_trust_blend = historical_trust_blend
        self.trust_floor = trust_floor
        self.use_rl = use_rl
        self.use_attention = use_attention
        self.attention_top_k = max(1, int(attention_top_k))
        self.attention_entropy_coef = max(0.0, float(attention_entropy_coef))
        self.attention_module = AttentionModule(temperature=attention_temperature)
        self.history: List[Dict[str, object]] = []

    def step(
        self,
        question: str,
        ground_truth: str,
        *,
        apply_updates: bool = True,
        task_id: Optional[str] = None,
    ) -> Dict[str, object]:
        agent_failures: List[Dict[str, object]] = []
        initial_responses, generation_failures = self._collect_responses(question)
        agent_failures.extend(generation_failures)

        initial_round = self._score_round(question, ground_truth, initial_responses)
        agent_failures.extend(initial_round["failures"])

        final_responses = initial_responses
        final_round = initial_round
        revision_history: List[Dict[str, object]] = []

        for revision_index in range(self.revision_rounds):
            revised_responses, revision_failures = self._collect_revisions(
                question=question,
                responses=final_responses,
                peer_scores=final_round["peer_scores"],
                attention_payloads=final_round["attention_payloads"],
            )
            agent_failures.extend(revision_failures)
            revised_round = self._score_round(question, ground_truth, revised_responses)
            agent_failures.extend(revised_round["failures"])
            revision_history.append(
                {
                    "round_index": revision_index + 1,
                    "responses": self._serialize_responses(revised_responses),
                    "rewards": revised_round["rewards"],
                    "ranking": revised_round["ranking"],
                    "attention": revised_round["attention_metadata"],
                    "verification": revised_round["verification"],
                }
            )
            final_responses = revised_responses
            final_round = revised_round

        result = self._build_result(
            question=question,
            ground_truth=ground_truth,
            task_id=task_id,
            initial_responses=initial_responses,
            initial_round=initial_round,
            final_responses=final_responses,
            final_round=final_round,
            revision_history=revision_history,
            agent_failures=agent_failures,
        )

        self.history.append(result)
        self.logger.log(result)

        if apply_updates:
            self._apply_updates(
                question=question,
                ground_truth=ground_truth,
                task_id=task_id,
                initial_responses=initial_responses,
                final_responses=final_responses,
                final_round=final_round,
                agent_failures=agent_failures,
                revision_history=revision_history,
            )

        return result

    def run_batch(
        self,
        tasks: Sequence[Dict[str, str]],
        *,
        apply_updates: bool = True,
    ) -> Dict[str, object]:
        results = []
        for task in tasks:
            tid = task.get("task_id")
            results.append(
                self.step(
                    task["question"],
                    task["ground_truth"],
                    apply_updates=apply_updates,
                    task_id=str(tid) if tid is not None else None,
                )
            )

        agent_stats = {}
        for agent in self.agents:
            history = agent.reward_history
            agent_stats[agent.name] = {
                "rounds": len(history),
                "mean_reward": safe_mean(history),
                "total_reward": sum(history),
            }

        return {"results": results, "agent_stats": agent_stats}

    def _build_result(
        self,
        *,
        question: str,
        ground_truth: str,
        task_id: Optional[str],
        initial_responses: List[Dict[str, Any]],
        initial_round: Dict[str, Any],
        final_responses: List[Dict[str, Any]],
        final_round: Dict[str, Any],
        revision_history: List[Dict[str, object]],
        agent_failures: List[Dict[str, object]],
    ) -> Dict[str, object]:
        gt_rewards = final_round["gt_rewards"]
        diagnostics = final_round["diagnostics"]
        rewards = final_round["rewards"]
        ranking = final_round["ranking"]
        gt_has_variance = final_round["gt_has_variance"]

        initial_gt_accuracy = {
            response["agent"]: float(gt)
            for response, gt in zip(self._serialize_responses(initial_responses), initial_round["gt_rewards"])
        }
        final_gt_accuracy = {
            response["agent"]: float(gt)
            for response, gt in zip(self._serialize_responses(final_responses), gt_rewards)
        }

        return {
            "question": question,
            "responses": self._serialize_responses(final_responses),
            "rewards": rewards,
            "ranking": ranking,
            "metadata": {
                "ground_truth": ground_truth,
                "task_type": self.task_type,
                "task_id": task_id,
                "alpha": self.alpha,
                "gt_has_variance": gt_has_variance,
                "aggregation_method": (
                    "strict_order_ground_truth_to_peer_to_trust_to_attention_to_final_reward"
                ),
                "reward_formula": (
                    "R_j = alpha * R_gt_j + (1 - alpha) * R_peer_j, "
                    "with R_peer_j = sum_i(a_ij * t_i * s_ij) / sum_i(a_ij * t_i)"
                ),
                "use_rl": self.use_rl,
                "use_attention": self.use_attention,
                "revision_rounds": self.revision_rounds,
                "agent_failures": agent_failures,
                "reward_verification": final_round["verification"],
                "reward_breakdown": {
                    reward["agent"]: {
                        "gt_reward": reward["gt_reward"],
                        "peer_score": reward["peer_score"],
                        "attention_entropy_bonus": reward.get("attention_entropy_bonus", 0.0),
                        "estimated_advantage": round(
                            float(reward["final_reward"]) - float(getattr(agent, "running_mean_reward", 0.0)),
                            4,
                        ),
                        "final_reward": reward["final_reward"],
                    }
                    for agent, reward in zip(self.agents, rewards)
                },
                "pre_revision_accuracy": initial_gt_accuracy,
                "post_revision_accuracy": final_gt_accuracy,
                "initial_round": {
                    "responses": self._serialize_responses(initial_responses),
                    "rewards": initial_round["rewards"],
                    "ranking": initial_round["ranking"],
                    "attention": initial_round["attention_metadata"],
                    "verification": initial_round["verification"],
                },
                "revision_history": revision_history,
                "peer_evaluations": [
                    {
                        "evaluator": item.evaluator,
                        "raw_scores": item.raw_scores,
                        "normalized_scores": item.normalized_scores,
                        "weight": item.weight,
                        "trust_weight": item.trust_weight,
                        "anti_collusion_weight": item.anti_collusion_weight,
                        "historical_trust": item.historical_trust,
                        "flat_penalty_applied": item.flat_penalty_applied,
                        "disagreement_penalty": item.disagreement_penalty,
                        "normalization_preserved_ranking": item.normalization_preserved_ranking,
                        "self_evaluation_blocked": item.self_evaluation_blocked,
                        "evaluator_alignment": final_round["evaluator_alignment"][item.evaluator],
                    }
                    for item in diagnostics
                ],
                "attention_weights": final_round["attention_metadata"],
            },
        }

    def _apply_updates(
        self,
        *,
        question: str,
        ground_truth: str,
        task_id: Optional[str],
        initial_responses: List[Dict[str, Any]],
        final_responses: List[Dict[str, Any]],
        final_round: Dict[str, Any],
        agent_failures: List[Dict[str, object]],
        revision_history: List[Dict[str, object]],
    ) -> None:
        rewards = final_round["rewards"]
        diagnostics = final_round["diagnostics"]
        attention_meta = final_round["attention_metadata"]
        serialized_final = self._serialize_responses(final_responses)
        serialized_initial = self._serialize_responses(initial_responses)
        peer_eval_meta = [
            {
                "evaluator": item.evaluator,
                "raw_scores": item.raw_scores,
                "normalized_scores": item.normalized_scores,
                "weight": item.weight,
                "trust_weight": item.trust_weight,
                "anti_collusion_weight": item.anti_collusion_weight,
                "historical_trust": item.historical_trust,
                "flat_penalty_applied": item.flat_penalty_applied,
                "disagreement_penalty": item.disagreement_penalty,
                "normalization_preserved_ranking": item.normalization_preserved_ranking,
                "self_evaluation_blocked": item.self_evaluation_blocked,
                "evaluator_alignment": final_round["evaluator_alignment"][item.evaluator],
            }
            for item in diagnostics
        ]
        for agent_index, (agent, reward_dict) in enumerate(zip(self.agents, rewards)):
            diag = diagnostics[agent_index]
            evaluator_alignment = final_round["evaluator_alignment"][agent.name]
            estimated_advantage = float(reward_dict["final_reward"]) - float(getattr(agent, "running_mean_reward", 0.0))
            feedback = {
                "task_id": task_id,
                "question": question,
                "ground_truth": ground_truth,
                "responses": serialized_final,
                "gt_rewards": final_round["gt_rewards"],
                "agent_index": agent_index,
                "own_response": serialized_final[agent_index],
                "initial_response": serialized_initial[agent_index],
                "gt_reward": float(reward_dict["gt_reward"]),
                "peer_score": float(reward_dict["peer_score"]),
                "final_reward": float(reward_dict["final_reward"]),
                "peer_evaluations": peer_eval_meta,
                "evaluator_disagreement": diag.disagreement_penalty,
                "evaluator_flat_penalty": diag.flat_penalty_applied,
                "evaluator_weight": diag.weight,
                "evaluator_trust_weight": diag.trust_weight,
                "evaluator_historical_trust": diag.historical_trust,
                "evaluator_alignment": evaluator_alignment,
                "agent_failures": agent_failures,
                "revision_history": revision_history,
                "attention": attention_meta[agent_index],
                "reward_verification": final_round["verification"],
                "estimated_advantage": estimated_advantage,
            }
            trajectory = {
                "state": question,
                "action": final_responses[agent_index].get("answer", ""),
                "log_probs": final_responses[agent_index].get("log_probs"),
                "reward": float(reward_dict["final_reward"]),
                "embedding": np.asarray(final_responses[agent_index].get("embedding"), dtype=float).tolist(),
                "policy_trace": final_responses[agent_index].get("_trajectory"),
                "feedback": feedback,
                "attention_weights": attention_meta[agent_index].get("weights", []),
                "attention_entropy": attention_meta[agent_index].get("entropy", 0.0),
            }
            agent.update(trajectory)

    def _collect_responses(self, question: str) -> tuple[List[Dict[str, Any]], List[Dict[str, object]]]:
        payloads, failures = self._run_agent_calls(
            lambda _index, agent: agent.generate(question),
            lambda _index, _agent, _exc: self._default_response_payload("", "generation_failed"),
            phase="generate",
        )
        return [self._normalize_response_payload(agent.name, payload) for agent, payload in zip(self.agents, payloads)], failures

    def _collect_peer_scores(
        self,
        question: str,
        responses: List[Dict[str, Any]],
    ) -> tuple[List[List[Optional[float]]], List[Dict[str, object]]]:
        matrix: List[List[Optional[float]]] = []
        score_rows, failures = self._run_agent_calls(
            lambda _index, agent: agent.evaluate(question, self._serialize_responses(responses)),
            lambda _index, _agent, _exc: [0.0 for _ in responses],
            phase="evaluate",
        )
        for agent_index, scores in enumerate(score_rows):
            evaluator_scores: List[Optional[float]] = []
            for response_index, score in enumerate(scores):
                if response_index == agent_index:
                    evaluator_scores.append(None)
                    continue
                evaluator_scores.append(self._clamp_score(score))
            matrix.append(evaluator_scores)
        return matrix, failures

    def _collect_revisions(
        self,
        *,
        question: str,
        responses: List[Dict[str, Any]],
        peer_scores: List[float],
        attention_payloads: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, object]]]:
        payloads, failures = self._run_agent_calls(
            lambda index, agent: agent.revise(
                question,
                responses[index],
                responses,
                peer_scores,
                attention_payloads[index]["context"],
                attention_payloads[index]["weights"],
            ),
            lambda index, _agent, _exc: responses[index],
            phase="revise",
        )
        return [self._normalize_response_payload(agent.name, payload) for agent, payload in zip(self.agents, payloads)], failures

    def _score_round(
        self,
        question: str,
        ground_truth: str,
        responses: List[Dict[str, Any]],
    ) -> Dict[str, object]:
        gt_rewards = self._compute_ground_truth_rewards(responses, ground_truth)
        raw_peer_matrix, failures = self._collect_peer_scores(question, responses)
        diagnostics = self._normalize_peer_scores(raw_peer_matrix)
        evaluator_alignment = self._apply_trust_weighting(diagnostics, gt_rewards)
        peer_scores, attention_metadata, attention_payloads = self._aggregate_peer_scores(diagnostics, responses)
        rewards = self._compute_final_rewards(gt_rewards, peer_scores, attention_metadata)
        ranking = self._rank_agents(rewards)
        gt_has_variance = not math.isclose(max(gt_rewards), min(gt_rewards))
        verification = self._build_verification_report(
            responses=responses,
            gt_rewards=gt_rewards,
            raw_peer_matrix=raw_peer_matrix,
            diagnostics=diagnostics,
            attention_metadata=attention_metadata,
            peer_scores=peer_scores,
            rewards=rewards,
            ranking=ranking,
            gt_has_variance=gt_has_variance,
        )
        return {
            "gt_rewards": gt_rewards,
            "diagnostics": diagnostics,
            "peer_scores": peer_scores,
            "rewards": rewards,
            "ranking": ranking,
            "gt_has_variance": gt_has_variance,
            "evaluator_alignment": evaluator_alignment,
            "failures": failures,
            "attention_metadata": attention_metadata,
            "attention_payloads": attention_payloads,
            "verification": verification,
        }

    def _compute_ground_truth_rewards(
        self,
        responses: List[Dict[str, Any]],
        ground_truth: str,
    ) -> List[float]:
        return [
            compute_ground_truth_reward(str(response["answer"]), ground_truth, self.task_type)
            for response in responses
        ]

    def _normalize_peer_scores(
        self,
        peer_matrix: List[List[Optional[float]]],
    ) -> List[EvaluatorDiagnostics]:
        diagnostics: List[EvaluatorDiagnostics] = []

        for agent_index, (agent, row) in enumerate(zip(self.agents, peer_matrix)):
            visible_scores = [score for score in row if score is not None]
            flat_penalty_applied = self._all_scores_identical(visible_scores)
            normalized_visible = normalize_scores(visible_scores)
            ranking_preserved = self._ranking_preserved(visible_scores, normalized_visible)

            normalized_row: List[Optional[float]] = []
            visible_index = 0
            for score in row:
                if score is None:
                    normalized_row.append(None)
                else:
                    normalized_row.append(normalized_visible[visible_index])
                    visible_index += 1

            diagnostics.append(
                EvaluatorDiagnostics(
                    evaluator=agent.name,
                    raw_scores=list(row),
                    normalized_scores=normalized_row,
                    weight=1.0,
                    trust_weight=1.0,
                    anti_collusion_weight=1.0,
                    historical_trust=safe_mean(getattr(agent, "evaluator_alignment_history", []), default=0.5),
                    flat_penalty_applied=flat_penalty_applied,
                    disagreement_penalty=0.0,
                    normalization_preserved_ranking=ranking_preserved,
                    self_evaluation_blocked=row[agent_index] is None,
                )
            )
        return diagnostics

    def _apply_trust_weighting(
        self,
        diagnostics: List[EvaluatorDiagnostics],
        gt_rewards: List[float],
    ) -> Dict[str, Optional[float]]:
        gt_has_variance = not math.isclose(max(gt_rewards), min(gt_rewards))
        evaluator_alignment: Dict[str, Optional[float]] = {}

        for diag in diagnostics:
            normalized_visible = [score for score in diag.normalized_scores if score is not None]
            anti_collusion_weight = 1.0
            if diag.flat_penalty_applied:
                anti_collusion_weight *= self.flat_score_weight

            disagreement_penalty = 0.0
            alignment: Optional[float] = None
            if gt_has_variance and normalized_visible:
                aligned_gt = [gt_rewards[index] for index, score in enumerate(diag.normalized_scores) if score is not None]
                mean_abs_error = safe_mean(
                    abs(score - gt_score)
                    for score, gt_score in zip(normalized_visible, aligned_gt)
                )
                disagreement_penalty = self.disagreement_penalty_scale * mean_abs_error
                anti_collusion_weight *= max(self.minimum_evaluator_weight, 1.0 - disagreement_penalty)
                alignment = max(0.0, min(1.0, 1.0 - mean_abs_error))

            trust_weight = anti_collusion_weight
            if self.use_trust_weighting:
                trust_multiplier = (
                    (1.0 - self.historical_trust_blend)
                    + (self.historical_trust_blend * diag.historical_trust)
                )
                trust_weight = max(self.trust_floor, anti_collusion_weight * trust_multiplier)

            diag.anti_collusion_weight = anti_collusion_weight
            diag.disagreement_penalty = disagreement_penalty
            diag.trust_weight = trust_weight
            diag.weight = trust_weight
            evaluator_alignment[diag.evaluator] = alignment

        return evaluator_alignment

    def _aggregate_peer_scores(
        self,
        diagnostics: List[EvaluatorDiagnostics],
        responses: List[Dict[str, Any]],
    ) -> tuple[List[float], List[Dict[str, Any]], List[Dict[str, Any]]]:
        peer_scores: List[float] = []
        attention_metadata: List[Dict[str, Any]] = []
        attention_payloads: List[Dict[str, Any]] = []

        for response_index in range(len(self.agents)):
            query_embedding = np.asarray(responses[response_index].get("embedding"), dtype=float)
            evaluator_scores: List[float] = []
            evaluator_weights: List[float] = []
            evaluator_keys: List[np.ndarray] = []
            evaluator_names: List[str] = []
            peer_context_candidates: List[Dict[str, Any]] = []

            for evaluator_index, diag in enumerate(diagnostics):
                if evaluator_index == response_index:
                    continue
                score = diag.normalized_scores[response_index]
                if score is None:
                    continue
                evaluator_scores.append(float(score))
                evaluator_weights.append(float(diag.weight))
                evaluator_keys.append(np.asarray(responses[evaluator_index].get("embedding"), dtype=float))
                evaluator_names.append(diag.evaluator)
                peer_context_candidates.append(self._serialize_response(responses[evaluator_index]))

            if not evaluator_scores:
                peer_scores.append(0.0)
                attention_metadata.append(
                    {
                        "agent": self.agents[response_index].name,
                        "weights": [],
                        "base_attention": [],
                        "combined_weights": [],
                        "trust_weights": [],
                        "entropy": 0.0,
                        "selected_agents": [],
                    }
                )
                attention_payloads.append({"context": [], "weights": []})
                continue

            if self.use_attention:
                weights = self.attention_module.compute_base_attention(
                    query_embedding,
                    evaluator_keys,
                )
            else:
                weights = np.full(len(evaluator_scores), 1.0 / len(evaluator_scores))

            weights = np.asarray(weights, dtype=float)
            entropy = float(-(weights * np.log(weights + 1e-8)).sum())
            combined = weights * np.asarray(evaluator_weights, dtype=float)
            combined_sum = float(combined.sum())
            if math.isclose(combined_sum, 0.0):
                combined_weights = np.full(len(evaluator_scores), 1.0 / len(evaluator_scores))
            else:
                combined_weights = combined / combined_sum

            base_peer_score = float(sum(weight * score for weight, score in zip(combined_weights, evaluator_scores)))
            peer_scores.append(min(1.0, max(0.0, base_peer_score)))

            ranked_indices = sorted(range(len(combined_weights)), key=lambda idx: float(combined_weights[idx]), reverse=True)
            top_indices = ranked_indices[: min(self.attention_top_k, len(ranked_indices))]
            selected_context = [peer_context_candidates[idx] for idx in top_indices]
            selected_weights = [float(combined_weights[idx]) for idx in top_indices]

            attention_metadata.append(
                {
                    "agent": self.agents[response_index].name,
                    "weights": [round(float(value), 6) for value in weights.tolist()],
                    "base_attention": [round(float(value), 6) for value in weights.tolist()],
                    "combined_weights": [round(float(value), 6) for value in combined_weights.tolist()],
                    "trust_weights": [round(float(value), 6) for value in evaluator_weights],
                    "entropy": round(entropy, 6),
                    "entropy_bonus": 0.0,
                    "peer_scores": [round(float(value), 6) for value in evaluator_scores],
                    "peer_agents": evaluator_names,
                    "selected_agents": [evaluator_names[idx] for idx in top_indices],
                }
            )
            attention_payloads.append(
                {
                    "context": selected_context,
                    "weights": selected_weights,
                }
            )

        return peer_scores, attention_metadata, attention_payloads

    def _compute_final_rewards(
        self,
        gt_rewards: List[float],
        peer_scores: List[float],
        attention_metadata: List[Dict[str, Any]],
    ) -> List[Dict[str, float | str]]:
        rewards = []
        for agent, gt_reward, peer_score, attn in zip(self.agents, gt_rewards, peer_scores, attention_metadata):
            final_reward = (self.alpha * gt_reward) + ((1.0 - self.alpha) * peer_score)
            rewards.append(
                {
                    "agent": agent.name,
                    "gt_reward": round(gt_reward, 4),
                    "peer_score": round(peer_score, 4),
                    "attention_entropy_bonus": round(float(attn.get("entropy_bonus", 0.0)), 4),
                    "final_reward": round(final_reward, 4),
                }
            )
        return rewards

    def _rank_agents(self, rewards: List[Dict[str, float | str]]) -> List[str]:
        sorted_rewards = sorted(
            rewards,
            key=lambda item: (
                -float(item["final_reward"]),
                -float(item["gt_reward"]),
                -float(item["peer_score"]),
                str(item["agent"]),
            ),
        )
        return [str(item["agent"]) for item in sorted_rewards]

    def _build_verification_report(
        self,
        *,
        responses: List[Dict[str, Any]],
        gt_rewards: List[float],
        raw_peer_matrix: List[List[Optional[float]]],
        diagnostics: List[EvaluatorDiagnostics],
        attention_metadata: List[Dict[str, Any]],
        peer_scores: List[float],
        rewards: List[Dict[str, float | str]],
        ranking: List[str],
        gt_has_variance: bool,
    ) -> Dict[str, Any]:
        sanity_checks = self._run_sanity_checks(
            diagnostics=diagnostics,
            attention_metadata=attention_metadata,
            peer_scores=peer_scores,
            rewards=rewards,
        )
        return {
            "stage_order": [
                "ground_truth_reward",
                "raw_peer_scores",
                "score_normalization",
                "trust_weighting",
                "attention_weighting",
                "combined_peer_reward",
                "final_reward",
                "sanity_checks",
            ],
            "ground_truth_reward": [
                {
                    "agent": response["agent"],
                    "answer": response["answer"],
                    "reward": round(float(gt_reward), 4),
                    "is_correct": math.isclose(float(gt_reward), 1.0),
                }
                for response, gt_reward in zip(self._serialize_responses(responses), gt_rewards)
            ],
            "raw_peer_scores": [
                {
                    "evaluator": agent.name,
                    "scores": row,
                    "self_evaluation_blocked": row[index] is None,
                }
                for index, (agent, row) in enumerate(zip(self.agents, raw_peer_matrix))
            ],
            "score_normalization": [
                {
                    "evaluator": diag.evaluator,
                    "raw_scores": diag.raw_scores,
                    "normalized_scores": diag.normalized_scores,
                    "preserves_ranking": diag.normalization_preserved_ranking,
                    "flat_penalty_applied": diag.flat_penalty_applied,
                }
                for diag in diagnostics
            ],
            "trust_weighting": {
                "gt_has_variance": gt_has_variance,
                "evaluators": [
                    {
                        "evaluator": diag.evaluator,
                        "historical_trust": round(float(diag.historical_trust), 4),
                        "anti_collusion_weight": round(float(diag.anti_collusion_weight), 4),
                        "disagreement_penalty": round(float(diag.disagreement_penalty), 4),
                        "trust_weight": round(float(diag.trust_weight), 4),
                    }
                    for diag in diagnostics
                ],
            },
            "attention_weighting": attention_metadata,
            "combined_peer_reward": [
                {
                    "agent": agent.name,
                    "peer_reward": round(float(score), 4),
                }
                for agent, score in zip(self.agents, peer_scores)
            ],
            "final_reward": rewards,
            "final_ranking": ranking,
            "sanity_checks": sanity_checks,
        }

    def _run_sanity_checks(
        self,
        *,
        diagnostics: List[EvaluatorDiagnostics],
        attention_metadata: List[Dict[str, Any]],
        peer_scores: List[float],
        rewards: List[Dict[str, float | str]],
    ) -> Dict[str, Any]:
        attention_probabilities_valid = True
        for item in attention_metadata:
            weights = item.get("weights") or []
            if not weights:
                continue
            if any(float(weight) < 0.0 for weight in weights):
                attention_probabilities_valid = False
                break
            if not math.isclose(sum(float(weight) for weight in weights), 1.0, rel_tol=1e-6, abs_tol=1e-6):
                attention_probabilities_valid = False
                break

        uniform_average_check = None
        if diagnostics and attention_metadata:
            trust_values = [float(diag.trust_weight) for diag in diagnostics]
            all_equal_trust = all(math.isclose(value, trust_values[0]) for value in trust_values[1:])
            all_uniform_attention = all(
                not item.get("weights")
                or all(
                    math.isclose(float(weight), 1.0 / len(item["weights"]), rel_tol=1e-6, abs_tol=1e-6)
                    for weight in item["weights"]
                )
                for item in attention_metadata
            )
            if all_equal_trust and all_uniform_attention:
                simple_means = []
                for response_index in range(len(peer_scores)):
                    visible = [
                        float(diag.normalized_scores[response_index])
                        for evaluator_index, diag in enumerate(diagnostics)
                        if evaluator_index != response_index and diag.normalized_scores[response_index] is not None
                    ]
                    simple_means.append(safe_mean(visible))
                uniform_average_check = all(
                    math.isclose(float(left), float(right), rel_tol=1e-6, abs_tol=1e-6)
                    for left, right in zip(peer_scores, simple_means)
                )

        return {
            "self_evaluation_blocked": all(diag.self_evaluation_blocked for diag in diagnostics),
            "normalization_preserves_ranking": all(diag.normalization_preserved_ranking for diag in diagnostics),
            "attention_probabilities_valid": attention_probabilities_valid,
            "peer_scores_bounded": all(0.0 <= float(score) <= 1.0 for score in peer_scores),
            "final_rewards_bounded": all(0.0 <= float(item["final_reward"]) <= 1.0 for item in rewards),
            "uniform_trust_attention_reduce_to_mean": uniform_average_check,
        }

    def _run_agent_calls(
        self,
        fn: Callable[[int, Agent], T],
        default_factory: Callable[[int, Agent, Exception], T],
        *,
        phase: str,
    ) -> tuple[List[T], List[Dict[str, object]]]:
        if len(self.agents) <= 1:
            return self._run_agent_calls_serial(fn, default_factory, phase=phase)

        requested_workers = self.max_concurrency if self.max_concurrency is not None else len(self.agents)
        max_workers = max(1, min(requested_workers, len(self.agents)))
        if max_workers == 1:
            return self._run_agent_calls_serial(fn, default_factory, phase=phase)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self._safe_agent_call, index, agent, fn, default_factory, phase)
                for index, agent in enumerate(self.agents)
            ]
            resolved = [future.result() for future in futures]
        return [item[0] for item in resolved], [item[1] for item in resolved if item[1] is not None]

    def _run_agent_calls_serial(
        self,
        fn: Callable[[int, Agent], T],
        default_factory: Callable[[int, Agent, Exception], T],
        *,
        phase: str,
    ) -> tuple[List[T], List[Dict[str, object]]]:
        values: List[T] = []
        failures: List[Dict[str, object]] = []
        for index, agent in enumerate(self.agents):
            value, failure = self._safe_agent_call(index, agent, fn, default_factory, phase)
            values.append(value)
            if failure is not None:
                failures.append(failure)
        return values, failures

    def _safe_agent_call(
        self,
        index: int,
        agent: Agent,
        fn: Callable[[int, Agent], T],
        default_factory: Callable[[int, Agent, Exception], T],
        phase: str,
    ) -> tuple[T, Optional[Dict[str, object]]]:
        try:
            return fn(index, agent), None
        except Exception as exc:
            if not self.continue_on_agent_error:
                raise
            return default_factory(index, agent, exc), {
                "phase": phase,
                "agent": agent.name,
                "error": str(exc),
            }

    @staticmethod
    def _all_scores_identical(scores: List[float]) -> bool:
        if not scores:
            return False
        first_score = scores[0]
        return all(math.isclose(score, first_score) for score in scores[1:])

    @staticmethod
    def _ranking_preserved(raw_scores: List[float], normalized_scores: List[float]) -> bool:
        if len(raw_scores) != len(normalized_scores):
            return False
        for left_index in range(len(raw_scores)):
            for right_index in range(left_index + 1, len(raw_scores)):
                raw_left = raw_scores[left_index]
                raw_right = raw_scores[right_index]
                norm_left = normalized_scores[left_index]
                norm_right = normalized_scores[right_index]
                if math.isclose(raw_left, raw_right):
                    continue
                if raw_left < raw_right and not norm_left < norm_right:
                    return False
                if raw_left > raw_right and not norm_left > norm_right:
                    return False
        return True

    @staticmethod
    def _clamp_score(score: float) -> float:
        return max(0.0, min(1.0, float(score)))

    @staticmethod
    def _serialize_log_probs(log_probs: Any) -> List[float]:
        if log_probs is None:
            return [0.0]
        if hasattr(log_probs, "detach") and hasattr(log_probs, "cpu"):
            return [float(value) for value in log_probs.detach().cpu().numpy().tolist()]
        return [float(value) for value in np.asarray(log_probs, dtype=float).tolist()]

    def _default_response_payload(self, answer: str, reasoning: str) -> Dict[str, Any]:
        return {
            "answer": answer,
            "reasoning": reasoning,
            "log_probs": [0.0],
            "embedding": np.zeros(self.agents[0].feature_dim if self.agents else 32, dtype=float),
            "_trajectory": {},
            "policy_action": "fallback",
        }

    def _normalize_response_payload(self, agent_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        embedding = payload.get("embedding")
        if embedding is None:
            embedding = np.zeros(self.agents[0].feature_dim if self.agents else 32, dtype=float)
        else:
            embedding = np.asarray(embedding, dtype=float)
        return {
            "agent": agent_name,
            "answer": str(payload.get("answer", "")).strip(),
            "reasoning": str(payload.get("reasoning", "")).strip(),
            "log_probs": payload.get("log_probs", [0.0]),
            "embedding": embedding,
            "_trajectory": dict(payload.get("_trajectory") or {}),
            "policy_action": str(payload.get("policy_action", "direct")),
            "revision_source": payload.get("revision_source"),
        }

    def _serialize_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        serialized = {
            "agent": response["agent"],
            "answer": response["answer"],
            "reasoning": response["reasoning"],
            "log_probs": self._serialize_log_probs(response.get("log_probs")),
            "embedding": np.asarray(response.get("embedding"), dtype=float).tolist(),
            "policy_action": response.get("policy_action"),
        }
        if response.get("revision_source") is not None:
            serialized["revision_source"] = response.get("revision_source")
        return serialized

    def _serialize_responses(self, responses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self._serialize_response(response) for response in responses]
