from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from environment import MultiAgentEnvironment

EXPECTED_VERIFICATION_STAGE_ORDER = [
    "ground_truth_reward",
    "raw_peer_salvageability_scores",
    "score_normalization",
    "trust_weighting",
    "attention_weighting",
    "combined_peer_recoverability",
    "recoverability_components",
    "final_recoverability_reward",
    "sanity_checks",
]


def load_completed_task_ids(results_path: str | Path) -> Set[str]:
    path = Path(results_path)
    if not path.exists():
        return set()

    completed: Set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            task_id = record.get("task_id")
            if task_id is not None:
                completed.add(str(task_id))
    return completed


def compute_summary_metrics(records: Sequence[Dict[str, object]]) -> Dict[str, object]:
    total_examples = len(records)
    if total_examples == 0:
        return {
            "num_examples": 0,
            "agent_metrics": {},
            "leaderboard": [],
            "pairwise_win_matrix": {},
            "verification_metrics": {},
        }

    agent_names = [reward["agent"] for reward in records[0]["rewards"]]
    agent_metrics: Dict[str, Dict[str, float]] = {
        agent: {
            "mean_gt_reward": 0.0,
            "mean_peer_score": 0.0,
            "mean_final_reward": 0.0,
            "gt_accuracy": 0.0,
            "wins": 0.0,
            "failure_count": 0.0,
            "mean_revision_gain": 0.0,
            "mean_evaluator_alignment": 0.0,
            "alignment_count": 0.0,
            "mean_trust_weight": 0.0,
            "trust_count": 0.0,
            "mean_attention_entropy": 0.0,
            "attention_count": 0.0,
            "mean_attention_bonus": 0.0,
            "mean_estimated_advantage": 0.0,
            "advantage_count": 0.0,
            "pre_revision_accuracy": 0.0,
            "post_revision_accuracy": 0.0,
            "mean_step_progress_reward": 0.0,
            "mean_belief_reward": 0.0,
            "mean_failure_penalty": 0.0,
            "mean_branch_bonus": 0.0,
            "mean_peer_recoverability_reward": 0.0,
        }
        for agent in agent_names
    }
    pairwise_win_matrix: Dict[str, Dict[str, float]] = {
        agent: {other: 0.0 for other in agent_names if other != agent}
        for agent in agent_names
    }
    verification_totals = {
        "stage_order_matches_expected": 0.0,
        "self_evaluation_blocked": 0.0,
        "normalization_preserves_ranking": 0.0,
        "attention_probabilities_valid": 0.0,
        "peer_scores_bounded": 0.0,
        "final_rewards_bounded": 0.0,
    }

    for record in records:
        reward_map = {reward["agent"]: reward for reward in record["rewards"]}
        ranking = record["ranking"]
        metadata = record.get("metadata") or {}
        initial_round = metadata.get("initial_round") or {}
        initial_reward_map = {
            reward["agent"]: reward
            for reward in (initial_round.get("rewards") or [])
        }
        failures = metadata.get("agent_failures") or []
        failed_agents = {str(item.get("agent")) for item in failures}
        peer_meta = metadata.get("peer_evaluations") or []
        alignment_map = {
            str(item.get("evaluator")): item.get("evaluator_alignment")
            for item in peer_meta
        }
        trust_map = {
            str(item.get("evaluator")): item.get("trust_weight")
            for item in peer_meta
        }
        attention_map = {
            str(item.get("agent")): item
            for item in (metadata.get("attention_weights") or [])
        }
        reward_breakdown = metadata.get("reward_breakdown") or {}
        pre_revision = metadata.get("pre_revision_accuracy") or {}
        post_revision = metadata.get("post_revision_accuracy") or {}
        verification = metadata.get("reward_verification") or {}
        sanity_checks = verification.get("sanity_checks") or {}
        if verification.get("stage_order") == EXPECTED_VERIFICATION_STAGE_ORDER:
            verification_totals["stage_order_matches_expected"] += 1.0
        for key in (
            "self_evaluation_blocked",
            "normalization_preserves_ranking",
            "attention_probabilities_valid",
            "peer_scores_bounded",
            "final_rewards_bounded",
        ):
            if bool(sanity_checks.get(key)):
                verification_totals[key] += 1.0
        for agent in agent_names:
            reward = reward_map[agent]
            agent_metrics[agent]["mean_gt_reward"] += float(reward["gt_reward"])
            agent_metrics[agent]["mean_peer_score"] += float(reward["peer_score"])
            agent_metrics[agent]["mean_final_reward"] += float(reward["final_reward"])
            if "step_progress_reward" in reward:
                agent_metrics[agent]["mean_step_progress_reward"] += float(reward["step_progress_reward"])
            if "belief_reward" in reward:
                agent_metrics[agent]["mean_belief_reward"] += float(reward["belief_reward"])
            if "failure_penalty" in reward:
                agent_metrics[agent]["mean_failure_penalty"] += float(reward["failure_penalty"])
            if "branch_bonus" in reward:
                agent_metrics[agent]["mean_branch_bonus"] += float(reward["branch_bonus"])
            if "peer_recoverability_reward" in reward:
                agent_metrics[agent]["mean_peer_recoverability_reward"] += float(
                    reward["peer_recoverability_reward"]
                )
            agent_metrics[agent]["gt_accuracy"] += 1.0 if math.isclose(float(reward["gt_reward"]), 1.0) else 0.0
            agent_metrics[agent]["pre_revision_accuracy"] += float(pre_revision.get(agent, 0.0))
            agent_metrics[agent]["post_revision_accuracy"] += float(post_revision.get(agent, 0.0))
            if agent in failed_agents:
                agent_metrics[agent]["failure_count"] += 1.0
            if agent in initial_reward_map:
                agent_metrics[agent]["mean_revision_gain"] += (
                    float(reward["final_reward"]) - float(initial_reward_map[agent]["final_reward"])
                )
            alignment = alignment_map.get(agent)
            if isinstance(alignment, (int, float)):
                agent_metrics[agent]["mean_evaluator_alignment"] += float(alignment)
                agent_metrics[agent]["alignment_count"] += 1.0
            trust = trust_map.get(agent)
            if isinstance(trust, (int, float)):
                agent_metrics[agent]["mean_trust_weight"] += float(trust)
                agent_metrics[agent]["trust_count"] += 1.0
            attention_item = attention_map.get(agent)
            if isinstance(attention_item, dict):
                entropy = attention_item.get("entropy")
                if isinstance(entropy, (int, float)):
                    agent_metrics[agent]["mean_attention_entropy"] += float(entropy)
                    agent_metrics[agent]["attention_count"] += 1.0
            reward_item = reward_breakdown.get(agent)
            if isinstance(reward_item, dict):
                attention_bonus = reward_item.get("attention_entropy_bonus")
                if isinstance(attention_bonus, (int, float)):
                    agent_metrics[agent]["mean_attention_bonus"] += float(attention_bonus)
                estimated_advantage = reward_item.get("estimated_advantage")
                if isinstance(estimated_advantage, (int, float)):
                    agent_metrics[agent]["mean_estimated_advantage"] += float(estimated_advantage)
                    agent_metrics[agent]["advantage_count"] += 1.0
        if ranking:
            agent_metrics[str(ranking[0])]["wins"] += 1.0

        for left_agent in agent_names:
            for right_agent in agent_names:
                if left_agent == right_agent:
                    continue
                left_reward = float(reward_map[left_agent]["final_reward"])
                right_reward = float(reward_map[right_agent]["final_reward"])
                if left_reward > right_reward:
                    pairwise_win_matrix[left_agent][right_agent] += 1.0

    leaderboard = []
    for agent in agent_names:
        metrics = agent_metrics[agent]
        metrics["mean_gt_reward"] = round(metrics["mean_gt_reward"] / total_examples, 4)
        metrics["mean_peer_score"] = round(metrics["mean_peer_score"] / total_examples, 4)
        metrics["mean_final_reward"] = round(metrics["mean_final_reward"] / total_examples, 4)
        metrics["gt_accuracy"] = round(metrics["gt_accuracy"] / total_examples, 4)
        metrics["win_rate"] = round(metrics["wins"] / total_examples, 4)
        metrics["failure_rate"] = round(metrics["failure_count"] / total_examples, 4)
        metrics["mean_revision_gain"] = round(metrics["mean_revision_gain"] / total_examples, 4)
        metrics["pre_revision_accuracy"] = round(metrics["pre_revision_accuracy"] / total_examples, 4)
        metrics["post_revision_accuracy"] = round(metrics["post_revision_accuracy"] / total_examples, 4)
        metrics["mean_evaluator_alignment"] = round(
            metrics["mean_evaluator_alignment"] / metrics["alignment_count"], 4
        ) if metrics["alignment_count"] > 0 else None
        metrics["mean_trust_weight"] = round(
            metrics["mean_trust_weight"] / metrics["trust_count"], 4
        ) if metrics["trust_count"] > 0 else None
        metrics["mean_attention_entropy"] = round(
            metrics["mean_attention_entropy"] / metrics["attention_count"], 4
        ) if metrics["attention_count"] > 0 else None
        metrics["mean_attention_bonus"] = round(metrics["mean_attention_bonus"] / total_examples, 4)
        metrics["mean_estimated_advantage"] = round(
            metrics["mean_estimated_advantage"] / metrics["advantage_count"], 4
        ) if metrics["advantage_count"] > 0 else None
        metrics["mean_step_progress_reward"] = round(metrics["mean_step_progress_reward"] / total_examples, 4)
        metrics["mean_belief_reward"] = round(metrics["mean_belief_reward"] / total_examples, 4)
        metrics["mean_failure_penalty"] = round(metrics["mean_failure_penalty"] / total_examples, 4)
        metrics["mean_branch_bonus"] = round(metrics["mean_branch_bonus"] / total_examples, 4)
        metrics["mean_peer_recoverability_reward"] = round(
            metrics["mean_peer_recoverability_reward"] / total_examples, 4
        )
        del metrics["wins"]
        del metrics["failure_count"]
        del metrics["alignment_count"]
        del metrics["trust_count"]
        del metrics["attention_count"]
        del metrics["advantage_count"]
        leaderboard.append(
            {
                "agent": agent,
                "mean_final_reward": metrics["mean_final_reward"],
                "mean_gt_reward": metrics["mean_gt_reward"],
                "gt_accuracy": metrics["gt_accuracy"],
                "win_rate": metrics["win_rate"],
                "failure_rate": metrics["failure_rate"],
                "mean_trust_weight": metrics["mean_trust_weight"],
                "mean_attention_entropy": metrics["mean_attention_entropy"],
                "post_revision_accuracy": metrics["post_revision_accuracy"],
                "mean_step_progress_reward": metrics["mean_step_progress_reward"],
                "mean_belief_reward": metrics["mean_belief_reward"],
                "mean_failure_penalty": metrics["mean_failure_penalty"],
                "mean_branch_bonus": metrics["mean_branch_bonus"],
                "mean_peer_recoverability_reward": metrics["mean_peer_recoverability_reward"],
            }
        )

    leaderboard.sort(
        key=lambda item: (
            -float(item["mean_final_reward"]),
            -float(item["mean_gt_reward"]),
            -float(item["win_rate"]),
            str(item["agent"]),
        )
    )
    verification_metrics = {
        key: round(value / total_examples, 4)
        for key, value in verification_totals.items()
    }

    return {
        "num_examples": total_examples,
        "agent_metrics": agent_metrics,
        "leaderboard": leaderboard,
        "pairwise_win_matrix": pairwise_win_matrix,
        "verification_metrics": verification_metrics,
    }


@dataclass
class ExperimentRunner:
    env: MultiAgentEnvironment
    output_dir: Path
    run_manifest: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_path = self.output_dir / "results.jsonl"
        self.summary_path = self.output_dir / "summary.json"
        self.learning_curve_path = self.output_dir / "learning_curve.jsonl"
        self.manifest_path = self.output_dir / "run_manifest.json"

    def _learning_curve_line_count(self) -> int:
        if not self.learning_curve_path.exists():
            return 0
        count = 0
        with self.learning_curve_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    count += 1
        return count

    def _append_learning_curve(
        self,
        record: Dict[str, object],
        *,
        episode_index: int,
    ) -> None:
        rewards = record.get("rewards") or []
        ranking = record.get("ranking") or []
        winner = str(ranking[0]) if ranking else None
        per_agent: Dict[str, Dict[str, Any]] = {}
        for item in rewards:
            agent_name = str(item["agent"])
            per_agent[agent_name] = {
                "gt_reward": float(item["gt_reward"]),
                "peer_score": float(item["peer_score"]),
                "final_reward": float(item["final_reward"]),
                "won": agent_name == winner,
            }
        alignment: Dict[str, Optional[float]] = {}
        for agent in self.env.agents:
            hist = getattr(agent, "evaluator_alignment_history", None)
            if isinstance(hist, list) and hist:
                alignment[agent.name] = float(hist[-1])
            else:
                alignment[agent.name] = None

        peer_meta = (record.get("metadata") or {}).get("peer_evaluations") or []
        peer_consensus = None
        if peer_meta:
            weights = [float(p.get("weight", 0.0)) for p in peer_meta]
            if weights:
                peer_consensus = round(sum(weights) / len(weights), 4)

        row = {
            "episode_index": episode_index,
            "task_id": record.get("task_id"),
            "per_agent": per_agent,
            "winner": winner,
            "evaluator_alignment": alignment,
            "mean_evaluator_weight": peer_consensus,
        }
        with self.learning_curve_path.open("a", encoding="utf-8") as lc_handle:
            lc_handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def run(
        self,
        tasks: Iterable[Dict[str, str]],
        *,
        resume: bool = False,
        apply_updates: bool = True,
        flush_every: int = 1,
        track_learning_curve: bool = True,
    ) -> Dict[str, object]:
        self._ensure_manifest(resume=resume)
        completed_ids = load_completed_task_ids(self.results_path) if resume else set()
        new_records: List[Dict[str, object]] = []
        processed = 0
        lc_start = self._learning_curve_line_count() if (track_learning_curve and resume) else 0
        if not resume:
            lc_start = 0

        with self.results_path.open("a", encoding="utf-8") as handle:
            for task in tasks:
                task_id = str(task["task_id"])
                if task_id in completed_ids:
                    continue

                episode = self.env.step(
                    question=task["question"],
                    ground_truth=task["ground_truth"],
                    apply_updates=apply_updates,
                    task_id=task_id,
                )
                record = {
                    "task_id": task_id,
                    "source": task.get("source"),
                    "question": task["question"],
                    "ground_truth": task["ground_truth"],
                    "responses": episode["responses"],
                    "rewards": episode["rewards"],
                    "ranking": episode["ranking"],
                    "metadata": episode["metadata"],
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                new_records.append(record)
                processed += 1

                if track_learning_curve:
                    episode_index = lc_start + processed - 1
                    self._append_learning_curve(record, episode_index=episode_index)

                if flush_every > 0 and processed % flush_every == 0:
                    handle.flush()

        all_records = self._read_all_records()
        summary = compute_summary_metrics(all_records)
        self.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        result: Dict[str, object] = {
            "output_dir": str(self.output_dir),
            "results_path": str(self.results_path),
            "summary_path": str(self.summary_path),
            "learning_curve_path": str(self.learning_curve_path) if track_learning_curve else None,
            "manifest_path": str(self.manifest_path),
            "new_examples_processed": processed,
            "total_examples": summary["num_examples"],
            "summary": summary,
        }
        return result

    def _read_all_records(self) -> List[Dict[str, object]]:
        records: List[Dict[str, object]] = []
        if not self.results_path.exists():
            return records
        with self.results_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    records.append(json.loads(stripped))
        return records

    def _ensure_manifest(self, *, resume: bool) -> None:
        manifest = self._build_manifest()
        if not self.manifest_path.exists():
            self.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            return

        existing = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self._manifest_identity(existing) == self._manifest_identity(manifest):
            return

        if resume:
            raise ValueError(
                "Resume requested for an output directory whose run manifest does not match the current configuration. "
                f"Existing manifest: {self.manifest_path}"
            )

        if self.results_path.exists() and self.results_path.stat().st_size > 0:
            raise ValueError(
                "Output directory already contains results from a different configuration. "
                f"Use a new output directory or --resume with the original manifest: {self.manifest_path}"
            )

        self.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def _build_manifest(self) -> Dict[str, Any]:
        user_manifest = dict(self.run_manifest or {})
        return {
            "format_version": 1,
            "created_at_utc": user_manifest.pop("created_at_utc", datetime.now(timezone.utc).isoformat()),
            "environment": {
                "alpha": self.env.alpha,
                "task_type": self.env.task_type,
                "seed": self.env.seed,
                "flat_score_weight": self.env.flat_score_weight,
                "disagreement_penalty_scale": self.env.disagreement_penalty_scale,
                "minimum_evaluator_weight": self.env.minimum_evaluator_weight,
                "max_concurrency": self.env.max_concurrency,
                "use_trust_weighting": self.env.use_trust_weighting,
                "historical_trust_blend": self.env.historical_trust_blend,
                "trust_floor": self.env.trust_floor,
                "revision_rounds": self.env.revision_rounds,
                "continue_on_agent_error": self.env.continue_on_agent_error,
                "use_attention": self.env.use_attention,
                "attention_top_k": self.env.attention_top_k,
                "attention_temperature": self.env.attention_module.temperature,
                "attention_entropy_coef": self.env.attention_entropy_coef,
                "recoverability_beta": self.env.recoverability_beta,
                "recoverability_gamma": self.env.recoverability_gamma,
                "recoverability_delta": self.env.recoverability_delta,
                "recoverability_eta": self.env.recoverability_eta,
                "recoverability_zeta": self.env.recoverability_zeta,
            },
            "agents": [
                {
                    "name": agent.name,
                    "class": agent.__class__.__name__,
                    "model": getattr(agent, "model", None),
                }
                for agent in self.env.agents
            ],
            "run": user_manifest,
        }

    @staticmethod
    def _manifest_identity(manifest: Dict[str, Any]) -> Dict[str, Any]:
        identity = dict(manifest)
        identity.pop("created_at_utc", None)
        return identity
