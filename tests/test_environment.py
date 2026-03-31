from __future__ import annotations

import json
import time
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agents import Agent, ICLOllamaAgent, OllamaAgent, SelfRefineOllamaAgent
from agents.base import GENERATION_ACTIONS
from data import build_gsm8k_tasks_from_rows, extract_gsm8k_final_answer
from experiment import ExperimentRunner, compute_summary_metrics, load_completed_task_ids
from run_demo import DEFAULT_OLLAMA_MODELS, build_heuristic_agents, build_ollama_agents
from run_experiment import chunk_tasks, load_tasks
from environment import MultiAgentEnvironment
from utils import embed_text, extract_json_value

EXPECTED_STAGE_ORDER = [
    "ground_truth_reward",
    "raw_peer_scores",
    "score_normalization",
    "trust_weighting",
    "attention_weighting",
    "combined_peer_reward",
    "final_reward",
    "sanity_checks",
]


class SlowAgent(Agent):
    def __init__(self, name: str, delay: float) -> None:
        super().__init__(name=name, seed=0)
        self.delay = delay

    def generate(self, question: str) -> dict[str, str]:
        time.sleep(self.delay)
        return {"answer": question, "reasoning": self.name}

    def evaluate(self, question: str, responses: list[dict[str, str]]) -> list[float]:
        time.sleep(self.delay)
        return [0.0 if response["agent"] == self.name else 1.0 for response in responses]


class RevisingAgent(Agent):
    def __init__(self, name: str, initial_answer: str, revised_answer: str) -> None:
        super().__init__(name=name, seed=0)
        self.initial_answer = initial_answer
        self.revised_answer = revised_answer

    def generate(self, question: str) -> dict[str, str]:
        del question
        return {"answer": self.initial_answer, "reasoning": "initial"}

    def evaluate(self, question: str, responses: list[dict[str, str]]) -> list[float]:
        del question
        return [0.0 if response["agent"] == self.name else 1.0 for response in responses]

    def revise(
        self,
        question: str,
        own_response: dict[str, str],
        responses: list[dict[str, str]],
        peer_scores: list[float] | None = None,
        attention_context: list[dict[str, str]] | None = None,
        attention_weights: list[float] | None = None,
    ) -> dict[str, str]:
        del question, own_response, responses, peer_scores, attention_context, attention_weights
        return {"answer": self.revised_answer, "reasoning": "revised"}


class FailingAgent(Agent):
    def __init__(self, name: str) -> None:
        super().__init__(name=name, seed=0)

    def generate(self, question: str) -> dict[str, str]:
        del question
        raise RuntimeError("boom-generate")

    def evaluate(self, question: str, responses: list[dict[str, str]]) -> list[float]:
        del question, responses
        raise RuntimeError("boom-evaluate")


class PolicyLearningAgent(Agent):
    def __init__(self, name: str) -> None:
        super().__init__(name=name, seed=0, use_rl=True, rl_learning_rate=0.2)

    def generate(self, question: str) -> dict[str, object]:
        trace = self._sample_generation_policy(question)
        answer = "4" if trace["action"] == "skeptical" else "5"
        return self._build_response_payload(answer, f"mode={trace['action']}", policy_trace=trace)

    def evaluate(self, question: str, responses: list[dict[str, str]]) -> list[float]:
        del question
        return [0.0 if response["agent"] == self.name else 0.5 for response in responses]


class DeterministicAgent(Agent):
    def __init__(
        self,
        name: str,
        *,
        answer: str,
        scores: list[float],
        embedding: list[float] | None = None,
    ) -> None:
        super().__init__(name=name, seed=0)
        self.answer = answer
        self.scores = scores
        self.embedding = embedding or [1.0, 0.0, 0.0]

    def generate(self, question: str) -> dict[str, object]:
        del question
        return {
            "answer": self.answer,
            "reasoning": "fixed",
            "embedding": self.embedding,
            "log_probs": [0.0],
            "_trajectory": {},
            "policy_action": "fixed",
        }

    def evaluate(self, question: str, responses: list[dict[str, str]]) -> list[float]:
        del question, responses
        return list(self.scores)


class MultiAgentEnvironmentTests(unittest.TestCase):
    def test_step_returns_expected_schema(self) -> None:
        env = MultiAgentEnvironment(agents=build_heuristic_agents(), alpha=0.8, task_type="exact", seed=7)
        result = env.step(question="What is 2 + 2?", ground_truth="4")

        self.assertEqual(result["question"], "What is 2 + 2?")
        self.assertEqual(len(result["responses"]), 4)
        self.assertEqual(len(result["rewards"]), 4)
        self.assertEqual(len(result["ranking"]), 4)

        reward_map = {item["agent"]: item for item in result["rewards"]}
        self.assertGreater(reward_map["Qwen-Small"]["final_reward"], reward_map["Kimi-Mini"]["final_reward"])
        self.assertGreater(reward_map["Llama-Tiny"]["final_reward"], reward_map["Mistral-FlatJudge"]["final_reward"])

    def test_flat_evaluator_is_penalized(self) -> None:
        env = MultiAgentEnvironment(agents=build_heuristic_agents(), alpha=0.8, task_type="exact", seed=7)
        result = env.step(question="What is 2 + 2?", ground_truth="4")
        diagnostics = {
            item["evaluator"]: item
            for item in result["metadata"]["peer_evaluations"]
        }

        self.assertTrue(diagnostics["Mistral-FlatJudge"]["flat_penalty_applied"])
        self.assertLess(diagnostics["Mistral-FlatJudge"]["weight"], 1.0)

    def test_reward_verification_reports_strict_stage_order(self) -> None:
        env = MultiAgentEnvironment(agents=build_heuristic_agents(), alpha=0.8, task_type="exact", seed=7)
        result = env.step(question="What is 2 + 2?", ground_truth="4", apply_updates=False)
        verification = result["metadata"]["reward_verification"]

        self.assertEqual(
            verification["stage_order"],
            [
                "ground_truth_reward",
                "raw_peer_scores",
                "score_normalization",
                "trust_weighting",
                "attention_weighting",
                "combined_peer_reward",
                "final_reward",
                "sanity_checks",
            ],
        )
        gt_map = {
            item["agent"]: item["reward"]
            for item in verification["ground_truth_reward"]
        }
        self.assertEqual(gt_map["Qwen-Small"], 1.0)
        self.assertEqual(gt_map["Llama-Tiny"], 1.0)
        self.assertEqual(gt_map["Kimi-Mini"], 0.0)
        self.assertEqual(gt_map["Mistral-FlatJudge"], 0.0)

    def test_peer_verification_blocks_self_scoring_and_preserves_rankings(self) -> None:
        env = MultiAgentEnvironment(agents=build_heuristic_agents(), alpha=0.8, task_type="exact", seed=7)
        result = env.step(question="What is 2 + 2?", ground_truth="4", apply_updates=False)

        for item in result["metadata"]["peer_evaluations"]:
            agent_index = next(index for index, agent in enumerate(env.agents) if agent.name == item["evaluator"])
            self.assertIsNone(item["raw_scores"][agent_index])
            self.assertTrue(item["self_evaluation_blocked"])
            self.assertTrue(item["normalization_preserved_ranking"])

    def test_run_batch_tracks_agent_stats(self) -> None:
        env = MultiAgentEnvironment(agents=build_heuristic_agents(), alpha=0.8, task_type="exact", seed=7)
        batch = env.run_batch(
            tasks=[
                {"question": "What is 2 + 2?", "ground_truth": "4"},
                {"question": "What is 2 + 2?", "ground_truth": "4"},
            ]
        )

        self.assertEqual(len(batch["results"]), 2)
        self.assertEqual(batch["agent_stats"]["Qwen-Small"]["rounds"], 2)

    def test_gsm8k_answer_extraction(self) -> None:
        answer = "Janet makes 3 dollars per hour.\nShe works 6 hours.\n#### 18"
        self.assertEqual(extract_gsm8k_final_answer(answer), "18")

    def test_gsm8k_row_conversion(self) -> None:
        tasks = build_gsm8k_tasks_from_rows(
            [
                {
                    "question": "If Sam has 2 apples and buys 3 more, how many apples does he have?",
                    "answer": "He has 5 apples.\n#### 5",
                }
            ]
        )
        self.assertEqual(tasks[0]["question"], "If Sam has 2 apples and buys 3 more, how many apples does he have?")
        self.assertEqual(tasks[0]["ground_truth"], "5")
        self.assertEqual(tasks[0]["source"], "openai/gsm8k")

    def test_extract_json_value_handles_wrapped_json(self) -> None:
        payload = extract_json_value('result:\n{"answer":"4","reasoning":"short"}\n')
        self.assertEqual(payload["answer"], "4")

    def test_build_ollama_agents_uses_requested_models(self) -> None:
        agents = build_ollama_agents(models=DEFAULT_OLLAMA_MODELS, host="http://127.0.0.1:11434", seed=7)
        self.assertEqual([agent.name for agent in agents], DEFAULT_OLLAMA_MODELS)
        self.assertTrue(all(isinstance(agent, OllamaAgent) for agent in agents))
        self.assertFalse(any(isinstance(agent, SelfRefineOllamaAgent) for agent in agents))

    def test_ollama_agent_extracts_chat_content(self) -> None:
        raw = {"message": {"content": '{"answer":"4","reasoning":"short"}'}}
        self.assertEqual(OllamaAgent._extract_model_output(raw), '{"answer":"4","reasoning":"short"}')

    def test_chunk_tasks_splits_batches(self) -> None:
        tasks = [{"task_id": str(index)} for index in range(5)]
        batches = list(chunk_tasks(tasks, batch_size=2))
        self.assertEqual([len(batch) for batch in batches], [2, 2, 1])

    def test_chunk_tasks_splits_iterators(self) -> None:
        task_iter = ({"task_id": str(index)} for index in range(5))
        batches = list(chunk_tasks(task_iter, batch_size=2))
        self.assertEqual([len(batch) for batch in batches], [2, 2, 1])

    def test_compute_summary_metrics(self) -> None:
        summary = compute_summary_metrics(
            [
                {
                    "metadata": {
                        "peer_evaluations": [],
                        "agent_failures": [],
                        "initial_round": {"rewards": []},
                        "reward_verification": {
                            "stage_order": EXPECTED_STAGE_ORDER,
                            "sanity_checks": {
                                "self_evaluation_blocked": True,
                                "normalization_preserves_ranking": True,
                                "attention_probabilities_valid": True,
                                "peer_scores_bounded": True,
                                "final_rewards_bounded": True,
                            },
                        },
                    },
                    "rewards": [
                        {"agent": "a", "gt_reward": 1.0, "peer_score": 0.8, "final_reward": 0.96},
                        {"agent": "b", "gt_reward": 0.0, "peer_score": 0.2, "final_reward": 0.04},
                    ],
                    "ranking": ["a", "b"],
                },
                {
                    "metadata": {
                        "peer_evaluations": [],
                        "agent_failures": [],
                        "initial_round": {"rewards": []},
                        "reward_verification": {
                            "stage_order": EXPECTED_STAGE_ORDER,
                            "sanity_checks": {
                                "self_evaluation_blocked": True,
                                "normalization_preserves_ranking": True,
                                "attention_probabilities_valid": True,
                                "peer_scores_bounded": True,
                                "final_rewards_bounded": True,
                            },
                        },
                    },
                    "rewards": [
                        {"agent": "a", "gt_reward": 0.0, "peer_score": 0.4, "final_reward": 0.08},
                        {"agent": "b", "gt_reward": 1.0, "peer_score": 0.7, "final_reward": 0.94},
                    ],
                    "ranking": ["b", "a"],
                },
            ]
        )
        self.assertEqual(summary["num_examples"], 2)
        self.assertEqual(summary["agent_metrics"]["a"]["win_rate"], 0.5)
        self.assertIn("pairwise_win_matrix", summary)
        self.assertIn("mean_trust_weight", summary["leaderboard"][0])
        self.assertEqual(summary["verification_metrics"]["stage_order_matches_expected"], 1.0)
        self.assertEqual(summary["verification_metrics"]["self_evaluation_blocked"], 1.0)

    def test_step_passes_evaluator_alignment_feedback(self) -> None:
        env = MultiAgentEnvironment(agents=build_heuristic_agents(), alpha=0.8, task_type="exact", seed=7)
        env.step(question="What is 3 + 5?", ground_truth="8")
        for agent in env.agents:
            self.assertEqual(len(agent.reward_history), 1)
            # GT spread across agents makes oracle alignment defined for evaluators
            self.assertEqual(len(agent.evaluator_alignment_history), 1)

    def test_icl_generation_memory_selection(self) -> None:
        agent = ICLOllamaAgent(
            name="m",
            model="m",
            host="http://127.0.0.1:11434",
            seed=0,
            memory_strategy="reward_weighted",
            prompt_memory_size=2,
            memory_buffer_size=10,
        )
        agent.generation_memory = [
            {"question": "q1", "answer": "1", "reasoning": "r", "final_reward": 0.2, "gt_reward": 0.0},
            {"question": "q2", "answer": "2", "reasoning": "r", "final_reward": 0.9, "gt_reward": 1.0},
            {"question": "q3", "answer": "3", "reasoning": "r", "final_reward": 0.5, "gt_reward": 0.5},
        ]
        picked = agent._select_generation_examples()
        self.assertEqual(len(picked), 2)
        self.assertEqual(float(picked[0]["final_reward"]), 0.9)

    def test_experiment_runner_persists_and_resumes(self) -> None:
        env = MultiAgentEnvironment(agents=build_heuristic_agents(), alpha=0.8, task_type="exact", seed=7)
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = ExperimentRunner(env=env, output_dir=Path(tmpdir), run_manifest={"backend": "heuristic"})
            tasks = [
                {"task_id": "sample:0", "question": "What is 2 + 2?", "ground_truth": "4", "source": "sample"},
                {"task_id": "sample:1", "question": "What is 2 + 2?", "ground_truth": "4", "source": "sample"},
            ]
            result = runner.run(tasks, resume=False, apply_updates=False)
            self.assertEqual(result["new_examples_processed"], 2)
            self.assertEqual(load_completed_task_ids(Path(tmpdir) / "results.jsonl"), {"sample:0", "sample:1"})

            resumed = runner.run(tasks, resume=True, apply_updates=False)
            self.assertEqual(resumed["new_examples_processed"], 0)

            summary = json.loads((Path(tmpdir) / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["num_examples"], 2)
            manifest = json.loads((Path(tmpdir) / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertIn("attention_temperature", manifest["environment"])

            lc_path = Path(tmpdir) / "learning_curve.jsonl"
            self.assertTrue(lc_path.exists())
            lc_lines = [line for line in lc_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(lc_lines), 2)
            self.assertTrue((Path(tmpdir) / "run_manifest.json").exists())

    def test_experiment_runner_rejects_manifest_mismatch_on_resume(self) -> None:
        env = MultiAgentEnvironment(agents=build_heuristic_agents(), alpha=0.8, task_type="exact", seed=7)
        with tempfile.TemporaryDirectory() as tmpdir:
            tasks = [{"task_id": "sample:0", "question": "What is 2 + 2?", "ground_truth": "4", "source": "sample"}]
            runner = ExperimentRunner(env=env, output_dir=Path(tmpdir), run_manifest={"backend": "heuristic"})
            runner.run(tasks, resume=False, apply_updates=False)

            mismatched = ExperimentRunner(env=env, output_dir=Path(tmpdir), run_manifest={"backend": "icl"})
            with self.assertRaises(ValueError):
                mismatched.run(tasks, resume=True, apply_updates=False)

    def test_concurrent_execution_preserves_agent_order(self) -> None:
        env = MultiAgentEnvironment(
            agents=[SlowAgent("a", 0.01), SlowAgent("b", 0.01), SlowAgent("c", 0.01)],
            alpha=0.8,
            task_type="exact",
            seed=7,
            max_concurrency=3,
        )
        result = env.step(question="q", ground_truth="q", apply_updates=False)
        self.assertEqual([response["agent"] for response in result["responses"]], ["a", "b", "c"])

    def test_revision_round_updates_final_response(self) -> None:
        env = MultiAgentEnvironment(
            agents=[RevisingAgent("a", "1", "4"), RevisingAgent("b", "2", "4")],
            alpha=0.8,
            task_type="exact",
            seed=7,
            revision_rounds=1,
        )
        result = env.step(question="What is 2 + 2?", ground_truth="4", apply_updates=False)
        self.assertEqual(result["responses"][0]["answer"], "4")
        self.assertEqual(result["metadata"]["initial_round"]["responses"][0]["answer"], "1")
        self.assertEqual(len(result["metadata"]["revision_history"]), 1)

    def test_continue_on_agent_error_records_failures(self) -> None:
        env = MultiAgentEnvironment(
            agents=[build_heuristic_agents()[0], FailingAgent("broken")],
            alpha=0.8,
            task_type="exact",
            seed=7,
            continue_on_agent_error=True,
        )
        result = env.step(question="What is 2 + 2?", ground_truth="4", apply_updates=False)
        failures = result["metadata"]["agent_failures"]
        self.assertTrue(any(item["agent"] == "broken" for item in failures))
        self.assertEqual(len(result["responses"]), 2)

    def test_trust_weighting_uses_alignment_history(self) -> None:
        agents = build_heuristic_agents()
        agents[0].evaluator_alignment_history = [1.0, 1.0]
        agents[1].evaluator_alignment_history = [0.2, 0.2]
        env = MultiAgentEnvironment(
            agents=agents,
            alpha=0.8,
            task_type="exact",
            seed=7,
            use_trust_weighting=True,
            historical_trust_blend=1.0,
        )
        result = env.step(question="What is 2 + 2?", ground_truth="4", apply_updates=False)
        peer_meta = {item["evaluator"]: item for item in result["metadata"]["peer_evaluations"]}
        self.assertGreater(peer_meta["Qwen-Small"]["trust_weight"], peer_meta["Kimi-Mini"]["trust_weight"])

    def test_load_tasks_supports_start_index_slicing(self) -> None:
        fake_rows = [
            {"question": "q1", "ground_truth": "1", "source": "openai/gsm8k"},
            {"question": "q2", "ground_truth": "2", "source": "openai/gsm8k"},
        ]
        with mock.patch("data.iter_gsm8k_tasks", return_value=iter(fake_rows)):
            tasks = load_tasks("gsm8k", split="test", limit=2, seed=7, start_index=1, streaming=False)
        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[0]["task_id"], "gsm8k:test:1")

    def test_attention_weights_sum_to_one(self) -> None:
        env = MultiAgentEnvironment(
            agents=build_heuristic_agents(use_rl=True),
            alpha=0.8,
            task_type="exact",
            seed=7,
            use_attention=True,
        )
        result = env.step(question="What is 2 + 2?", ground_truth="4", apply_updates=False)
        for item in result["metadata"]["attention_weights"]:
            weights = item["weights"]
            if weights:
                self.assertAlmostEqual(sum(weights), 1.0, places=5)

    def test_hand_computed_peer_and_final_rewards_match_formula(self) -> None:
        agents = [
            DeterministicAgent(name="a", answer="1", scores=[0.0, 0.2, 0.8]),
            DeterministicAgent(name="b", answer="0", scores=[0.9, 0.0, 0.3]),
            DeterministicAgent(name="c", answer="2", scores=[0.6, 0.4, 0.0]),
        ]
        env = MultiAgentEnvironment(
            agents=agents,
            alpha=0.8,
            task_type="exact",
            seed=7,
            use_attention=False,
            use_trust_weighting=False,
            disagreement_penalty_scale=0.0,
            flat_score_weight=1.0,
        )
        result = env.step(question="q", ground_truth="1", apply_updates=False)
        rewards = {item["agent"]: item for item in result["rewards"]}

        self.assertAlmostEqual(rewards["a"]["peer_score"], 1.0, places=4)
        self.assertAlmostEqual(rewards["b"]["peer_score"], 0.0, places=4)
        self.assertAlmostEqual(rewards["c"]["peer_score"], 0.5, places=4)
        self.assertAlmostEqual(rewards["a"]["final_reward"], 1.0, places=4)
        self.assertAlmostEqual(rewards["b"]["final_reward"], 0.0, places=4)
        self.assertAlmostEqual(rewards["c"]["final_reward"], 0.1, places=4)

    def test_sanity_checks_reduce_uniform_case_to_simple_average(self) -> None:
        agents = [
            DeterministicAgent(name="a", answer="1", scores=[0.0, 0.2, 0.8]),
            DeterministicAgent(name="b", answer="0", scores=[0.9, 0.0, 0.3]),
            DeterministicAgent(name="c", answer="2", scores=[0.6, 0.4, 0.0]),
        ]
        env = MultiAgentEnvironment(
            agents=agents,
            alpha=0.8,
            task_type="exact",
            seed=7,
            use_attention=False,
            use_trust_weighting=False,
            disagreement_penalty_scale=0.0,
            flat_score_weight=1.0,
        )
        result = env.step(question="q", ground_truth="1", apply_updates=False)
        sanity_checks = result["metadata"]["reward_verification"]["sanity_checks"]

        self.assertTrue(sanity_checks["self_evaluation_blocked"])
        self.assertTrue(sanity_checks["attention_probabilities_valid"])
        self.assertTrue(sanity_checks["uniform_trust_attention_reduce_to_mean"])

    def test_rl_update_is_called_and_policy_improves(self) -> None:
        agent = PolicyLearningAgent("learner")
        env = MultiAgentEnvironment(
            agents=[agent],
            alpha=0.8,
            task_type="exact",
            seed=7,
            use_rl=True,
            use_attention=True,
        )
        state = embed_text("What is 2 + 2?", dim=agent.feature_dim)
        skeptical_index = GENERATION_ACTIONS.index("skeptical")
        before = float(agent.generation_policy.probabilities(state)[skeptical_index])
        for _ in range(40):
            env.step(question="What is 2 + 2?", ground_truth="4", apply_updates=True)
        after = float(agent.generation_policy.probabilities(state)[skeptical_index])
        self.assertGreater(agent.update_calls, 0)
        self.assertGreater(after, before)


if __name__ == "__main__":
    unittest.main()
