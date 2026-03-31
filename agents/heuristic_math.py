from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import Agent
from utils.math_solver import extract_numeric_token, format_number, solve_math_question


class HeuristicMathAgent(Agent):
    """Question-aware stand-in for small LLM agents on arithmetic-style tasks."""

    def __init__(
        self,
        name: str,
        *,
        response_style: str,
        judge_style: str,
        seed: Optional[int] = None,
        use_rl: bool = False,
        rl_learning_rate: float = 0.05,
    ) -> None:
        super().__init__(
            name=name,
            seed=seed,
            use_rl=use_rl,
            rl_learning_rate=rl_learning_rate,
        )
        self.response_style = response_style
        self.judge_style = judge_style

    def generate(self, question: str) -> Dict[str, Any]:
        policy_trace = self._sample_generation_policy(question)
        solved_value, solver_reasoning = solve_math_question(question)
        if solved_value is None:
            return self._build_response_payload(
                "unknown",
                "I could not infer a stable arithmetic program from the question.",
                policy_trace=policy_trace,
            )

        response_style = self.response_style
        if self.use_rl:
            action = policy_trace["action"]
            response_style = {
                "direct": "accurate",
                "deliberate": "verbal",
                "skeptical": "accurate",
                "concise": "compact",
            }.get(action, self.response_style)

        if response_style == "accurate":
            answer = format_number(solved_value)
            reasoning = solver_reasoning
        elif response_style == "off_by_one":
            answer = format_number(solved_value + 1)
            reasoning = f"{solver_reasoning} I may have overcounted by one."
        elif response_style == "verbal":
            answer = f"The answer is {format_number(solved_value)}."
            reasoning = f"{solver_reasoning} This is the final result."
        elif response_style == "compact":
            answer = format_number(solved_value)
            reasoning = "Computed from the quantities in the prompt."
        else:
            raise ValueError(f"Unsupported response_style: {response_style}")

        return self._build_response_payload(answer, reasoning, policy_trace=policy_trace)

    def evaluate(self, question: str, responses: List[Dict[str, str]]) -> List[float]:
        inferred_value, _ = solve_math_question(question)
        inferred_answer = format_number(inferred_value) if inferred_value is not None else None
        scores: List[float] = []

        for response in responses:
            if response["agent"] == self.name:
                scores.append(0.0)
                continue

            response_numeric = extract_numeric_token(response["answer"])
            correctness = 0.0
            if inferred_answer is not None and response_numeric is not None:
                correctness = 1.0 if response_numeric == inferred_answer else 0.1
            elif response_numeric is not None:
                correctness = 0.35

            clarity = 1.0 if len(response["answer"].split()) <= 5 else 0.75
            reasoning_quality = 1.0 if response["reasoning"] else 0.4

            if self.judge_style == "strict":
                score = 0.75 * correctness + 0.15 * reasoning_quality + 0.10 * clarity
            elif self.judge_style == "clarity":
                score = 0.55 * correctness + 0.25 * clarity + 0.20 * reasoning_quality
            elif self.judge_style == "semantic":
                score = 0.65 * correctness + 0.10 * clarity + 0.25 * reasoning_quality
            elif self.judge_style == "flat":
                score = 0.8
            else:
                raise ValueError(f"Unsupported judge_style: {self.judge_style}")

            scores.append(min(1.0, max(0.0, score)))

        return scores

    def revise(
        self,
        question: str,
        own_response: Dict[str, Any],
        responses: List[Dict[str, Any]],
        peer_scores: Optional[List[float]] = None,
        attention_context: Optional[List[Dict[str, Any]]] = None,
        attention_weights: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        policy_trace = self._sample_revision_policy(question, own_response, attention_context)
        peer_scores = peer_scores or [0.0 for _ in responses]
        responses = attention_context or responses
        candidate = own_response
        candidate_score = -1.0

        for response, score in zip(responses, peer_scores):
            if response["agent"] == self.name:
                continue
            numeric = extract_numeric_token(response["answer"])
            if numeric is None:
                continue
            if score > candidate_score:
                candidate = response
                candidate_score = score

        own_numeric = extract_numeric_token(own_response.get("answer", ""))
        candidate_numeric = extract_numeric_token(candidate.get("answer", ""))
        if candidate_numeric and candidate_numeric != own_numeric and candidate_score > 0.5:
            return self._build_response_payload(
                candidate["answer"],
                f"Revised after peer comparison. Adopted peer-supported answer: {candidate['answer']}",
                policy_trace=policy_trace,
            )

        return super().revise(
            question,
            own_response,
            responses,
            peer_scores,
            attention_context=attention_context,
            attention_weights=attention_weights,
        )
