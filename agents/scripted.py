from __future__ import annotations

from typing import Callable, Dict, List, Optional

from .base import Agent


ResponseGenerator = Callable[[str, str], Dict[str, str]]
EvaluationPolicy = Callable[[str, str, List[Dict[str, str]]], List[float]]


class ScriptedAgent(Agent):
    """Deterministic pluggable agent used for reproducible simulation."""

    def __init__(
        self,
        name: str,
        response_generator: ResponseGenerator,
        evaluation_policy: EvaluationPolicy,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(name=name, seed=seed)
        self._response_generator = response_generator
        self._evaluation_policy = evaluation_policy

    def generate(self, question: str) -> Dict[str, str]:
        payload = self._response_generator(self.name, question)
        return {
            "answer": str(payload.get("answer", "")).strip(),
            "reasoning": str(payload.get("reasoning", "")).strip(),
        }

    def evaluate(self, question: str, responses: List[Dict[str, str]]) -> List[float]:
        scores = self._evaluation_policy(self.name, question, responses)
        if len(scores) != len(responses):
            raise ValueError(
                f"Agent {self.name} returned {len(scores)} scores for {len(responses)} responses."
            )
        return [float(score) for score in scores]
