from __future__ import annotations

from typing import Any, List, Optional, Sequence

from .ollama_agent import OllamaAgent


class SelfRefineOllamaAgent(OllamaAgent):
    """Baseline agent that revises only its own answer, without using peer responses."""

    def revise(
        self,
        question: str,
        own_response: dict[str, Any],
        responses: List[dict[str, Any]],
        peer_scores: Optional[List[float]] = None,
        attention_context: Optional[List[dict[str, Any]]] = None,
        attention_weights: Optional[List[float]] = None,
    ) -> dict[str, Any]:
        del responses, peer_scores, attention_context, attention_weights
        policy_trace = self._sample_revision_policy(question, own_response, None)
        prompt = (
            "You are revising your own answer without access to peer feedback.\n"
            f"Agent name: {self.name}\n"
            f"Question: {question}\n\n"
            f"Revision policy mode: {policy_trace['action']}\n"
            f"Your current answer: {own_response.get('answer', '')}\n"
            f"Your current reasoning: {own_response.get('reasoning', '')}\n\n"
            "Check your answer carefully and improve it if needed.\n"
            "Return only valid JSON with this exact schema:\n"
            '{"answer":"string","reasoning":"string"}\n\n'
            "Requirements:\n"
            "- Keep the final answer concise.\n"
            "- Use the reasoning field to briefly explain whether you changed the answer.\n"
            "- No markdown, no extra keys, no surrounding commentary."
        )
        payload = self._generate_json(prompt)
        answer = str(payload.get("answer", "")).strip()
        reasoning = str(payload.get("reasoning", "")).strip()
        if not answer:
            raise RuntimeError(f"Self-refine model {self.model} returned an empty revised answer.")
        return self._build_response_payload(answer, reasoning, policy_trace=policy_trace)


def build_self_refine_ollama_agents(
    models: Sequence[str],
    host: Optional[str],
    seed: int,
    *,
    use_rl: bool = False,
    rl_learning_rate: float = 0.05,
) -> list[SelfRefineOllamaAgent]:
    return [
        SelfRefineOllamaAgent(
            name=model,
            model=model,
            host=host,
            seed=seed + index,
            use_rl=use_rl,
            rl_learning_rate=rl_learning_rate,
        )
        for index, model in enumerate(models)
    ]
