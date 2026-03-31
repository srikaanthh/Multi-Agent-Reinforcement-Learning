from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional
from urllib import error, request

from .base import Agent
from utils.json_utils import extract_json_value


class OllamaAgent(Agent):
    """Agent implementation backed by a local Ollama model."""

    def __init__(
        self,
        name: str,
        model: str,
        *,
        host: Optional[str] = None,
        temperature: float = 0.2,
        num_predict: int = 256,
        timeout_seconds: int = 180,
        max_retries: int = 2,
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
        self.model = model
        self.host = (host or os.getenv("OLLAMA_HOST") or "http://127.0.0.1:11434").rstrip("/")
        self.temperature = temperature
        self.num_predict = num_predict
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(0, max_retries)

    def generate(self, question: str) -> Dict[str, Any]:
        policy_trace = self._sample_generation_policy(question)
        prompt = self._build_generation_prompt(question, policy_trace["action"])
        payload = self._generate_json(prompt)
        answer = str(payload.get("answer", "")).strip()
        reasoning = str(payload.get("reasoning", "")).strip()
        if not answer:
            raise RuntimeError(f"Ollama model {self.model} returned an empty answer.")
        return self._build_response_payload(answer, reasoning, policy_trace=policy_trace)

    def evaluate(self, question: str, responses: List[Dict[str, str]]) -> List[float]:
        prompt = self._build_evaluation_prompt(question, responses)
        payload = self._generate_json(prompt)
        raw_scores = payload.get("scores")
        if not isinstance(raw_scores, list):
            raise RuntimeError(f"Ollama model {self.model} returned invalid evaluation payload: {payload!r}")
        return self._normalize_score_list(raw_scores, responses)

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
        prompt = self._build_revision_prompt(
            question,
            own_response,
            attention_context or responses,
            peer_scores,
            policy_action=policy_trace["action"],
            attention_weights=attention_weights,
        )
        payload = self._generate_json(prompt)
        answer = str(payload.get("answer", "")).strip()
        reasoning = str(payload.get("reasoning", "")).strip()
        if not answer:
            raise RuntimeError(f"Ollama model {self.model} returned an empty revised answer.")
        return self._build_response_payload(
            answer,
            reasoning,
            policy_trace=policy_trace,
            extra={"revision_source": "attention_context"},
        )

    def _generate_json(self, prompt: str) -> Dict[str, object]:
        chat_raw = self._post_json(
            endpoint="/api/chat",
            body={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "format": "json",
                "think": False,
                "options": self._request_options(),
            },
        )
        model_output = self._extract_model_output(chat_raw)

        if not model_output:
            generate_raw = self._post_json(
                endpoint="/api/generate",
                body={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "think": False,
                    "options": self._request_options(),
                },
            )
            model_output = self._extract_model_output(generate_raw)
            raw_payload = generate_raw
        else:
            raw_payload = chat_raw

        if not model_output:
            payload_preview = json.dumps(raw_payload, ensure_ascii=True)[:500]
            raise RuntimeError(
                f"Ollama model {self.model} returned no usable content. Raw payload preview: {payload_preview}"
            )

        parsed = extract_json_value(model_output)
        if not isinstance(parsed, dict):
            raise RuntimeError(f"Ollama model {self.model} returned non-object JSON: {parsed!r}")
        return parsed

    def _post_json(self, endpoint: str, body: Dict[str, object]) -> Dict[str, object]:
        http_request = request.Request(
            f"{self.host}{endpoint}",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                with request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                    raw = json.loads(response.read().decode("utf-8"))
                if "error" in raw and raw["error"]:
                    raise RuntimeError(f"Ollama returned an error for model `{self.model}`: {raw['error']}")
                return raw
            except error.HTTPError as exc:
                response_text = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(
                    f"Ollama request to {endpoint} failed for model `{self.model}` with HTTP {exc.code}: {response_text}"
                )
            except error.URLError as exc:
                last_error = RuntimeError(
                    f"Failed to contact Ollama at {self.host}. Ensure `ollama serve` is running and model `{self.model}` is available."
                )
            except RuntimeError as exc:
                last_error = exc

            if attempt < self.max_retries:
                time.sleep(0.5 * (attempt + 1))

        assert last_error is not None
        raise last_error

    def _request_options(self) -> Dict[str, object]:
        return {
            "temperature": self.temperature,
            "seed": self.seed,
            "num_predict": self.num_predict,
        }

    @staticmethod
    def _extract_model_output(raw: Dict[str, object]) -> str:
        response_text = str(raw.get("response", "")).strip()
        if response_text:
            return response_text

        message = raw.get("message")
        if isinstance(message, dict):
            content = str(message.get("content", "")).strip()
            if content:
                return content

        return ""

    def _build_generation_prompt(self, question: str, policy_action: str) -> str:
        style_instruction = self._generation_style_instruction(policy_action)
        return (
            "You are one agent in a multi-agent evaluation environment.\n"
            f"Agent name: {self.name}\n"
            f"Question: {question}\n\n"
            f"Policy mode: {policy_action}\n"
            f"{style_instruction}\n\n"
            "Return only valid JSON with this exact schema:\n"
            '{"answer":"string","reasoning":"string"}\n\n'
            "Requirements:\n"
            "- Solve the question as well as you can.\n"
            "- Put the final short answer in `answer`.\n"
            "- Put a concise explanation in `reasoning`.\n"
            "- Do not include markdown, extra keys, or surrounding commentary."
        )

    def _build_evaluation_prompt(self, question: str, responses: List[Dict[str, str]]) -> str:
        lines = []
        for index, response in enumerate(responses):
            lines.append(
                f"{index}. agent={response['agent']}\n"
                f"answer={response['answer']}\n"
                f"reasoning={response['reasoning']}"
            )
        rendered_responses = "\n\n".join(lines)
        exact_schema = json.dumps({"scores": [0.0 for _ in responses]})
        return (
            "You are scoring peer answers in a multi-agent evaluation environment.\n"
            f"Your evaluator name: {self.name}\n"
            f"Question: {question}\n\n"
            "Score each response on correctness, reasoning quality, and clarity.\n"
            "Use scores in [0, 1]. Higher is better.\n"
            "Do not self-promote: if a response is from your own agent, assign 0.\n\n"
            f"Responses:\n{rendered_responses}\n\n"
            "Return only valid JSON with this exact schema:\n"
            f"{exact_schema}\n\n"
            "Requirements:\n"
            "- Return one score per response in the same order.\n"
            "- Use decimals between 0 and 1.\n"
            "- No explanation, no markdown, no extra keys."
        )

    def _build_revision_prompt(
        self,
        question: str,
        own_response: Dict[str, Any],
        responses: List[Dict[str, Any]],
        peer_scores: Optional[List[float]] = None,
        *,
        policy_action: str,
        attention_weights: Optional[List[float]] = None,
    ) -> str:
        rendered = []
        peer_scores = peer_scores or [0.0 for _ in responses]
        attention_weights = attention_weights or [0.0 for _ in responses]
        for index, (response, peer_score, attention_weight) in enumerate(zip(responses, peer_scores, attention_weights)):
            rendered.append(
                f"{index}. agent={response['agent']}\n"
                f"answer={response['answer']}\n"
                f"reasoning={response['reasoning']}\n"
                f"peer_score={peer_score:.4f}\n"
                f"attention_weight={attention_weight:.4f}"
            )
        return (
            "You are revising your answer after observing peer responses in a multi-agent evaluation environment.\n"
            f"Agent name: {self.name}\n"
            f"Question: {question}\n\n"
            f"Revision policy mode: {policy_action}\n"
            f"{self._revision_style_instruction(policy_action)}\n\n"
            f"Your current answer: {own_response.get('answer', '')}\n"
            f"Your current reasoning: {own_response.get('reasoning', '')}\n\n"
            f"Attention-filtered peer responses and peer support:\n{chr(10).join(rendered)}\n\n"
            "Revise only if the peer evidence improves correctness or clarity.\n"
            "Return only valid JSON with this exact schema:\n"
            '{"answer":"string","reasoning":"string"}\n\n'
            "Requirements:\n"
            "- Keep the final answer concise.\n"
            "- Use the reasoning field to briefly explain whether you changed your answer and why.\n"
            "- No markdown, no extra keys, no surrounding commentary."
        )

    @staticmethod
    def _coerce_score(value: object) -> float:
        score = float(value)
        if score < 0.0:
            return 0.0
        if score > 1.0:
            return 1.0
        return score

    def _normalize_score_list(self, raw_scores: List[object], responses: List[Dict[str, str]]) -> List[float]:
        target_length = len(responses)
        scores = [self._coerce_score(score) for score in raw_scores if self._is_number_like(score)]

        if len(scores) == target_length:
            return scores

        self_index = next(
            (index for index, response in enumerate(responses) if response["agent"] == self.name),
            None,
        )

        if self_index is not None and len(scores) == target_length - 1:
            scores.insert(self_index, 0.0)

        if len(scores) < target_length:
            scores.extend([0.0] * (target_length - len(scores)))
        elif len(scores) > target_length:
            scores = scores[:target_length]

        if self_index is not None and self_index < len(scores):
            scores[self_index] = 0.0

        return scores

    @staticmethod
    def _generation_style_instruction(policy_action: str) -> str:
        if policy_action == "deliberate":
            return "Think carefully, make the intermediate reasoning explicit, and avoid skipping arithmetic checks."
        if policy_action == "skeptical":
            return "Be skeptical of the first intuition, verify edge cases, and correct likely mistakes before answering."
        if policy_action == "concise":
            return "Optimize for a short, direct answer with only the minimum necessary reasoning."
        return "Answer directly and clearly."

    @staticmethod
    def _revision_style_instruction(policy_action: str) -> str:
        if policy_action == "peer_weighted":
            return "Prefer high-attention, high-trust peer evidence when it clearly improves correctness."
        if policy_action == "consistency_check":
            return "Use the peer context to cross-check arithmetic and internal consistency before changing your answer."
        return "Keep your answer unless the filtered peer evidence is clearly better."

    @staticmethod
    def _is_number_like(value: object) -> bool:
        try:
            float(value)
        except (TypeError, ValueError):
            return False
        return True
