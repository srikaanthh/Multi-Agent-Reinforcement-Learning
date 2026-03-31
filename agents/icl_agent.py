from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Sequence

from .ollama_agent import OllamaAgent

MemoryStrategy = Literal["none", "reward_weighted", "random", "oracle", "recency"]


def _truncate(text: str, max_len: int = 400) -> str:
    text = text.strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


class ICLOllamaAgent(OllamaAgent):
    """Ollama agent with in-context examples for generation and evaluation (paper experiments)."""

    def __init__(
        self,
        name: str,
        model: str,
        *,
        host: Optional[str] = None,
        temperature: float = 0.2,
        num_predict: int = 256,
        timeout_seconds: int = 180,
        seed: Optional[int] = None,
        memory_strategy: MemoryStrategy = "reward_weighted",
        prompt_memory_size: int = 5,
        memory_buffer_size: int = 128,
        oracle_gt_threshold: float = 0.99,
        include_eval_memory_in_prompt: bool = True,
        use_rl: bool = False,
        rl_learning_rate: float = 0.05,
    ) -> None:
        super().__init__(
            name=name,
            model=model,
            host=host,
            temperature=temperature,
            num_predict=num_predict,
            timeout_seconds=timeout_seconds,
            seed=seed,
            use_rl=use_rl,
            rl_learning_rate=rl_learning_rate,
        )
        self.memory_strategy = memory_strategy
        self.prompt_memory_size = max(0, prompt_memory_size)
        self.memory_buffer_size = max(1, memory_buffer_size)
        self.oracle_gt_threshold = oracle_gt_threshold
        self.include_eval_memory_in_prompt = include_eval_memory_in_prompt

    def update(self, trajectory: Dict[str, Any]) -> None:
        super().update(trajectory)
        feedback = trajectory.get("feedback")
        if not isinstance(feedback, dict) or self.memory_strategy == "none":
            return

        own = feedback.get("own_response") or {}
        gen_entry: Dict[str, Any] = {
            "question": str(feedback.get("question", "")),
            "answer": str(own.get("answer", "")),
            "reasoning": str(own.get("reasoning", "")),
            "final_reward": float(feedback.get("final_reward", 0.0)),
            "gt_reward": float(feedback.get("gt_reward", 0.0)),
        }
        self.generation_memory.append(gen_entry)
        if len(self.generation_memory) > self.memory_buffer_size:
            self.generation_memory = self.generation_memory[-self.memory_buffer_size :]

        eval_entry: Dict[str, Any] = {
            "question": str(feedback.get("question", "")),
            "ground_truth": str(feedback.get("ground_truth", "")),
            "evaluator_alignment": feedback.get("evaluator_alignment"),
            "peer_evaluations": feedback.get("peer_evaluations"),
        }
        self.eval_memory.append(eval_entry)
        if len(self.eval_memory) > self.memory_buffer_size:
            self.eval_memory = self.eval_memory[-self.memory_buffer_size :]

    def _select_generation_examples(self) -> List[Dict[str, Any]]:
        if self.prompt_memory_size == 0 or not self.generation_memory:
            return []

        pool: List[Dict[str, Any]] = list(self.generation_memory)
        if self.memory_strategy == "oracle":
            pool = [e for e in pool if float(e.get("gt_reward", 0.0)) >= self.oracle_gt_threshold]
        if not pool:
            return []

        if self.memory_strategy == "random":
            k = min(self.prompt_memory_size, len(pool))
            return self.rng.sample(pool, k=k)

        if self.memory_strategy == "recency":
            return pool[-self.prompt_memory_size :]

        # reward_weighted and oracle (filtered): rank by final reward
        ranked = sorted(pool, key=lambda e: float(e.get("final_reward", 0.0)), reverse=True)
        return ranked[: self.prompt_memory_size]

    def _select_eval_examples(self) -> List[Dict[str, Any]]:
        if not self.include_eval_memory_in_prompt or self.prompt_memory_size == 0:
            return []
        # Prefer past steps where oracle alignment was observable and high
        scored = [e for e in self.eval_memory if isinstance(e.get("evaluator_alignment"), (int, float))]
        if not scored:
            return []
        ranked = sorted(
            scored,
            key=lambda e: float(e.get("evaluator_alignment") or 0.0),
            reverse=True,
        )
        return ranked[: self.prompt_memory_size]

    def _format_generation_memory_block(self) -> str:
        examples = self._select_generation_examples()
        if not examples:
            return ""

        lines = [
            "You are learning from past experience in this multi-agent environment.",
            "Below are prior questions you attempted, your answers, and the final reward you received (higher is better).",
            "Use these patterns to improve your reasoning and final numeric/text answers.",
        ]
        for index, ex in enumerate(examples, start=1):
            lines.append(
                f"\n--- Example {index} (final_reward={float(ex.get('final_reward', 0.0)):.3f}, "
                f"gt_reward={float(ex.get('gt_reward', 0.0)):.3f}) ---"
            )
            lines.append(f"Question: {_truncate(str(ex.get('question', '')), 600)}")
            lines.append(f"Your answer: {_truncate(str(ex.get('answer', '')), 200)}")
            lines.append(f"Your reasoning: {_truncate(str(ex.get('reasoning', '')), 400)}")
        lines.append("\nNow answer the NEW question below.")
        return "\n".join(lines)

    def _format_eval_memory_block(self) -> str:
        examples = self._select_eval_examples()
        if not examples:
            return ""

        lines = [
            "When scoring peers, you previously received an alignment score vs. the reference when available "
            "(1.0 = your scores matched the reference spread well).",
            "Below are past questions where your judging alignment was relatively high — keep similar calibration.",
        ]
        for index, ex in enumerate(examples, start=1):
            align = ex.get("evaluator_alignment")
            align_s = f"{float(align):.3f}" if isinstance(align, (int, float)) else "n/a"
            lines.append(f"\n--- Past question {index} (evaluator_alignment={align_s}) ---")
            lines.append(f"Question: {_truncate(str(ex.get('question', '')), 500)}")
            lines.append(f"Reference answer (for calibration): {_truncate(str(ex.get('ground_truth', '')), 120)}")
        lines.append("\nNow score the NEW responses below.")
        return "\n".join(lines)

    def _build_generation_prompt(self, question: str, policy_action: str) -> str:
        base = super()._build_generation_prompt(question, policy_action)
        if self.memory_strategy == "none" or self.prompt_memory_size == 0:
            return base
        block = self._format_generation_memory_block()
        if not block:
            return base
        return block + "\n\n" + base

    def _build_evaluation_prompt(self, question: str, responses: List[Dict[str, str]]) -> str:
        base = super()._build_evaluation_prompt(question, responses)
        if self.memory_strategy == "none" or not self.include_eval_memory_in_prompt:
            return base
        block = self._format_eval_memory_block()
        if not block:
            return base
        return block + "\n\n" + base


def build_icl_ollama_agents(
    models: Sequence[str],
    host: Optional[str],
    seed: int,
    *,
    memory_strategy: MemoryStrategy = "reward_weighted",
    prompt_memory_size: int = 5,
    memory_buffer_size: int = 128,
    oracle_gt_threshold: float = 0.99,
    include_eval_memory_in_prompt: bool = True,
    use_rl: bool = False,
    rl_learning_rate: float = 0.05,
) -> List[ICLOllamaAgent]:
    """Factory for ICL Ollama agents (one model name per agent)."""
    return [
        ICLOllamaAgent(
            name=model,
            model=model,
            host=host,
            seed=seed + index,
            memory_strategy=memory_strategy,
            prompt_memory_size=prompt_memory_size,
            memory_buffer_size=memory_buffer_size,
            oracle_gt_threshold=oracle_gt_threshold,
            include_eval_memory_in_prompt=include_eval_memory_in_prompt,
            use_rl=use_rl,
            rl_learning_rate=rl_learning_rate,
        )
        for index, model in enumerate(models)
    ]
