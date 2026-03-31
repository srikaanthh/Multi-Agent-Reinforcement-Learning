from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List

from environment import MultiAgentEnvironment
from experiment import ExperimentRunner
from run_demo import build_heuristic_agents, build_icl_ollama_agents, build_ollama_agents
from agents import build_self_refine_ollama_agents


def iter_tasks(
    dataset_name: str,
    split: str,
    limit: int | None,
    seed: int,
    start_index: int,
    *,
    streaming: bool = False,
) -> Iterator[Dict[str, str]]:
    if dataset_name == "gsm8k":
        from data import iter_gsm8k_tasks

        task_stream = iter_gsm8k_tasks(
            split=split,
            limit=limit,
            start_index=start_index,
            shuffle=False,
            seed=seed,
            streaming=streaming,
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    for offset, task in enumerate(task_stream, start=start_index):
        yield {
            "task_id": f"{dataset_name}:{split}:{offset}",
            "question": task["question"],
            "ground_truth": task["ground_truth"],
            "source": task.get("source", dataset_name),
        }


def load_tasks(
    dataset_name: str,
    split: str,
    limit: int | None,
    seed: int,
    start_index: int,
    *,
    streaming: bool = False,
) -> List[Dict[str, str]]:
    return list(
        iter_tasks(
            dataset_name=dataset_name,
            split=split,
            limit=limit,
            seed=seed,
            start_index=start_index,
            streaming=streaming,
        )
    )


def chunk_tasks(tasks: Iterable[Dict[str, str]], batch_size: int) -> Iterable[List[Dict[str, str]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    batch: List[Dict[str, str]] = []
    for task in tasks:
        batch.append(task)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def build_run_manifest(args: argparse.Namespace, *, task_type: str) -> Dict[str, object]:
    return {
        "dataset": args.dataset,
        "split": args.split,
        "backend": args.backend,
        "models": list(args.models),
        "seed": args.seed,
        "alpha": args.alpha,
        "task_type": task_type,
        "batch_size": args.batch_size,
        "limit": args.limit,
        "start_index": args.start_index,
        "streaming": args.streaming,
        "apply_updates": bool(args.apply_updates),
        "ollama_host": args.ollama_host,
        "icl_strategy": getattr(args, "icl_strategy", None),
        "prompt_memory_size": getattr(args, "prompt_memory_size", None),
        "memory_buffer_size": getattr(args, "memory_buffer_size", None),
        "revision_rounds": args.revision_rounds,
        "continue_on_agent_error": bool(args.continue_on_agent_error),
        "max_concurrency": args.max_concurrency,
        "use_trust_weighting": bool(args.use_trust_weighting),
        "historical_trust_blend": args.historical_trust_blend,
        "trust_floor": args.trust_floor,
        "use_rl": bool(args.use_rl),
        "use_attention": bool(args.use_attention),
        "attention_top_k": args.attention_top_k,
        "attention_temperature": args.attention_temperature,
        "attention_entropy_coef": args.attention_entropy_coef,
        "rl_learning_rate": args.rl_learning_rate,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run large-batch MARL experiments.")
    parser.add_argument("--dataset", choices=["gsm8k"], default="gsm8k")
    parser.add_argument("--split", default="train")
    parser.add_argument("--backend", choices=["ollama", "heuristic", "icl", "self_refine"], default="ollama")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--apply-updates", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--streaming", action="store_true", help="Stream dataset examples instead of materializing the split.")
    parser.add_argument("--output-dir", default="outputs/experiment")
    parser.add_argument("--ollama-host", default=None)
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "qwen3.5:latest",
            "kimi-k2.5:cloud",
            "qwen2.5-coder:1.5b-base",
            "qwen2.5:3b-instruct",
        ],
    )
    parser.add_argument(
        "--icl-strategy",
        choices=["none", "reward_weighted", "random", "oracle", "recency"],
        default="reward_weighted",
        help="ICL memory strategy for --backend icl.",
    )
    parser.add_argument("--prompt-memory-size", type=int, default=5, help="Examples in ICL prompts.")
    parser.add_argument("--memory-buffer-size", type=int, default=128, help="Max stored ICL experiences per agent.")
    parser.add_argument("--max-concurrency", type=int, default=None, help="Max concurrent agent calls inside one MARL step.")
    parser.add_argument("--revision-rounds", type=int, default=0, help="Number of peer-observation revision rounds before final scoring.")
    parser.add_argument("--historical-trust-blend", type=float, default=0.5, help="Blend factor for historical evaluator trust in peer aggregation.")
    parser.add_argument("--trust-floor", type=float, default=0.1, help="Lower bound for trust-weighted evaluator weights.")
    parser.add_argument("--attention-top-k", type=int, default=2, help="Number of attention-selected peers exposed during revision.")
    parser.add_argument("--attention-temperature", type=float, default=1.0, help="Softmax temperature for attention over peer agents.")
    parser.add_argument("--attention-entropy-coef", type=float, default=0.05, help="Entropy bonus coefficient for attention-weighted peer rewards.")
    parser.add_argument("--rl-learning-rate", type=float, default=0.05, help="Learning rate for contextual-bandit REINFORCE updates.")
    parser.set_defaults(use_rl=True)
    parser.add_argument("--use-rl", action="store_true", dest="use_rl", help="Enable contextual-bandit RL updates inside agents.")
    parser.add_argument("--disable-rl", action="store_false", dest="use_rl", help="Disable RL updates and use static prompting.")
    parser.set_defaults(use_attention=True)
    parser.add_argument("--use-attention", action="store_true", dest="use_attention", help="Use trust-weighted attention over peer agents.")
    parser.add_argument("--disable-attention", action="store_false", dest="use_attention", help="Disable attention and use scalar peer aggregation.")
    parser.set_defaults(use_trust_weighting=True)
    parser.add_argument("--use-trust-weighting", action="store_true", dest="use_trust_weighting", help="Use trust-weighted peer aggregation.")
    parser.add_argument("--disable-trust-weighting", action="store_false", dest="use_trust_weighting", help="Disable historical trust weighting and use anti-collusion weights only.")
    parser.set_defaults(continue_on_agent_error=True)
    parser.add_argument("--continue-on-agent-error", action="store_true", dest="continue_on_agent_error", help="Continue the run with fallbacks if an agent call fails.")
    parser.add_argument("--stop-on-agent-error", action="store_false", dest="continue_on_agent_error", help="Abort immediately if an agent call fails.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_iter = iter_tasks(
        dataset_name=args.dataset,
        split=args.split,
        limit=args.limit,
        seed=args.seed,
        start_index=args.start_index,
        streaming=args.streaming,
    )
    if args.backend == "heuristic":
        agents = build_heuristic_agents(use_rl=args.use_rl, rl_learning_rate=args.rl_learning_rate)
    elif args.backend == "icl":
        agents = build_icl_ollama_agents(
            models=args.models,
            host=args.ollama_host,
            seed=args.seed,
            memory_strategy=args.icl_strategy,
            prompt_memory_size=args.prompt_memory_size,
            memory_buffer_size=args.memory_buffer_size,
            use_rl=args.use_rl,
            rl_learning_rate=args.rl_learning_rate,
        )
    elif args.backend == "self_refine":
        agents = build_self_refine_ollama_agents(
            models=args.models,
            host=args.ollama_host,
            seed=args.seed,
            use_rl=args.use_rl,
            rl_learning_rate=args.rl_learning_rate,
        )
    else:
        agents = build_ollama_agents(
            models=args.models,
            host=args.ollama_host,
            seed=args.seed,
            use_rl=args.use_rl,
            rl_learning_rate=args.rl_learning_rate,
        )
    env = MultiAgentEnvironment(
        agents=agents,
        alpha=args.alpha,
        task_type="flexible",
        seed=args.seed,
        max_concurrency=args.max_concurrency,
        revision_rounds=args.revision_rounds,
        continue_on_agent_error=args.continue_on_agent_error,
        use_trust_weighting=args.use_trust_weighting,
        historical_trust_blend=args.historical_trust_blend,
        trust_floor=args.trust_floor,
        use_rl=args.use_rl,
        use_attention=args.use_attention,
        attention_top_k=args.attention_top_k,
        attention_temperature=args.attention_temperature,
        attention_entropy_coef=args.attention_entropy_coef,
    )
    runner = ExperimentRunner(
        env=env,
        output_dir=Path(args.output_dir),
        run_manifest=build_run_manifest(args, task_type="flexible"),
    )

    aggregate_result = None
    for batch in chunk_tasks(task_iter, args.batch_size):
        aggregate_result = runner.run(
            batch,
            resume=args.resume,
            apply_updates=args.apply_updates,
            flush_every=1,
            track_learning_curve=True,
        )

    if aggregate_result is None:
        aggregate_result = runner.run(
            [],
            resume=args.resume,
            apply_updates=args.apply_updates,
            track_learning_curve=True,
        )

    print(json.dumps(aggregate_result, indent=2))


if __name__ == "__main__":
    main()
