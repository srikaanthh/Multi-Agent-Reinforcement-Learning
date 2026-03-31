from __future__ import annotations

import argparse
import json
from typing import Dict, List

from agents import HeuristicMathAgent, OllamaAgent, build_icl_ollama_agents, build_self_refine_ollama_agents
from data import load_gsm8k_tasks
from environment import MultiAgentEnvironment
 

DEFAULT_OLLAMA_MODELS = [
    "qwen3.5:latest",
    "kimi-k2.5:cloud",
    "qwen2.5-coder:1.5b-base",
    "llava:latest",
    "qwen2.5:3b-instruct",
]


def build_heuristic_agents(*, use_rl: bool = False, rl_learning_rate: float = 0.05) -> List[HeuristicMathAgent]:
    return [
        HeuristicMathAgent(
            name="Qwen-Small",
            response_style="accurate",
            judge_style="strict",
            seed=11,
            use_rl=use_rl,
            rl_learning_rate=rl_learning_rate,
        ),
        HeuristicMathAgent(
            name="Kimi-Mini",
            response_style="off_by_one",
            judge_style="clarity",
            seed=22,
            use_rl=use_rl,
            rl_learning_rate=rl_learning_rate,
        ),
        HeuristicMathAgent(
            name="Llama-Tiny",
            response_style="compact",
            judge_style="semantic",
            seed=33,
            use_rl=use_rl,
            rl_learning_rate=rl_learning_rate,
        ),
        HeuristicMathAgent(
            name="Mistral-FlatJudge",
            response_style="verbal",
            judge_style="flat",
            seed=44,
            use_rl=use_rl,
            rl_learning_rate=rl_learning_rate,
        ),
    ]


def build_ollama_agents(
    models: List[str],
    host: str | None,
    seed: int,
    *,
    use_rl: bool = False,
    rl_learning_rate: float = 0.05,
) -> List[OllamaAgent]:
    return [
        OllamaAgent(
            name=model,
            model=model,
            host=host,
            seed=seed + index,
            use_rl=use_rl,
            rl_learning_rate=rl_learning_rate,
        )
        for index, model in enumerate(models)
    ]


def load_tasks(dataset_name: str, split: str, limit: int, seed: int) -> List[Dict[str, str]]:
    if dataset_name == "sample":
        return [
            {
                "question": "What is 2 + 2?",
                "ground_truth": "4",
                "source": "sample",
            }
        ]
    if dataset_name == "gsm8k":
        return load_gsm8k_tasks(split=split, limit=limit, shuffle=False, seed=seed)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MARL environment demo.")
    parser.add_argument("--dataset", choices=["gsm8k", "sample"], default="gsm8k")
    parser.add_argument("--backend", choices=["ollama", "heuristic", "icl", "self_refine"], default="ollama")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ollama-host", default=None)
    parser.add_argument("--models", nargs="+", default=DEFAULT_OLLAMA_MODELS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = load_tasks(dataset_name=args.dataset, split=args.split, limit=args.limit, seed=args.seed)
    if args.backend == "heuristic":
        agents = build_heuristic_agents()
    elif args.backend == "icl":
        agents = build_icl_ollama_agents(
            models=args.models,
            host=args.ollama_host,
            seed=args.seed,
            memory_strategy="reward_weighted",
            prompt_memory_size=5,
        )
    elif args.backend == "self_refine":
        agents = build_self_refine_ollama_agents(
            models=args.models,
            host=args.ollama_host,
            seed=args.seed,
        )
    else:
        agents = build_ollama_agents(models=args.models, host=args.ollama_host, seed=args.seed)

    env = MultiAgentEnvironment(
        agents=agents,
        alpha=0.8,
        task_type="flexible" if args.dataset == "gsm8k" else "exact",
        seed=args.seed,
    )
    if len(tasks) == 1:
        result = env.step(question=tasks[0]["question"], ground_truth=tasks[0]["ground_truth"])
    else:
        result = env.run_batch(tasks=tasks)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
