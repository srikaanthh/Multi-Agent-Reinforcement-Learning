from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from environment import MultiAgentEnvironment

from .runner import ExperimentRunner

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Install PyYAML to use ablation configs: pip install pyyaml") from exc


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping.")
    return data


def _build_agents(
    run: Dict[str, Any],
    seed: int,
    ollama_host: Optional[str],
    models: List[str],
):
    from agents import build_icl_ollama_agents, build_self_refine_ollama_agents
    from run_demo import build_heuristic_agents, build_ollama_agents

    backend = str(run.get("backend", "ollama"))
    if backend == "heuristic":
        return build_heuristic_agents(
            use_rl=bool(run.get("use_rl", False)),
            rl_learning_rate=float(run.get("rl_learning_rate", 0.05)),
        )
    if backend == "ollama":
        return build_ollama_agents(
            models=models,
            host=ollama_host,
            seed=seed,
            use_rl=bool(run.get("use_rl", False)),
            rl_learning_rate=float(run.get("rl_learning_rate", 0.05)),
        )
    if backend == "self_refine":
        return build_self_refine_ollama_agents(
            models=models,
            host=ollama_host,
            seed=seed,
            use_rl=bool(run.get("use_rl", False)),
            rl_learning_rate=float(run.get("rl_learning_rate", 0.05)),
        )
    if backend == "icl":
        return build_icl_ollama_agents(
            models=models,
            host=ollama_host,
            seed=seed,
            memory_strategy=run.get("memory_strategy", "reward_weighted"),
            prompt_memory_size=int(run.get("prompt_memory_size", 5)),
            memory_buffer_size=int(run.get("memory_buffer_size", 128)),
            oracle_gt_threshold=float(run.get("oracle_gt_threshold", 0.99)),
            include_eval_memory_in_prompt=bool(run.get("include_eval_memory_in_prompt", True)),
            use_rl=bool(run.get("use_rl", False)),
            rl_learning_rate=float(run.get("rl_learning_rate", 0.05)),
        )
    raise ValueError(f"Unknown backend: {backend}")


def run_ablation_from_config(
    config_path: Path,
    *,
    override_output_root: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Run all (seed x run) combinations from a YAML experiment file."""
    cfg = _load_yaml(config_path)
    from run_experiment import chunk_tasks, iter_tasks

    dataset = str(cfg.get("dataset", "gsm8k"))
    split = str(cfg.get("split", "train"))
    limit = cfg.get("limit")
    start_index = int(cfg.get("start_index", 0))
    batch_size = int(cfg.get("batch_size", 10))
    alpha = float(cfg.get("alpha", 0.8))
    apply_updates = bool(cfg.get("apply_updates", True))
    resume = bool(cfg.get("resume", False))
    streaming = bool(cfg.get("streaming", False))
    continue_on_agent_error = bool(cfg.get("continue_on_agent_error", True))
    use_trust_weighting = bool(cfg.get("use_trust_weighting", True))
    historical_trust_blend = float(cfg.get("historical_trust_blend", 0.5))
    trust_floor = float(cfg.get("trust_floor", 0.1))
    use_rl = bool(cfg.get("use_rl", False))
    use_attention = bool(cfg.get("use_attention", True))
    attention_top_k = int(cfg.get("attention_top_k", 2))
    attention_temperature = float(cfg.get("attention_temperature", 1.0))
    attention_entropy_coef = float(cfg.get("attention_entropy_coef", 0.05))
    ollama_host = cfg.get("ollama_host")
    models: List[str] = list(cfg.get("models") or [])
    seeds: List[int] = [int(s) for s in (cfg.get("seeds") or [7])]
    runs: List[Dict[str, Any]] = list(cfg.get("runs") or [])
    max_concurrency = cfg.get("max_concurrency")
    if not runs:
        raise ValueError("Config must define a non-empty `runs` list.")
    if not models:
        raise ValueError("Config must define `models`.")

    output_root = Path(cfg.get("output_root", "outputs/ablations"))
    if override_output_root is not None:
        output_root = override_output_root

    results_summary: List[Dict[str, Any]] = []

    for run in runs:
        run_name = str(run.get("name", "unnamed"))
        for seed in seeds:
            tasks = iter_tasks(
                dataset_name=dataset,
                split=split,
                limit=int(limit) if limit is not None else None,
                seed=seed,
                start_index=start_index,
                streaming=streaming,
            )
            agents = _build_agents(run, seed, ollama_host, models)
            env = MultiAgentEnvironment(
                agents=agents,
                alpha=float(run.get("alpha", alpha)),
                task_type=str(run.get("task_type", "flexible")),
                seed=seed,
                max_concurrency=int(max_concurrency) if max_concurrency is not None else None,
                revision_rounds=int(run.get("revision_rounds", cfg.get("revision_rounds", 0))),
                continue_on_agent_error=continue_on_agent_error,
                use_trust_weighting=bool(run.get("use_trust_weighting", use_trust_weighting)),
                historical_trust_blend=float(run.get("historical_trust_blend", historical_trust_blend)),
                trust_floor=float(run.get("trust_floor", trust_floor)),
                use_rl=bool(run.get("use_rl", use_rl)),
                use_attention=bool(run.get("use_attention", use_attention)),
                attention_top_k=int(run.get("attention_top_k", attention_top_k)),
                attention_temperature=float(run.get("attention_temperature", attention_temperature)),
                attention_entropy_coef=float(run.get("attention_entropy_coef", attention_entropy_coef)),
            )
            out_dir = output_root / f"{run_name}_seed{seed}"
            runner = ExperimentRunner(
                env=env,
                output_dir=out_dir,
                run_manifest={
                    "config_path": str(config_path),
                    "dataset": dataset,
                    "split": split,
                    "limit": int(limit) if limit is not None else None,
                    "start_index": start_index,
                    "batch_size": batch_size,
                    "seed": seed,
                    "resume": resume,
                    "streaming": streaming,
                    "alpha": float(run.get("alpha", alpha)),
                    "task_type": str(run.get("task_type", "flexible")),
                    "apply_updates": apply_updates,
                    "ollama_host": ollama_host,
                    "models": models,
                    "max_concurrency": int(max_concurrency) if max_concurrency is not None else None,
                    "continue_on_agent_error": continue_on_agent_error,
                    "revision_rounds": int(run.get("revision_rounds", cfg.get("revision_rounds", 0))),
                    "use_trust_weighting": bool(run.get("use_trust_weighting", use_trust_weighting)),
                    "historical_trust_blend": float(run.get("historical_trust_blend", historical_trust_blend)),
                    "trust_floor": float(run.get("trust_floor", trust_floor)),
                    "use_rl": bool(run.get("use_rl", use_rl)),
                    "use_attention": bool(run.get("use_attention", use_attention)),
                    "attention_top_k": int(run.get("attention_top_k", attention_top_k)),
                    "attention_temperature": float(run.get("attention_temperature", attention_temperature)),
                    "attention_entropy_coef": float(run.get("attention_entropy_coef", attention_entropy_coef)),
                    "run": run,
                },
            )
            last: Optional[Dict[str, object]] = None
            for batch in chunk_tasks(tasks, batch_size):
                last = runner.run(
                    batch,
                    resume=resume,
                    apply_updates=apply_updates,
                    flush_every=1,
                    track_learning_curve=bool(cfg.get("track_learning_curve", True)),
                )
            if last is None:
                last = runner.run(
                    [],
                    resume=resume,
                    apply_updates=apply_updates,
                    track_learning_curve=bool(cfg.get("track_learning_curve", True)),
                )

            results_summary.append(
                {
                    "run": run_name,
                    "seed": seed,
                    "output_dir": str(out_dir),
                    "summary": last.get("summary") if last else None,
                }
            )

    summary_path = output_root / "ablation_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(results_summary, indent=2), encoding="utf-8")
    return results_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ablation experiments from a YAML config.")
    parser.add_argument("--config", type=Path, default=Path("configs/experiments.yaml"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Override output_root from config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_ablation_from_config(args.config, override_output_root=args.output_root)
    print(json.dumps({"runs_completed": len(summary), "first": summary[0] if summary else None}, indent=2))


if __name__ == "__main__":
    main()
