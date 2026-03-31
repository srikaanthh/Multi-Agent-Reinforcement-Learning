#!/usr/bin/env python3
"""Run a fast local ablation (heuristic agents, no Ollama). From repo root: python scripts/run_smoke_ablation.py"""

from pathlib import Path

from experiment import run_ablation_from_config


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = root / "configs" / "smoke.yaml"
    run_ablation_from_config(cfg, override_output_root=None)


if __name__ == "__main__":
    main()
