#!/usr/bin/env python3
"""Plot learning curves from ``learning_curve.jsonl`` (per-episode metrics)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Install matplotlib: pip install matplotlib") from exc


def load_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def plot_mean_gt_reward(
    rows: List[Dict[str, Any]],
    *,
    agent: Optional[str] = None,
    window: int = 1,
    out_path: Optional[Path] = None,
    title: str = "Mean ground-truth reward (per episode)",
) -> None:
    if not rows:
        raise ValueError("No learning curve rows to plot.")

    indices = [int(r["episode_index"]) for r in rows]
    if agent:
        series = [
            float(r["per_agent"][agent]["gt_reward"])
            for r in rows
            if agent in r.get("per_agent", {})
        ]
        if len(series) != len(indices):
            indices = indices[: len(series)]
    else:
        # Mean across agents at each step
        series = []
        for r in rows:
            pa = r.get("per_agent") or {}
            vals = [float(v["gt_reward"]) for v in pa.values()]
            series.append(sum(vals) / len(vals) if vals else 0.0)

    if window > 1:
        smoothed: List[float] = []
        for i in range(len(series)):
            start = max(0, i - window + 1)
            chunk = series[start : i + 1]
            smoothed.append(sum(chunk) / len(chunk))
        series = smoothed

    plt.figure(figsize=(8, 4))
    plt.plot(indices[: len(series)], series, marker="o", markersize=2, linewidth=1)
    plt.xlabel("Episode")
    plt.ylabel("GT reward" + (f" ({agent})" if agent else " (mean)"))
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
    else:
        plt.show()
    plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot learning curves from learning_curve.jsonl")
    p.add_argument("learning_curve", type=Path, help="Path to learning_curve.jsonl")
    p.add_argument("--agent", default=None, help="Plot one agent; default: mean across agents.")
    p.add_argument("--window", type=int, default=1, help="Rolling average window.")
    p.add_argument("-o", "--output", type=Path, default=None, help="Save figure to PNG.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.learning_curve)
    plot_mean_gt_reward(rows, agent=args.agent, window=args.window, out_path=args.output)


if __name__ == "__main__":
    main()
