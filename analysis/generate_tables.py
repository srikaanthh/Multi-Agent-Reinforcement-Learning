#!/usr/bin/env python3
"""Generate LaTeX tables from ablation statistics JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def to_latex_table(report: Dict[str, Any], *, caption: str, label: str) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\begin{tabular}{lccc}",
        r"\hline",
        r"Run & $n$ & Mean final reward (top) & 95\% CI \\",
        r"\hline",
    ]
    for run, stats in sorted(report.items()):
        n = int(stats.get("n", 0))
        m = float(stats.get("mean_final_reward_top_agent", 0.0))
        ci = stats.get("ci95")
        if isinstance(ci, (list, tuple)) and len(ci) == 2:
            lo, hi = float(ci[0]), float(ci[1])
        else:
            lo = hi = m
        lines.append(f"{run} & {n} & {m:.4f} & [{lo:.4f}, {hi:.4f}] \\\\")
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate LaTeX from stats JSON")
    p.add_argument("stats_json", type=Path, help="Output of compute_statistics or similar")
    p.add_argument("-o", "--output", type=Path, default=Path("table.tex"))
    p.add_argument("--caption", default="Ablation results (mean final reward of leaderboard top).")
    p.add_argument("--label", default="tab:marl_ablation")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report = json.loads(args.stats_json.read_text(encoding="utf-8"))
    tex = to_latex_table(report, caption=args.caption, label=args.label)
    args.output.write_text(tex, encoding="utf-8")
    print(tex)


if __name__ == "__main__":
    main()
