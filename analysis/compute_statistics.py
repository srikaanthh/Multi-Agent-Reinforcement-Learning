#!/usr/bin/env python3
"""Aggregate statistics across ablation runs (multiple seeds / output dirs)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var)


def _ci95_mean(xs: Sequence[float]) -> tuple[float, float]:
    """Normal approximation 95%% CI for the mean."""
    if not xs:
        return (0.0, 0.0)
    m = _mean(xs)
    s = _std(xs)
    n = len(xs)
    half = 1.96 * s / math.sqrt(n) if n else 0.0
    return (m - half, m + half)


def summarize_ablation_summary(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Expected a JSON list of run summaries.")

    by_run: Dict[str, List[float]] = {}
    by_run_win: Dict[str, List[float]] = {}

    for row in data:
        run = str(row.get("run", ""))
        summary = row.get("summary") or {}
        leaderboard = summary.get("leaderboard") or []
        metrics = summary.get("agent_metrics") or {}

        if leaderboard:
            top = leaderboard[0]
            by_run.setdefault(run, []).append(float(top.get("mean_final_reward", 0.0)))
            by_run_win.setdefault(run, []).append(float(top.get("win_rate", 0.0)))
        elif metrics:
            # Fallback: mean of mean_final_reward across agents
            vals = [float(v.get("mean_final_reward", 0.0)) for v in metrics.values()]
            by_run.setdefault(run, []).append(_mean(vals))

    report: Dict[str, Any] = {}
    for run, vals in sorted(by_run.items()):
        report[run] = {
            "n": len(vals),
            "mean_final_reward_top_agent": _mean(vals),
            "std": _std(vals),
            "ci95": _ci95_mean(vals),
        }
        if run in by_run_win and by_run_win[run]:
            report[run]["mean_win_rate_top_agent"] = _mean(by_run_win[run])
        rows = [row for row in data if str(row.get("run", "")) == run]
        if rows:
            top_acc = []
            top_fail = []
            top_trust = []
            for row in rows:
                leaderboard = (row.get("summary") or {}).get("leaderboard") or []
                if leaderboard:
                    top_acc.append(float(leaderboard[0].get("gt_accuracy", 0.0)))
                    top_fail.append(float(leaderboard[0].get("failure_rate", 0.0)))
                    trust = leaderboard[0].get("mean_trust_weight")
                    if isinstance(trust, (int, float)):
                        top_trust.append(float(trust))
            if top_acc:
                report[run]["mean_gt_accuracy_top_agent"] = _mean(top_acc)
            if top_fail:
                report[run]["mean_failure_rate_top_agent"] = _mean(top_fail)
            if top_trust:
                report[run]["mean_trust_weight_top_agent"] = _mean(top_trust)

    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute statistics from ablation_summary.json")
    p.add_argument("ablation_summary", type=Path)
    p.add_argument("-o", "--output", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report = summarize_ablation_summary(args.ablation_summary)
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
