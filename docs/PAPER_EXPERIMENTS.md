# Paper experiments (ICL multi-agent)

The reward pipeline is logged in strict order for every episode:

`ground_truth_reward -> raw_peer_scores -> score_normalization -> trust_weighting -> attention_weighting -> combined_peer_reward -> final_reward -> sanity_checks`

## Smoke test (no GPU / no Ollama)

```bash
python scripts/run_smoke_ablation.py
# or
python -c "from pathlib import Path; from experiment import run_ablation_from_config; run_ablation_from_config(Path('configs/smoke.yaml'))"
```

## Full ablations (local Ollama)

1. Edit `configs/experiments.yaml`: set `models` to tags from `ollama list`, and adjust `limit` / `seeds`.
2. Run:

```bash
python -m experiment.ablation --config configs/experiments.yaml
```

Outputs go under `outputs/ablations/` plus `ablation_summary.json`.
Each run summary now also includes `verification_metrics` so you can track how often the reward pipeline stayed internally consistent across the run.

## Single run (CLI)

```bash
python run_experiment.py --backend icl --dataset gsm8k --limit 50 --apply-updates --output-dir outputs/icl-run
```

Baselines: `--backend ollama` (no in-context memory), `--backend heuristic` (fast math stand-ins).

## Analysis

```bash
python analysis/compute_statistics.py outputs/ablations/ablation_summary.json -o outputs/ablations/stats.json
python analysis/generate_tables.py outputs/ablations/stats.json -o outputs/ablations/table.tex
python analysis/plot_learning_curves.py outputs/icl-run/learning_curve.jsonl -o outputs/icl-run/curve.png --window 5
```

Useful files to inspect after a run:

- `results.jsonl`: full per-example traces
- `summary.json`: aggregate metrics plus `verification_metrics`
- `learning_curve.jsonl`: per-episode reward snapshots
- `run_manifest.json`: resolved run configuration and environment settings

## Learning curves

Each run writes `learning_curve.jsonl` next to `results.jsonl` when `track_learning_curve=True` (default in `ExperimentRunner.run`).
