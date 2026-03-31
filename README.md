# Multi-Agent RL Environment For LLM Reward Verification

This repository implements a multi-agent evaluation and improvement loop for LLMs on GSM8K-style math tasks. It combines objective correctness, peer judgment, trust weighting, attention weighting, revision rounds, and optional contextual-bandit RL updates.

## Core Reward Design

The environment verifies rewards in strict order:

`ground_truth_reward -> raw_peer_scores -> score_normalization -> trust_weighting -> attention_weighting -> combined_peer_reward -> final_reward -> sanity_checks`

The final reward is:

`R_j = alpha * R_gt_j + (1 - alpha) * R_peer_j`

with

`R_peer_j = sum_i(a_ij * t_i * s_ij) / sum_i(a_ij * t_i)`

This keeps ground truth dominant while still letting trusted, relevant peers influence the final score.

## Repository Layout

- `environment/`
  Multi-agent environment and reward pipeline.

- `agents/`
  Agent base classes and implementations:
  heuristic agents, Ollama agents, self-refine agents, and ICL agents.

- `data/`
  GSM8K loading and answer extraction.

- `experiment/`
  Experiment runner, resumable manifests, and aggregate summaries.

- `analysis/`
  Statistics, tables, and learning-curve plots.

- `configs/`
  YAML configs for smoke tests and ablations.

- `notebook.ipynb`
  Local notebook for train/test runs and reward-verification inspection.

- `notebook_colab.ipynb`
  Colab-first notebook with smaller defaults and setup logic.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For development tools:

```bash
pip install -e .[dev]
```

## Quick Start

### Fast smoke test

```bash
python scripts/run_smoke_ablation.py
```

### Single CLI run

```bash
python run_experiment.py \
  --backend heuristic \
  --dataset gsm8k \
  --split test \
  --limit 5 \
  --output-dir outputs/quick-run
```

### Local notebook

Open `notebook.ipynb` and run the cells from top to bottom.

### Colab notebook

Open `notebook_colab.ipynb` in Colab.
It defaults to the `heuristic` backend because standard Colab does not include a local Ollama server.

## Important Run Outputs

Each run writes:

- `results.jsonl`
  Full per-example responses, rewards, ranking, and metadata.

- `summary.json`
  Aggregate leaderboard metrics and `verification_metrics`.

- `learning_curve.jsonl`
  Per-episode learning-curve rows.

- `run_manifest.json`
  Resolved run configuration plus environment settings.

## Reward Verification Output

Each episode stores a detailed report under:

`metadata.reward_verification`

Useful fields include:

- `stage_order`
- `ground_truth_reward`
- `raw_peer_scores`
- `score_normalization`
- `trust_weighting`
- `attention_weighting`
- `combined_peer_reward`
- `final_reward`
- `sanity_checks`

## Current Improvement Areas

The codebase is in solid shape for research iteration, but the main areas to keep improving are:

- model-backend ergonomics for non-Ollama environments
- richer aggregate reporting over reward-verification health
- clearer top-level documentation and setup guidance
- notebook UX when outputs do not exist yet

This pass addresses the documentation gap, adds aggregate verification metrics to summaries, exposes the missing attention temperature config in experiment entrypoints, and hardens both notebooks against missing output files.

## Tests

```bash
pytest -q tests/test_environment.py
```

## More Detail

- `PROJECT_SUMMARY.md`
- `docs/PAPER_EXPERIMENTS.md`
