# Project Summary

## What The Codebase Is Trying To Achieve

This repository is a research-oriented multi-agent reinforcement learning environment for LLMs. The main goal is to evaluate and improve multiple language-model agents on GSM8K-style math tasks while keeping the reward pipeline auditable.

The core idea is:

1. Multiple agents answer the same question.
2. Agents judge each other's answers with directed peer scores.
3. Those peer scores are normalized so one overly generous judge does not dominate.
4. Evaluator trust is estimated from historical alignment with objective correctness.
5. Attention weights decide which peer evaluations are most relevant for a specific response.
6. A peer reward is computed from normalized scores, trust, and attention.
7. The final reward mixes objective ground truth with the peer reward.

The final reward used in the environment is:

`R_j = alpha * R_gt_j + (1 - alpha) * R_peer_j`

with

`R_peer_j = sum_i(a_ij * t_i * s_ij) / sum_i(a_ij * t_i)`

where:

- `R_gt_j` is the ground-truth reward for agent `j`
- `s_ij` is evaluator `i`'s normalized score for agent `j`
- `t_i` is evaluator `i`'s trust weight
- `a_ij` is the attention weight for evaluator `i` when scoring response `j`

## Main Components

- `environment/`
  The multi-agent environment, reward pipeline, trust weighting, attention weighting, ranking, and per-episode diagnostics.

- `agents/`
  Agent abstractions and implementations:
  heuristic math agents, Ollama-backed agents, self-refine agents, and ICL memory agents.

- `data/`
  GSM8K loading and final-answer extraction.

- `experiment/`
  Batch running, resumable experiment outputs, run manifests, and summary computation.

- `analysis/`
  Statistics, LaTeX table generation, and learning-curve plotting.

- `notebook.ipynb`
  Local interactive notebook for train/test experiments and reward-verification inspection.

- `notebook_colab.ipynb`
  Colab-first notebook with smaller defaults and setup steps for hosted execution.

## Why The Reward Verification Matters

The repository treats reward correctness as a first-class concern. Every episode now logs a strict verification report under `metadata.reward_verification`, including:

- stage order
- ground-truth rewards
- raw peer-score matrix
- normalized peer-score matrix
- trust weights
- attention weights
- combined peer rewards
- final rewards
- sanity checks

This makes the system easier to debug scientifically, because reward failures are visible as soon as they happen instead of only showing up later as unstable learning behavior.
