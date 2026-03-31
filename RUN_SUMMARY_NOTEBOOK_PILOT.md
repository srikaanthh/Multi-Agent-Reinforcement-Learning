# Notebook Pilot Run Summary

Date: 2026-03-30

## Scope

The main notebook was executed in place with a pragmatic pilot configuration to finish end-to-end today:

- train split: 25 examples
- test split: 10 examples
- backend: `icl`
- models:
  - `qwen3.5:latest`
  - `kimi-k2.5:cloud`
  - `qwen2.5-coder:1.5b-base`
  - `qwen2.5:3b-instruct`
- trust weighting: enabled
- attention weighting: enabled
- revision rounds: 1

Artifacts:

- `outputs/notebook-train/summary.json`
- `outputs/notebook-test/summary.json`
- `outputs/notebook-train/results.jsonl`
- `outputs/notebook-test/results.jsonl`

## Headline Results

### Train summary

- `kimi-k2.5:cloud` led on mean final reward: `0.8511`
- `qwen3.5:latest` had the highest win rate: `0.48`
- `qwen2.5-coder:1.5b-base` had the weakest trust score: `0.2873`

Leaderboard snapshot:

1. `kimi-k2.5:cloud` — mean final reward `0.8511`
2. `qwen3.5:latest` — mean final reward `0.7749`
3. `qwen2.5:3b-instruct` — mean final reward `0.6368`
4. `qwen2.5-coder:1.5b-base` — mean final reward `0.4642`

### Test summary

- `kimi-k2.5:cloud` led on mean final reward: `0.6392`
- `qwen2.5-coder:1.5b-base` had the highest win rate: `0.60`
- trust ranking on test remained led by `kimi-k2.5:cloud`

Leaderboard snapshot:

1. `kimi-k2.5:cloud` — mean final reward `0.6392`
2. `qwen2.5-coder:1.5b-base` — mean final reward `0.6273`
3. `qwen3.5:latest` — mean final reward `0.5974`
4. `qwen2.5:3b-instruct` — mean final reward `0.5492`

## How The Models Rated Each Other

### Mean trust by evaluator

Train:

- `kimi-k2.5:cloud`: `0.6554`
- `qwen3.5:latest`: `0.5899`
- `qwen2.5:3b-instruct`: `0.5402`
- `qwen2.5-coder:1.5b-base`: `0.2873`

Test:

- `kimi-k2.5:cloud`: `0.5677`
- `qwen2.5:3b-instruct`: `0.5279`
- `qwen3.5:latest`: `0.4585`
- `qwen2.5-coder:1.5b-base`: `0.2145`

### Directed normalized peer scores

Train highlights:

- `qwen3.5:latest -> kimi-k2.5:cloud`: `0.7560`
- `qwen2.5:3b-instruct -> qwen3.5:latest`: `0.7800`
- `kimi-k2.5:cloud -> qwen3.5:latest`: `0.6733`
- `qwen2.5-coder:1.5b-base` was effectively flat around `0.5`, indicating weak discrimination

Test highlights:

- `qwen3.5:latest -> qwen2.5:3b-instruct`: `0.6941`
- `kimi-k2.5:cloud -> qwen3.5:latest`: `0.6567`
- `qwen2.5:3b-instruct -> qwen3.5:latest`: `0.6000`
- `qwen2.5-coder:1.5b-base` again stayed near `0.5` for everyone

## Operational Notes

- The notebook execution completed successfully in place.
- The train output directory contained earlier resumed examples, so the train run is a resumed pilot rather than a clean-from-zero run.
- The test run is a clean 10-example evaluation under the final notebook configuration.
- `qwen3.5:latest` showed the highest failure count during the pilot even though it remained competitive when successful.

## Takeaway

For review purposes, the most defensible story from this pilot is:

- the reward-verification pipeline is active and producing structured traces
- `kimi-k2.5:cloud` emerged as the most trusted evaluator in both train and test
- `qwen2.5-coder:1.5b-base` behaved like a low-discrimination judge and was downweighted by trust
- peer judgments are directional and non-symmetric, which supports the design choice to model evaluator-target relationships explicitly
