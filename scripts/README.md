# Scripts Structure

This directory is now reserved for one-off jobs:

- `scripts/analysis`: EDA, normalization, clustering, utility analysis
- `scripts/visualization`: figure generation scripts

Production/reproducible pipeline entrypoints were moved to `src/cli`.

## Recommended Usage

- One-off analysis:
  - `python scripts/analysis/audit_join.py`
- One-off visualization:
  - `python scripts/visualization/visualize_fund_analytics_5axes.py`

For main runs, use:

- `python -m cli.run_recommender`
- `python -m cli.explain_recommender`
- `python -m cli.evaluate`
- `python -m cli.improve_recommender_with_utility`
- or unified entrypoint: `python main.py <pipeline>`

Utility weight fine-tuning example:
- `python main.py improve --config run_config.toml --tune-trials 30 --normalize-utility-weights`

LLM model benchmark helper script:
- `./scripts/run_llm_model_benchmark.sh`
- default models: `gpt-5-mini`, `gpt-5.4-mini`, `gpt-5.4`, `gpt-5.5`
- override example:
  - `./scripts/run_llm_model_benchmark.sh --max-eval-users 120 --top-k 3`

Model-wise heatmap (0~20) from benchmark summary CSV:
- `python scripts/visualization/visualize_llm_model_benchmark_heatmap.py --summary-csv reports/e2e/llm_model_benchmark/<timestamp>/model_benchmark_summary.csv`

## Runtime Config (TOML)

Use root config file:

- `run_config.toml`

Examples:

- `python -m cli.improve_recommender_with_utility --config run_config.toml`
- `python -m cli.TPS_Main_v2 --config run_config.toml`
- `python -m cli.evaluate_custom_v2 --config run_config.toml`
- `python -m cli.evaluate_explainer --config run_config.toml`
