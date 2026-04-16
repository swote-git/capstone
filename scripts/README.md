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

## Runtime Config (TOML)

Use root config file:

- `run_config.toml`

Examples:

- `python -m cli.improve_recommender_with_utility --config run_config.toml`
- `python -m cli.TPS_Main_v2 --config run_config.toml`
- `python -m cli.evaluate_custom_v2 --config run_config.toml`
- `python -m cli.evaluate_explainer --config run_config.toml`
