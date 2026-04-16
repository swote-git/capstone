# Scripts Structure

- `scripts/analysis`: data analysis, normalization, clustering, and utility analysis
- `scripts/visualization`: visualization generation scripts
- `scripts/recommender`: recommendation run, explanation, and improvement pipelines
- `scripts/evaluation`: evaluation scripts

## Recommended Usage

Use categorized paths as the primary entry points:

- `python scripts/recommender/run_recommender.py`
- `python scripts/recommender/explain_recommender.py`
- `python scripts/evaluation/evaluate.py`
- `python scripts/visualization/visualize_fund_analytics_5axes.py`

## Runtime Config (TOML / YAML)

You can now run major scripts with a shared config file via `--config`.

- Example files:
  - `configs/runtime/example_run.toml`
  - `configs/runtime/example_run.yaml`
- Supported formats: `.toml`, `.yaml/.yml`, `.json`
- YAML requires `PyYAML` (`pip install pyyaml`)

Examples:

- `python scripts/recommender/improve_recommender_with_utility.py --config configs/runtime/example_run.toml`
- `python scripts/recommender/TPS_Main_v2.py --config configs/runtime/example_run.toml`
- `python scripts/evaluation/evaluate_custom_v2.py --config configs/runtime/example_run.toml`
- `python scripts/evaluation/evaluate_explainer.py --config configs/runtime/example_run.toml`

Behavior:

- `common` section applies to all scripts.
- Script-specific section overrides `common`.
- CLI flags still override config values.
