# Thin-Filer Financial Recommender (Offline Ranking)

This project implements a ranking-based recommender for thin-file users using only:

- `11.통신카드CB 결합정보` (quarterly anchor)
- `09.개인 CB정보` (yearly lagged features)
- `12.금융상품정보` (product catalog)

It follows the requested architecture:

1. User snapshot build (`user_id`, `as_of_date`)
2. Feature engineering (5 user feature categories)
3. Product normalization (`deposit|fund`, risk/liquidity/horizon/complexity)
4. Rule-based candidate generation (`50~200` items per user)
5. User-item pair feature build
6. Interpretable weighted baseline score
7. `LightGBMRanker` training (`lambdarank`, grouped by user)
8. Top-K recommendation

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Unified entrypoint from project root:

```bash
python3 main.py recommend --sample-users 20 --top-k 5
python3 main.py explain --fit --sample-users 200 --top-k 5
python3 main.py evaluate --fit --sample-users 1200 --max-train-users 800 --max-eval-users 300 --ks 5 10
```

Inference demo (baseline if model not trained):

```bash
PYTHONPATH=src python3 -m cli.run_recommender --sample-users 20 --top-k 5
```

Deposit-only recommender:

```bash
PYTHONPATH=src python3 -m cli.run_recommender --sample-users 20 --top-k 5 --family deposit
```

Fund-only recommender:

```bash
PYTHONPATH=src python3 -m cli.run_recommender --sample-users 20 --top-k 5 --family fund
```

Train + inference:

```bash
PYTHONPATH=src python3 -m cli.run_recommender --fit --max-train-users 1000 --sample-users 200 --top-k 5
```

Grounded explainer run (strict structured explanation + verifier):

```bash
PYTHONPATH=src python3 -m cli.explain_recommender --fit --sample-users 200 --max-train-users 800 --top-k 5
```

Grounded explainer with OpenAI LLM renderer (object-grounded verbalization):

```bash
export OPENAI_API_KEY=your_api_key
PYTHONPATH=src python3 -m cli.explain_recommender --fit --sample-users 200 --top-k 5 --use-llm-renderer --llm-model gpt-5-mini
```

Explainer batch evaluation (RC/HR/pass rate):

```bash
PYTHONPATH=src python3 -m cli.evaluate_explainer --fit --sample-users 300 --max-train-users 200 --max-eval-users 80 --top-k 5
```

Evaluation report (baseline vs ranker):

```bash
PYTHONPATH=src python3 -m cli.evaluate --sample-users 1000 --max-eval-users 300 --ks 5 10
```

Train then evaluate:

```bash
PYTHONPATH=src python3 -m cli.evaluate --fit --sample-users 1200 --max-train-users 800 --max-eval-users 300 --ks 5 10
```

Quarter filter example:

```bash
PYTHONPATH=src python3 -m cli.run_recommender --as-of-dates 2022Q2 2022Q3 --sample-users 100
```

Join integrity audit (recommended before training):

```bash
PYTHONPATH=src python3 scripts/analysis/audit_join.py --sample-users 5000 --sample-size 10000
```

Run with single root config (`run_config.toml`):

```bash
PYTHONPATH=src python3 -m cli.improve_recommender_with_utility --config run_config.toml
PYTHONPATH=src python3 -m cli.evaluate_explainer --config run_config.toml
```

## Execution Guideline (Recommended Order)

Use this order for stable and reproducible runs.

1. Environment check
```bash
source .venv/bin/activate
python3 -V
pip show lightgbm shap openai
```

2. Data join audit (must-run before model training)
```bash
PYTHONPATH=src python3 scripts/analysis/audit_join.py --sample-users 5000 --sample-size 10000
```
If `cb_join_rate` is near `0`, treat table `09` as effectively unavailable in interpretation.

3. Run family-specific recommender (recommended)
```bash
# Deposit-only
PYTHONPATH=src python3 -m cli.run_recommender --fit --family deposit --sample-users 200 --max-train-users 800 --top-k 5

# Fund-only
PYTHONPATH=src python3 -m cli.run_recommender --fit --family fund --sample-users 200 --max-train-users 800 --top-k 5
```

4. Run family-specific improved E2E report
```bash
# Deposit v2
PYTHONPATH=src python3 -m cli.improve_recommender_with_utility \
  --family deposit --sample-users 120 --max-train-users 80 --max-eval-users 40 \
  --candidate-max 80 --ks 5 10 \
  --out-dir reports/e2e/improved_recommender_deposit_v2 \
  --out-json reports/raw/e2e_improved_recommender_deposit_v2.json

# Fund v2
PYTHONPATH=src python3 -m cli.improve_recommender_with_utility \
  --family fund --sample-users 120 --max-train-users 80 --max-eval-users 40 \
  --candidate-max 80 --ks 5 10 \
  --out-dir reports/e2e/improved_recommender_fund_v2 \
  --out-json reports/raw/e2e_improved_recommender_fund_v2.json
```

5. Run explanation pipeline (template or LLM)
```bash
# Template renderer
PYTHONPATH=src python3 -m cli.explain_recommender --fit --family deposit --sample-users 200 --top-k 5

# OpenAI LLM renderer (grounded verbalization only)
export OPENAI_API_KEY=your_api_key
PYTHONPATH=src python3 -m cli.explain_recommender \
  --fit --family deposit --sample-users 200 --top-k 5 \
  --use-llm-renderer --llm-model gpt-5-mini
```

6. Validate explainer metrics
```bash
PYTHONPATH=src python3 -m cli.evaluate_explainer \
  --fit --family deposit --sample-users 300 --max-train-users 200 --max-eval-users 80 --top-k 5
```

### Where to check results
- Main summary: `reports/e2e/main_report.md`
- Family reports:
  - `reports/e2e/improved_recommender_deposit_v2/report.md`
  - `reports/e2e/improved_recommender_fund_v2/report.md`
- Raw JSON metrics:
  - `reports/raw/e2e_improved_recommender_deposit_v2.json`
  - `reports/raw/e2e_improved_recommender_fund_v2.json`
  - `reports/raw/e2e_evaluate_explainer.json`

### Quick quality checklist
- `cb_join_rate` is not silently ignored in analysis.
- `fund ind_proxy_label` is not collapsed to all zeros.
- `deposit` positive-per-query diagnostics are non-degenerate.
- Do not use only circular metrics (`proxy_label`, `hybrid_label`) as main KPI.

## Output format

```json
{
  "user_id": "123",
  "recommendations": [
    {"product_id": "...", "score": 0.91},
    {"product_id": "...", "score": 0.88},
    {"product_id": "...", "score": 0.85}
  ]
}
```

## Notes

- Tables `01~08` are not used by design (to avoid frequency leakage).
- Time alignment uses quarterly `11` snapshot and previous-year December from `09`.
- If direct interaction labels are unavailable, rule-based weak labels (`0~3`) are used for ranker training.
- Evaluation output includes snapshot join quality (`cb_join_rate`) and candidate-size distribution.
- Run join audit first; if ID overlap is near zero, treat `09` features as unavailable until an ID bridge exists.
- Explanations are generated by structured evidence pipeline (reason extractor -> fact retriever -> explanation object -> renderer -> verifier), not direct free-form LLM.
