# Thin-Filer Financial Recommender

Offline ranking recommender + grounded explanation pipeline for thin-file users.

## 1) Project Scope

This project uses only the following AI Hub datasets:
- `11.통신카드CB 결합정보` (quarterly anchor)
- `09.개인 CB정보` (yearly lagged join)
- `12.금융상품정보` (item catalog)

Out of scope:
- `01~08` card tables (excluded to avoid temporal leakage and frequency mismatch)

Core outputs:
- Top-K recommendation (`deposit` / `fund` / `all`)
- Grounded explanation (template or OpenAI renderer)
- Evaluation metrics (ranking + explainer quality + understanding effect)

## 2) Runtime Architecture

Runtime pipeline:

```text
Raw Data (11, 09-lag, 12)
-> Snapshot Builder
-> User Feature Engineering
-> Product Normalization
-> Candidate Generation
-> User-Item Pair Features
-> Baseline + LGBMRanker (+ optional MoE harness)
-> Top-K Recommendation
-> Reason Extractor
-> Explanation Object
-> Renderer (Template/OpenAI)
-> Verifier
-> User Simulator (before/after explanation)
-> LLM/Rule Evaluator
-> 6 metrics (Personalization/Grounding/Clarity/Compliance/UG/MR)
```

Detailed layer I/O spec:
- [layer_io_spec_ko.md](layer_io_spec_ko.md)

Architecture docs:
- [architecture_ko.md](architecture_ko.md)
- [architecture.md](architecture.md)

## 3) Repository Layout

```text
main.py                  # unified entrypoint
run_config.toml          # runtime config (TOML)

src/
  recommender/           # ranking engine
  explainer/             # reasoning/render/verify
  evaluate/              # evaluation modules
  user_parser/           # TPS/user parsing
  cli/                   # executable pipelines
  common/                # shared config/helpers

scripts/
  analysis/              # one-off analysis scripts
  visualization/         # one-off figure scripts

reports/                 # generated reports/figures
artifacts/               # serialized model artifacts
```

## 4) Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional for OpenAI renderer/evaluator:
1. Put API key in `.env`
2. Example: `OPENAI_API_KEY=...`

## 5) Unified Entrypoint (`main.py`)

Each pipeline is started via:

```bash
python3 main.py <pipeline> [args...]
```

Available pipeline names:
- `recommend`
- `explain`
- `evaluate`
- `evaluate-explainer`
- `benchmark-llm`
- `improve`
- `tps-main`
- `evaluate-custom`
- `demo-new-user`

Check pipeline-specific help:

```bash
python3 main.py recommend --help
python3 main.py explain --help
python3 main.py evaluate-explainer --help
python3 main.py benchmark-llm --help
```

## 6) Quick Start

### 6.1 Recommend (ranking)

```bash
python3 main.py recommend --fit --sample-users 200 --max-train-users 800 --top-k 5 --family deposit
python3 main.py recommend --fit --sample-users 200 --max-train-users 800 --top-k 5 --family fund
```

### 6.2 Explain (grounded)

Template renderer:

```bash
python3 main.py explain --fit --sample-users 200 --max-train-users 800 --top-k 5 --family deposit
```

OpenAI renderer:

```bash
set -a; source .env; set +a
python3 main.py explain --fit --sample-users 200 --top-k 5 --family deposit \
  --use-llm-renderer --llm-model gpt-5-mini \
  --use-explainer-moe \
  --compliance-rules-path src/explainer/compliance_rules.txt
```

`src/explainer/compliance_rules.txt` can be edited with legal clauses:
- `금지: ...`
- `필수: ...`

### 6.3 Evaluate recommender

```bash
python3 main.py evaluate --fit --sample-users 1200 --max-train-users 800 --max-eval-users 300 --ks 5 10 --family deposit
python3 main.py evaluate --fit --sample-users 1200 --max-train-users 800 --max-eval-users 300 --ks 5 10 --family fund
```

### 6.4 Evaluate explainer (+ understanding effect)

```bash
python3 main.py evaluate-explainer \
  --fit --family deposit --sample-users 300 --max-train-users 200 --max-eval-users 80 --top-k 5 \
  --enable-understanding-eval --max-understanding-samples 120
```

### 6.5 LLM model benchmark (multi-model comparison)

```bash
set -a; source .env; set +a
python3 main.py benchmark-llm \
  --fit --family all \
  --sample-users 300 --max-train-users 200 --max-eval-users 80 --top-k 5 \
  --max-understanding-samples 40 \
  --models "gpt-5-mini" "gpt-5.4-mini" "gpt-5.4" "gpt-5.5" \
  --no-template-fallback \
  --enable-understanding-eval \
  --continue-on-error
```

Outputs:
- `reports/e2e/llm_model_benchmark/<timestamp>/model_benchmark_summary.csv`
- `.../model_benchmark_summary.json`
- `.../raw/model_benchmark_detail.json`
- `.../benchmark_verifier_ug_mr.png`

Tip:
- `chatGPT ...` 별칭은 내부적으로 API model ID로 매핑됩니다.
- 일부 모델 ID는 계정 권한/시점에 따라 사용 불가할 수 있으며, 그 경우 `status=error`로 기록됩니다.
- 이해도 채점은 LLM evaluator 기준 문항별 `0~20`(실수) 점수이며, UG는 `(after-before)/100`으로 정규화됩니다.
- 설명 없음(before) 조건에서는 `recommended_product_detail/comparison/warnings`를 숨겨 ceiling effect를 줄입니다.
- 요약 CSV에는 `mean_total_before_100`, `mean_total_after_100`, `mean_delta_total_100`가 함께 기록됩니다.
- 모델별 설명 품질 4지표에 대한 LLM 직접 채점(`0~20`)도 포함됩니다:
  - `llm_personalization_score20`
  - `llm_product_grounding_score20`
  - `llm_terminology_clarity_score20`
  - `llm_compliance_score20`

### 6.6 Utility formula tuning (weight fine-tuning)

Use configurable utility weights and optional random search:

```bash
python3 main.py improve \
  --family deposit \
  --sample-users 1200 --max-train-users 800 --max-eval-users 300 \
  --candidate-max 120 --ks 5 10 \
  --normalize-utility-weights \
  --tune-trials 30 --tune-k 5 \
  --tune-item-only \
  --out-dir reports/e2e/improved_recommender_deposit_tuned \
  --out-json reports/raw/improved_recommender_deposit_tuned.json \
  --tune-out-csv reports/raw/utility_tuning_trials_deposit.csv
```

For config-driven runs, set weights under `[improve_recommender_with_utility]` in `run_config.toml`.
If you want to tune only item utility (not pair-layer weights), use `--tune-item-only`.

### 6.6 Score MoE harness (optional)

Use a mixture of scoring experts (`ranker` + `baseline` + `utility`) with router weights:

```bash
python3 main.py recommend \
  --fit --family all --top-k 5 \
  --use-moe-harness \
  --moe-ranker-weight 0.60 \
  --moe-baseline-weight 0.25 \
  --moe-utility-weight 0.15
```

Router adjustment knobs:
- `--moe-deposit-baseline-boost`
- `--moe-fund-utility-boost`
- `--moe-low-risk-fund-penalty`

### 6.7 Explainer MoE (optional)

Use explanation-layer experts (`llm_reason` -> `llm_compliance` -> `template`) with routing:

```bash
python3 main.py explain \
  --fit --family fund --use-llm-renderer \
  --use-explainer-moe \
  --compliance-rules-path src/explainer/compliance_rules.txt \
  --explainer-moe-debug
```

This is different from score MoE harness:
- score MoE = ranking score aggregation
- explainer MoE = explanation generation/compliance orchestration

## 7) Run with `run_config.toml`

`run_config.toml` is the single root runtime config.

```bash
python3 main.py recommend --config run_config.toml
python3 main.py explain --config run_config.toml
python3 main.py evaluate --config run_config.toml
python3 main.py evaluate-explainer --config run_config.toml
python3 main.py benchmark-llm --config run_config.toml
python3 main.py improve --config run_config.toml
```

Config resolution order:
1. `[common]`
2. section for each pipeline (e.g. `[explain_recommender]`)
3. CLI args (highest priority)

Prompt path keys:
- `llm_prompt_path` -> explanation renderer prompt (e.g. `src/explainer/explain.txt`)
- `simulator_prompt_path` -> user simulator prompt
- `evaluator_prompt_path` -> evaluator prompt

MoE keys (optional, usable from `[common]` or each section):
- `use_moe_harness`
- `moe_debug`
- `moe_ranker_weight`, `moe_baseline_weight`, `moe_utility_weight`
- `moe_deposit_baseline_boost`, `moe_fund_utility_boost`, `moe_low_risk_fund_penalty`

Explainer-MoE keys (section-level):
- `use_explainer_moe`
- `explainer_moe_debug`
- `compliance_rules_path`

## 8) Recommended Execution Order

1. Environment check

```bash
source .venv/bin/activate
python3 -V
pip show lightgbm shap openai
```

2. Join audit (must-check)

```bash
PYTHONPATH=src python3 scripts/analysis/audit_join.py --sample-users 5000 --sample-size 10000
```

3. Train/evaluate by family (`deposit`, `fund`)
4. Run grounded explanation pipeline
5. Run explainer understanding evaluation (UG/MR)
6. Consolidate reports under `reports/e2e/`

## 9) Output Artifacts

Typical outputs:
- Ranking JSON (stdout):

```json
{
  "user_id": "...",
  "recommendations": [
    {"product_id": "...", "score": 0.91},
    {"product_id": "...", "score": 0.88}
  ]
}
```

- Evaluation reports: `reports/e2e/*`
- Raw metrics JSON: `reports/raw/*`
- Understanding logs (JSONL): path from `understanding_log_jsonl`

## 10) Quality Guardrails

- If `cb_join_rate` is very low, treat table `09` features as effectively unavailable.
- Avoid claiming performance from circular labels only.
- Check label collapse by family (especially fund path).
- High verifier pass rate alone is not enough; always inspect UG/MR together.
