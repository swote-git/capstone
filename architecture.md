# Architecture Overview (Latest)

## 1. Goal
- Offline Top-K financial recommendation for thin-file users
- Grounded explainability for recommendation outcomes
- Single runtime entrypoint: `main.py`

## 2. End-to-End Pipeline
```text
[Raw Data: 11, 09(lag), 12]
  -> [User Snapshot Builder]
  -> [User Feature Engineering]
  -> [Product Normalization]
  -> [Candidate Generation]
  -> [User-Item Pair Features]
  -> [Baseline + LGBMRanker]
  -> [Top-K Recommendation]
  -> [Reason Extractor]
  -> [Explanation Object]
  -> [Renderer (Template/OpenAI)]
  -> [Verifier]
```

## 3. Source Layout (src)
```text
src/
  common/
    config.py
    helpers.py
    pipeline.py

  recommender/
    engine.py

  explainer/
    service.py
    reasoning.py
    render_verify.py
    common.py
    llm_renderer.py

  evaluate/
    recommender_eval.py
    explainer_eval.py

  user_parser/
    tps.py

  cli/
    run_recommender.py
    explain_recommender.py
    evaluate.py
    evaluate_explainer.py
    improve_recommender_with_utility.py
    evaluate_custom_v2.py
    TPS_Main_v2.py
    demo_new_user_v2.py
    runtime_config.py
```

## 4. Entrypoints
### 4.1 Unified Entrypoint
- `main.py`
- Examples:
```bash
PYTHONPATH=src python3 main.py recommend --sample-users 20 --top-k 5
PYTHONPATH=src python3 main.py explain --fit --sample-users 200 --top-k 5
PYTHONPATH=src python3 main.py evaluate --fit --sample-users 1200 --max-train-users 800 --max-eval-users 300 --ks 5 10
```

### 4.2 Direct Module Runs
```bash
PYTHONPATH=src python3 -m cli.run_recommender --config run_config.toml
PYTHONPATH=src python3 -m cli.improve_recommender_with_utility --config run_config.toml
```

## 5. Configuration Policy
- Single runtime config at repository root: `run_config.toml`
- Precedence: `common` section -> script section -> CLI args (final override)

## 6. Layering Rules
- `recommender/`: ranking/modeling only
- `explainer/`: explanation pipeline + verification only
- `evaluate/`: experiment/evaluation aggregation only
- `user_parser/`: external user input / CSV parsing only
- `cli/`: orchestration-only thin entry layer

## 7. Key Refactor Outcomes
- Removed `src/thin_filer` depth and reorganized by domain
- Kept `scripts/` for one-off analysis/visualization only
- Added root `main.py` as unified entrypoint
- Explicitly separated recommender/explainer/evaluate/user parsing concerns
