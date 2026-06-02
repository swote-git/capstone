# 아키텍처 개요 (최신)

## 1. 목표
- Thin-file 사용자 대상 오프라인 Top-K 금융상품 추천
- 설명 가능(grounded) 추천 결과 제공
- 실행 진입점 단일화: `main.py`

## 2. 최신 파이프라인
```text
[Raw Data: 11, 09(lag), 12]
  -> [User Snapshot Builder]
  -> [User Feature Engineering]
  -> [Product Normalization]
  -> [Candidate Generation]
  -> [User-Item Pair Features]
  -> [Baseline + LGBMRanker]
  -> [MoE Harness (선택): ranker/baseline/utility 전문가 게이팅]
  -> [Top-K Recommendation]
  -> [Reason Extractor]
  -> [Explanation Object]
  -> [Renderer(Template/OpenAI)]
  -> [Verifier]
  -> [User Simulator (설명 없음/있음 비교)]
  -> [LLM/Rule Evaluator]
  -> [6개 지표 산출: Personalization/Grounding/Clarity/Compliance/UG/MR]
```

## 3. 코드 구조 (src)
```text
src/
  common/
    config.py             # RecommenderConfig
    helpers.py            # 공통 상수/유틸
    pipeline.py           # to_json + 호환 facade

  recommender/
    engine.py             # ThinFilerRecommender 핵심
    moe_harness.py        # MoE 라우터 + 전문가 점수 결합

  explainer/
    service.py            # GroundedExplainer 오케스트레이션
    reasoning.py          # reason/fact/object 생성
    render_verify.py      # 렌더링 + 검증
    common.py             # 설명 스키마/문구 유틸
    llm_renderer.py       # OpenAI 렌더러

  evaluate/
    recommender_eval.py   # 추천 평가 공용 로직
    explainer_eval.py     # 설명 평가 공용 로직
    explainer_understanding_eval.py # 이해도(UG/MR) 평가 로직

  user_parser/
    tps.py                # 사용자 입력/TPS 파싱

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

## 4. 엔트리포인트
### 4.1 단일 진입점
- `main.py`
- 사용 예:
```bash
PYTHONPATH=src python3 main.py recommend --sample-users 20 --top-k 5
PYTHONPATH=src python3 main.py explain --fit --sample-users 200 --top-k 5
PYTHONPATH=src python3 main.py evaluate --fit --sample-users 1200 --max-train-users 800 --max-eval-users 300 --ks 5 10
```

### 4.2 모듈 직접 실행
```bash
PYTHONPATH=src python3 -m cli.run_recommender --config run_config.toml
PYTHONPATH=src python3 -m cli.improve_recommender_with_utility --config run_config.toml
```

## 5. 설정 정책
- 런타임 설정 파일: 루트 `run_config.toml` 단일 사용
- 적용 순서: `common` -> 스크립트 섹션 -> CLI 인자(최종 우선)

## 6. 레이어 분리 원칙
- `recommender/`: 추천 모델링/랭킹 전담
- `explainer/`: 설명 파이프라인 + 검증 전담
- `evaluate/`: 실험/평가 집계 + 이해도(UG/MR) 평가 전담
- `user_parser/`: 외부 사용자 입력/CSV 파싱 전담
- `cli/`: 실행 오케스트레이션만 담당 (얇은 엔트리)

## 7. 설명 평가 2계층
- 설명 품질 계층:
  - `reason_coverage_rc`, `hallucination_rate_hr`, `fact_consistency_rate`, `verification_pass_rate`
  - `personalization`, `product_grounding`, `terminology_clarity`, `compliance`
- 이해 효과 계층:
  - `understanding_gain (UG)` = 설명 후 정답률 - 설명 전 정답률
  - `misinterpretation_rate (MR)` = 오해 판정 개수 / 총 질문 수

## 8. 핵심 개선 사항
- `src/thin_filer` 깊이 제거 및 도메인별 디렉토리 재구성
- `scripts`는 단발성(analysis/visualization)만 유지
- 메인 엔트리포인트 `main.py` 도입
- 추천/설명/평가/파싱 로직을 명시적으로 분리
