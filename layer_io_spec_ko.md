# 실행 파이프라인 Layer I/O 명세 (최신 코드 기준)

이 문서는 현재 코드(`src/`) 기준으로 각 레이어의 역할, 중요 포인트, 입력/출력 데이터 형식을 정리한 실행 명세입니다.

적용 범위:
- 추천 파이프라인: `recommender/engine.py`
- 설명 파이프라인: `explainer/*`
- 설명 효과 평가 파이프라인: `evaluate/explainer_eval.py`, `evaluate/explainer_understanding_eval.py`

---

## 0) 전체 흐름

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
  -> [Product Fact Retriever]
  -> [Explanation Object Builder]
  -> [Renderer(Template/OpenAI)]
  -> [Verifier]
  -> [User Simulator (설명 없음/있음)]
  -> [LLM/Rule Evaluator]
  -> [6개 지표 산출]
```

---

## 1) User Snapshot Builder

코드:
- `ThinFilerRecommender.build_user_snapshots`
- `_load_table11`, `_load_table09`

역할:
- 분기(anchor) 기준 사용자 스냅샷 생성
- 11번(quarterly)과 09번(yearly lag) 시간 정렬 조인

중요 포인트:
- 시간 정렬 핵심: `lagged_cb_ym = (anchor_ym의 전년도 12월)`
- 조인 실패율(`cb_join_rate`)을 반드시 품질 지표로 기록
- 필요 시 heuristic bridge를 쓰되 기본은 비활성화

입력 형식:
- Table 11 CSV (다수 파일): `CUST_ID` + `TABLE11_NEEDED_COLS`
- Table 09 CSV (다수 파일): `ID` + `STDT`, `C1M210000`
- 런타임 파라미터: `as_of_dates`, `sample_users`

출력 형식 (DataFrame):
- 식별/시간: `CUST_ID`, `anchor_ym`, `as_of_date`, `lagged_cb_ym`
- 조인 결과: `ID`, `STDT`, `cb_join_found`
- 원본 변수: 11/09 선택 컬럼

---

## 2) User Feature Engineering

코드:
- `_engineer_user_features`
- `_build_user_component_features`, `_build_user_preference_features`

역할:
- 스냅샷에서 추천용 사용자 피처 생성
- TPS v2 스코어군(`tps_score`, `tps_trust`, `tps_activity`, `tps_potential`) 계산

중요 포인트:
- 추천 성능과 설명 품질 모두에 직접 영향
- 누락값/스케일 보정(`_safe_col`, `_clip01`)이 안정성 핵심
- TPS는 현재 모델 피처로 사용되며, utility 공식을 직접 치환하지는 않음

입력 형식:
- Snapshot DataFrame (1단계 출력)

출력 형식 (DataFrame에 컬럼 추가):
- 성향/제약: `risk_tol`, `liquidity_need`, `horizon_pref`, `complexity_tol`, `amount_bin`, `investment_possible`
- 행동/안정성: `digital_behavior_freq`, `card_usage_stability`, `telecom_payment_consistency` 등
- TPS: `tps_score`, `tps_trust`, `tps_activity`, `tps_potential`

---

## 3) Product Normalization

코드:
- `_load_and_normalize_products`

역할:
- 12번 원본(은행수신상품/공모펀드상품)을 공통 아이템 스키마로 변환

중요 포인트:
- 추천/설명 공통 기준 필드 확보
- 상품군(`product_family`) 분리 추천의 기반

입력 형식:
- `은행수신상품.csv`
- `공모펀드상품.csv`

출력 형식 (정규화 상품 카탈로그 DataFrame):
- 공통 필수:
  - `product_id`, `product_name`, `product_family`
  - `risk_level(0~3)`, `liquidity_level(0~3)`, `horizon(short|mid|long)`
  - `complexity(0~2)`, `min_amount_bin(0~3)`, `fee_level`, `principal_variation(0/1)`
  - `max_rate`, `horizon_code(0/1/2)`

---

## 4) Candidate Generation

코드:
- `generate_candidates`

역할:
- 사용자별 전체 상품군을 규칙 기반으로 축소

중요 포인트:
- Full pair 폭발 방지(연산량/학습시간 절감)
- 리스크/유동성/가입금액 조건 기반 1차 필터가 품질 핵심

입력 형식:
- 단일 사용자 행(`pd.Series`)
- 정규화 상품 카탈로그
- 파라미터: `candidate_min`, `candidate_max`, `risk_threshold`

출력 형식 (DataFrame):
- 사용자별 후보 상품 테이블(보통 50~100개)
- 컬럼: 상품 정규화 스키마(3단계 출력)

---

## 5) User-Item Pair Feature Builder

코드:
- `_add_pair_features`
- `_compute_pair_match_features`
- `_compute_baseline_score`

역할:
- `(user, item)` 쌍 특징 계산
- baseline 점수 생성

중요 포인트:
- 모델 입력의 핵심 구조
- 설명의 reason signal과 직접 연결되는 피처군

입력 형식:
- 사용자 1행 DataFrame
- 후보 상품 DataFrame

출력 형식 (Pair DataFrame):
- 매칭 피처:
  - `risk_match`, `liquidity_match`, `horizon_match`
  - `complexity_match`, `amount_feasibility`, `family_match`, `digital_match`
- baseline:
  - `baseline_score` (가중합)

---

## 6) Weak Label Builder (학습 시)

코드:
- `_build_labels`

역할:
- 상호작용 로그가 없는 환경에서 랭킹 학습용 약한 라벨 생성

중요 포인트:
- 라벨 설계가 모델 성능/해석의 상한을 결정
- circular evaluation 여부를 항상 점검해야 함

입력 형식:
- Pair DataFrame + `max_rate`

출력 형식:
- `pd.Series[int64]` in `{0,1,2,3}`

---

## 7) Baseline + LGBMRanker

코드:
- `build_training_dataset`, `fit`

역할:
- Baseline은 해석 가능한 기준선
- LGBMRanker는 그룹 랭킹 학습(`group = user query size`)

중요 포인트:
- 그룹 정의가 정확해야 lambdarank가 정상 동작
- 피처 컬럼은 `TRAIN_FEATURE_COLUMNS` 교집합 기준으로 자동 선택

입력 형식:
- 사용자 스냅샷 DataFrame
- 후보 생성 및 pair 피처 생성 결과

출력 형식:
- 학습 중간:
  - `X: DataFrame[feature_columns]`
  - `y: Series[label]`
  - `group: List[int]`
- 최종:
  - `self.model` (LGBMRanker), `self.feature_columns`

---

## 8) Top-K Recommendation

코드:
- `recommend`, `batch_recommend`

역할:
- 모델 점수(없으면 baseline)로 상위 K개 추천 반환

중요 포인트:
- 제품군 분리 추천(`deposit`, `fund`, `all`) 운영 가능
- 실제 서비스 출력 포맷의 기준 레이어

입력 형식:
- 단일 사용자 snapshot (`pd.Series`)
- `k`

출력 형식 (JSON):

```json
{
  "user_id": "CUST_...",
  "recommendations": [
    {"product_id": "...", "score": 0.9132},
    {"product_id": "...", "score": 0.8821}
  ]
}
```

---

## 9) Reason Extractor

코드:
- `explainer/reasoning.py::extract_reasons`

역할:
- 추천 점수에 기여한 상위 feature reason 추출

중요 포인트:
- SHAP 가능 시 로컬 기여도 우선
- 불가 시 importance/휴리스틱 fallback

입력 형식:
- `pair_row` (추천된 단일 상품의 user-item row)
- `rec.feature_columns`, 모델

출력 형식:
- `List[ReasonSignal]`
- 각 원소:
  - `feature`, `value`, `impact(positive|negative)`, `contribution`

---

## 10) Product Fact Retriever

코드:
- `retrieve_product_facts`

역할:
- 설명용 human-readable 상품 팩트 생성

중요 포인트:
- raw 컬럼명을 그대로 노출하지 않고 의미 스키마로 매핑
- verifier의 fact consistency 기준 데이터

입력 형식:
- `pair_row`

출력 형식 (dict):
- `family`, `risk`, `liquidity`, `horizon`, `complexity`, `principal_variation`
- `product_meta` (원시 수치)
- `match_detail` (매칭 피처 수치)

---

## 11) Explanation Object Builder

코드:
- `build_explanation_object`

역할:
- 설명의 단일 소스 오브 트루스(explanation object) 구성

중요 포인트:
- LLM은 이 객체를 언어화만 해야 함
- 사용자 특성, 상품 특성, 이유 신호, 비교, 경고를 구조적으로 분리

입력 형식:
- `user_snapshot`
- `product_facts`
- `reason_signals`

출력 형식 (dict, 핵심 키):
- `user_summary`
- `user_profile_detail`
- `recommended_product`
- `recommended_product_detail`
- `reason_signals`
- `model_reasons`
- `comparison`
- `warnings`

---

## 12) Renderer (Template/OpenAI)

코드:
- 템플릿: `render_verify.render_explanation`
- LLM: `llm_renderer.OpenAILLMRenderer.render`

역할:
- explanation object를 사용자 가독 텍스트로 변환

중요 포인트:
- LLM 시스템 프롬프트 파일(`explain.txt`) 커스터마이징 가능
- verifier 실패 시 템플릿 fallback 옵션 지원

입력 형식:
- `explanation_object` (dict)

출력 형식:
- `rendered_explanation` (문자열)
- 섹션 형식:
  - `[상품 정보 요약]`
  - `[추천 이유]`
  - `[유의사항]`
  - `[대안 비교]`
  - `[한줄 요약]`

---

## 13) Verifier

코드:
- `verify`, `reason_alignment`, `check_fact_consistency`, `hallucination_rate`, `contains_forbidden_claims`

역할:
- 생성 설명의 정합성/안전성 검증

중요 포인트:
- 금융 문구 금지 패턴 필터링(보장/무위험/강권유)
- 설명 품질을 수치화하여 자동 품질 게이트로 사용

입력 형식:
- `rendered_text`
- `explanation_object`
- `product_facts`

출력 형식 (dict):
- `reason_alignment: float`
- `fact_consistency: bool`
- `hallucination_rate: float`
- `forbidden_claims: List[str]`
- `passed: bool`

---

## 14) User Simulator (설명 없음/있음)

코드:
- `ExplainerUnderstandingEvaluator._simulate_answers`

역할:
- 설명 제시 전/후 사용자 Q&A 응답 생성

중요 포인트:
- UG/MR 계산의 기반 데이터
- LLM 시뮬레이터 미사용 시 규칙 기반 fallback 동작

입력 형식:
- `recommendation_payload`
- `explanation_text` (없음/있음)
- 고정 질문 Q1~Q5

출력 형식:
- `answers_before: {"Q1":"...","Q2":"...",...}`
- `answers_after: {"Q1":"...","Q2":"...",...}`

---

## 15) LLM/Rule Evaluator

코드:
- `_score_answers`, `_rule_evaluate_answers`

역할:
- 정답(ground truth) 대비 질문별 이해 정오 판정

중요 포인트:
- 설명 품질과 이해 효과를 분리해 측정
- 오해(misinterpretation) 유형을 별도 기록

입력 형식:
- `ground_truth`
- `answers_before/answers_after`
- `explanation_object`

출력 형식:
- `scores_before/after: {"Q1":0|1,...,"Q5":0|1}`
- `total_before/after: 0~5`
- `misinterpretations_before/after: List[str]`

---

## 16) 6개 지표 산출

코드:
- `ExplainerUnderstandingEvaluator._quality_scores`
- `ExplainerUnderstandingEvaluator.summarize`

역할:
- 설명 품질 4개 + 이해 효과 2개 지표 산출

중요 포인트:
- 품질(말을 잘했는가)과 효과(이해가 늘었는가)를 분리해서 판단
- 최종 보고서 KPI의 기준

입력 형식:
- 개별 평가 레코드 배열

출력 형식:
- `personalization`
- `product_grounding`
- `terminology_clarity`
- `compliance`
- `understanding_gain`
- `misinterpretation_rate`

---

## 17) 실행 관점의 핵심 체크포인트

- `cb_join_rate`가 매우 낮으면 09 기여는 사실상 없는 것으로 해석
- 후보 수(`candidate_count_mean`, p90)가 비정상적으로 작거나 크지 않은지 점검
- 라벨 붕괴(특정 family의 전부 0 라벨) 여부 점검
- verifier pass만 높고 reason diversity가 낮으면 설명 퇴화 가능성 점검
- UG(이해도 향상)와 MR(오해율)을 함께 보고 최종 판단

