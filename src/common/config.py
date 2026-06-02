from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict


@dataclass
class RecommenderConfig:
    data_root: Path = Path("data")
    # 원본 합성데이터 루트(환경별로 덮어쓰기 권장)
    raw_data_root: Path = Path("data")

    # 기본 디렉터리는 현재 저장소 구조 기준
    table11_dir: str = "11.통신카드CB 결합정보"
    table09_dir: str = "09.개인 CB정보"
    table12_dir: str = "12.금융상품정보"

    user_key_11: str = "CUST_ID"
    user_key_09: str = "ID"
    # "all" | "deposit" | "fund"
    recommender_family: str = "all"

    # TPS v2.1 가중치 설정: 신뢰도 중심 보조 지표
    tps_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "trust": 0.7,
            "activity": 0.15,
            "potential": 0.15,
        }
    )
    trust_overdue_weight: float = 10.0
    trust_inst_weight: float = 1.0
    activity_amt_weight: float = 0.2
    activity_cnt_weight: float = 0.4
    activity_digi_weight: float = 0.4
    potential_income_weight: float = 0.3
    potential_cb_weight: float = 0.4
    potential_tel_weight: float = 0.1
    potential_youth_weight: float = 0.2
    table11_nrows_per_file: int | None = None
    # 실험적 heuristic ID bridge는 기본 비활성화 (데이터 무결성 보호)
    enable_heuristic_id_bridge: bool = False

    candidate_min: int = 50
    candidate_max: int = 100
    top_k: int = 5
    max_train_users: int = 5000
    random_state: int = 42

    risk_threshold: float = 1.25
    investment_asset_threshold: float = 2_000_000.0
    enable_deposit_eligibility_filter: bool = True
    deposit_cluster_cap_topk: int = 1
    deposit_bank_cap_topk: int = 1

    baseline_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "risk_match": 0.35,
            "liquidity_match": 0.25,
            "horizon_match": 0.20,
            "complexity_match": 0.10,
            "digital_match": 0.10,
        }
    )

    ranker_params: Dict[str, object] = field(
        default_factory=lambda: {
            "objective": "lambdarank",
            "metric": "ndcg",
            "eval_at": [5],
            "lambdarank_truncation_level": 5,
            "label_gain": [0, 1, 3, 15],
            "n_estimators": 500,
            "learning_rate": 0.02,
            "num_leaves": 128,
            "random_state": 42,
            "verbose": -1,
        }
    )

    # Optional MoE harness over scoring experts (ranker/baseline/utility).
    use_moe_harness: bool = False
    moe_debug: bool = False
    moe_default_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "ranker": 0.60,
            "baseline": 0.25,
            "utility": 0.15,
        }
    )
    moe_deposit_baseline_boost: float = 0.05
    moe_fund_utility_boost: float = 0.10
    moe_low_risk_fund_penalty: float = 0.15
