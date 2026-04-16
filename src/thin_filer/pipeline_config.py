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

    # TPS v2.0 가중치 설정 (사용자 정의)
    tps_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "trust": 0.4,
            "activity": 0.3,
            "potential": 0.3
        }
    )
    # 실험적 heuristic ID bridge는 기본 비활성화 (데이터 무결성 보호)
    enable_heuristic_id_bridge: bool = False

    candidate_min: int = 50
    candidate_max: int = 100
    top_k: int = 5
    max_train_users: int = 5000
    random_state: int = 42

    risk_threshold: float = 1.25
    investment_asset_threshold: float = 2_000_000.0

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
            "n_estimators": 200,
            "learning_rate": 0.05,
            "num_leaves": 64,
            "random_state": 42,
            "verbose": -1,
        }
    )
