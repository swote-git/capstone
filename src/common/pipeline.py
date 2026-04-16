from __future__ import annotations

import json
from typing import Dict

from common.config import RecommenderConfig
from common.helpers import (
    MISSING_SENTINELS,
    PAIR_MATCH_FEATURES,
    TABLE09_NEEDED_COLS,
    TABLE11_NEEDED_COLS,
    TRAIN_FEATURE_COLUMNS,
    _bucket_amount,
    _clip01,
    _dcg_at_k,
    _extract_first_int,
    _lagged_cb_ym,
    _ndcg_at_k,
    _parse_amount_bin,
    _parse_ym_from_filename,
    _read_csv_selected,
    _safe_col,
    _to_numeric,
    _ym_to_quarter,
)
from recommender.engine import DepositRecommender, FundRecommender, ThinFilerRecommender


# Backward-compatible public API
__all__ = [
    "RecommenderConfig",
    "ThinFilerRecommender",
    "DepositRecommender",
    "FundRecommender",
    "to_json",
    "MISSING_SENTINELS",
    "TABLE11_NEEDED_COLS",
    "TABLE09_NEEDED_COLS",
    "PAIR_MATCH_FEATURES",
    "TRAIN_FEATURE_COLUMNS",
    "_to_numeric",
    "_clip01",
    "_safe_col",
    "_parse_ym_from_filename",
    "_ym_to_quarter",
    "_lagged_cb_ym",
    "_extract_first_int",
    "_parse_amount_bin",
    "_bucket_amount",
    "_read_csv_selected",
    "_dcg_at_k",
    "_ndcg_at_k",
]


def to_json(data: Dict[str, object]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)
