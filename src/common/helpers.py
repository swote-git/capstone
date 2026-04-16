from __future__ import annotations

import math
import re
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd


MISSING_SENTINELS = {
    "": np.nan,
    "*": np.nan,
    "NA": np.nan,
    "N/A": np.nan,
    8888888.8: np.nan,
    888888888: np.nan,
    99999999: np.nan,
    -99999999: np.nan,
}

TABLE11_NEEDED_COLS: List[str] = [
    "AGE",
    "PYE_C1M210000",
    "PYE_C18233003",
    "PYE_C18233004",
    "PYE_MAX_DLQ_DAY",
    "R3M_FOOD_AMT",
    "R3M_DEP_AMT",
    "R3M_MART_AMT",
    "R3M_E_COMM_AMT",
    "R3M_TRAVEL_AMT",
    "R3M_EDU_AMT",
    "R3M_ITRT_FIN_PAY",
    "QOQ_R3M_MBR_USE_CNT_RTC",
    "QOQ_CD_USE_AMT_RTC",
    "CD_USE_AMT",
    "TOT_ASST",
    "DAR",
    "ROP",
    "R3M_MBR_USE_CNT",
    "R6M_MBR_USE_CNT",
    "R9M_MBR_USE_CNT",
    "R12M_MBR_USE_CNT",
    "APP_GD",
    "B1Y_MOB_OS",
    "R3M_ITRT_COMM_MESSENGER",
    "R3M_ITRT_FIN_ASSET",
    "R3M_ITRT_FIN_STOCK",
    "PYE_AL012G011",
]

TABLE09_NEEDED_COLS: List[str] = ["STDT", "C1M210000"]

PAIR_MATCH_FEATURES: List[str] = [
    "risk_match",
    "liquidity_match",
    "horizon_match",
    "complexity_match",
    "amount_feasibility",
    "family_match",
    "digital_match",
]

TRAIN_FEATURE_COLUMNS: List[str] = [
    "risk_match",
    "liquidity_match",
    "horizon_match",
    "complexity_match",
    "amount_feasibility",
    "family_match",
    "digital_match",
    "risk_level",
    "liquidity_level",
    "complexity",
    "min_amount_bin",
    "principal_variation",
    "max_rate",
    "risk_tol",
    "liquidity_need",
    "complexity_tol",
    "amount_bin",
    "investment_possible",
    "credit_depth",
    "credit_recency",
    "telecom_payment_consistency",
    "card_usage_stability",
    "spending_vs_balance_ratio",
    "digital_behavior_freq",
    "tps_score",
    "tps_trust",
    "tps_activity",
    "tps_potential",
]


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.replace(MISSING_SENTINELS), errors="coerce")


def _clip01(values: pd.Series) -> pd.Series:
    return values.fillna(0.0).clip(lower=0.0, upper=1.0)


def _safe_col(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return _to_numeric(df[col]).fillna(default)


def _parse_ym_from_filename(path: Path) -> int:
    match = re.match(r"(\d{6})_", path.name)
    if not match:
        raise ValueError(f"Cannot parse YYYYMM prefix from filename: {path.name}")
    return int(match.group(1))


def _ym_to_quarter(ym: int) -> str:
    y = ym // 100
    m = ym % 100
    q = (m - 1) // 3 + 1
    return f"{y}Q{q}"


def _lagged_cb_ym(anchor_ym: int) -> int:
    return ((anchor_ym // 100) - 1) * 100 + 12


def _extract_first_int(text: object, default: int = 0) -> int:
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return default
    s = str(text)
    m = re.search(r"-?\d+", s)
    return int(m.group(0)) if m else default


def _parse_amount_bin(value: object) -> int:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 0
    s = str(value)
    if "제한없음" in s:
        return 0
    n = _extract_first_int(s, default=0)
    if n <= 0:
        return 0
    if n < 100:
        return 0
    if n < 500:
        return 1
    if n < 1000:
        return 2
    return 3


def _bucket_amount(amount: pd.Series) -> pd.Series:
    amount = amount.fillna(0.0)
    return pd.cut(
        amount,
        bins=[-1, 1_000_000, 5_000_000, 20_000_000, float("inf")],
        labels=[0, 1, 2, 3],
    ).astype("int64")


def _read_csv_selected(path: Path, desired_cols: Sequence[str]) -> pd.DataFrame:
    try:
        header = pd.read_csv(path, nrows=0, encoding="utf-8")
        usecols = [c for c in desired_cols if c in header.columns]
        return pd.read_csv(path, usecols=usecols, dtype=str, low_memory=False, encoding="utf-8")
    except UnicodeDecodeError:
        header = pd.read_csv(path, nrows=0, encoding="cp949")
        usecols = [c for c in desired_cols if c in header.columns]
        return pd.read_csv(path, usecols=usecols, dtype=str, low_memory=False, encoding="cp949")


def _dcg_at_k(relevance: np.ndarray, k: int) -> float:
    top = relevance[:k]
    if top.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, top.size + 2))
    gains = np.power(2.0, top) - 1.0
    return float(np.sum(gains * discounts))


def _ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    if y_true.size == 0:
        return 0.0
    order = np.argsort(-y_score)
    ideal = np.argsort(-y_true)
    dcg = _dcg_at_k(y_true[order], k)
    idcg = _dcg_at_k(y_true[ideal], k)
    if idcg <= 0:
        return 0.0
    return dcg / idcg
