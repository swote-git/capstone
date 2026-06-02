#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from common.config import RecommenderConfig
from common.helpers import _ndcg_at_k
from recommender.engine import ThinFilerRecommender
from .runtime_config import parse_args_with_config

try:
    from lightgbm import LGBMRanker
except Exception:
    LGBMRanker = None

FIT_CORE_TERMS = [
    "risk_match",
    "liquidity_match",
    "horizon_match",
    "complexity_match",
    "amount_feasibility",
]
DEP_REAL_TERMS = [
    "amount_feasibility",
    "digital_match",
    "complexity_simplicity(=1-complexity/2)",
    "liquidity_match",
]
FUND_REAL_TERMS = [
    "risk_match",
    "family_match",
    "horizon_match",
    "amount_feasibility",
]
HYBRID_TERMS = [
    "fit_core",
    "item_utility_prior",
    "realizability",
    "rate_factor",
]
DEP_LABEL_TERMS = [
    "item_utility_prior",
    "rate_norm",
    "amount_feasibility",
    "complexity_simplicity",
    "liquidity_match",
]
FUND_LABEL_TERMS = [
    "item_utility_prior",
    "risk_adj_norm",
    "family_match",
    "horizon_match",
    "digital_match",
]
DEP_ITEM_UTILITY_TERMS = ["U_rate", "U_bonus", "U_feasibility", "U_liquidity"]
FUND_ITEM_UTILITY_TERMS = ["U_return", "U_risk_eff", "U_cost_eff", "U_liquidity", "U_simplicity"]
UTILITY_FOCUS_FEATURES = [
    "hybrid_utility_score",
    "item_utility_prior",
    "realizability",
    "rate_factor",
    "risk_match",
    "liquidity_match",
    "horizon_match",
    "complexity_match",
    "amount_feasibility",
    "family_match",
    "digital_match",
]


def set_korean_font() -> str:
    candidates = ["NanumGothic", "Noto Sans CJK KR", "Noto Sans KR", "AppleGothic", "Malgun Gothic"]
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = next((c for c in candidates if c in available), "DejaVu Sans")
    plt.rcParams["font.family"] = chosen
    plt.rcParams["axes.unicode_minus"] = False
    return chosen


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Improve recommender with split utility priors (deposit/fund)")
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--family", choices=["all", "deposit", "fund"], default="all")
    p.add_argument("--sample-users", type=int, default=1200)
    p.add_argument("--max-train-users", type=int, default=800)
    p.add_argument("--max-eval-users", type=int, default=300)
    p.add_argument("--ks", nargs="+", type=int, default=[5, 10])
    p.add_argument("--candidate-max", type=int, default=120)
    p.add_argument("--out-dir", type=Path, default=Path("reports/improved_recommender"))
    p.add_argument("--out-json", type=Path, default=Path("reports/raw/improved_recommender_report.json"))
    p.add_argument("--fit-core-weights", nargs=5, type=float, default=[0.30, 0.25, 0.20, 0.15, 0.10])
    p.add_argument("--dep-real-weights", nargs=4, type=float, default=[0.45, 0.25, 0.20, 0.10])
    p.add_argument("--fund-real-weights", nargs=4, type=float, default=[0.40, 0.30, 0.20, 0.10])
    p.add_argument("--hybrid-weights", nargs=4, type=float, default=[0.40, 0.35, 0.20, 0.05])
    p.add_argument("--dep-label-weights", nargs=5, type=float, default=[0.35, 0.20, 0.20, 0.15, 0.10])
    p.add_argument("--fund-label-weights", nargs=5, type=float, default=[0.35, 0.25, 0.15, 0.15, 0.10])
    p.add_argument("--dep-item-weights", nargs=4, type=float, default=[0.45, 0.25, 0.20, 0.10])
    p.add_argument("--fund-item-weights", nargs=5, type=float, default=[0.35, 0.25, 0.20, 0.10, 0.10])
    p.add_argument("--fund-low-risk-penalty", type=float, default=0.10)
    p.add_argument("--fund-gate-risk-min", type=float, default=0.35)
    p.add_argument("--fund-gate-family-min", type=float, default=0.50)
    p.add_argument("--normalize-utility-weights", action="store_true")
    p.add_argument("--tune-trials", type=int, default=0, help="Random search trials for utility weights (0 disables tuning)")
    p.add_argument("--tune-seed", type=int, default=42)
    p.add_argument("--tune-k", type=int, default=5)
    p.add_argument("--tune-out-csv", type=Path, default=Path("reports/raw/utility_tuning_trials.csv"))
    p.add_argument("--tune-item-only", action="store_true", help="Tune only item utility weights; keep pair-layer utility params fixed")
    return parse_args_with_config(p, section="improve_recommender_with_utility")


@dataclass
class UtilityParams:
    fit_core_weights: List[float]
    dep_real_weights: List[float]
    fund_real_weights: List[float]
    hybrid_weights: List[float]
    dep_label_weights: List[float]
    fund_label_weights: List[float]
    dep_item_weights: List[float]
    fund_item_weights: List[float]
    fund_low_risk_penalty: float
    fund_gate_risk_min: float
    fund_gate_family_min: float
    normalized: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _normalize_nonneg_weights(weights: Sequence[float]) -> List[float]:
    arr = np.asarray([max(0.0, float(v)) for v in weights], dtype=float)
    s = float(arr.sum())
    if s <= 0:
        return [1.0 / len(arr)] * len(arr)
    return (arr / s).tolist()


def build_utility_params(args: argparse.Namespace) -> UtilityParams:
    params = UtilityParams(
        fit_core_weights=[float(x) for x in args.fit_core_weights],
        dep_real_weights=[float(x) for x in args.dep_real_weights],
        fund_real_weights=[float(x) for x in args.fund_real_weights],
        hybrid_weights=[float(x) for x in args.hybrid_weights],
        dep_label_weights=[float(x) for x in args.dep_label_weights],
        fund_label_weights=[float(x) for x in args.fund_label_weights],
        dep_item_weights=[float(x) for x in args.dep_item_weights],
        fund_item_weights=[float(x) for x in args.fund_item_weights],
        fund_low_risk_penalty=float(args.fund_low_risk_penalty),
        fund_gate_risk_min=float(args.fund_gate_risk_min),
        fund_gate_family_min=float(args.fund_gate_family_min),
        normalized=False,
    )
    if args.normalize_utility_weights:
        params.fit_core_weights = _normalize_nonneg_weights(params.fit_core_weights)
        params.dep_real_weights = _normalize_nonneg_weights(params.dep_real_weights)
        params.fund_real_weights = _normalize_nonneg_weights(params.fund_real_weights)
        params.hybrid_weights = _normalize_nonneg_weights(params.hybrid_weights)
        params.dep_label_weights = _normalize_nonneg_weights(params.dep_label_weights)
        params.fund_label_weights = _normalize_nonneg_weights(params.fund_label_weights)
        params.dep_item_weights = _normalize_nonneg_weights(params.dep_item_weights)
        params.fund_item_weights = _normalize_nonneg_weights(params.fund_item_weights)
        params.normalized = True
    return params


def split_users(snapshots: pd.DataFrame, user_col: str, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    users = snapshots[user_col].drop_duplicates().sample(frac=1.0, random_state=42)
    cutoff = max(1, int(len(users) * train_ratio))
    train_users = set(users.iloc[:cutoff])
    train_df = snapshots[snapshots[user_col].isin(train_users)].copy()
    eval_df = snapshots[~snapshots[user_col].isin(train_users)].copy()
    if eval_df.empty:
        eval_df = train_df.copy()
    return train_df, eval_df


def load_item_priors(deposit_csv: Path, fund_csv: Path, family: str = "all") -> pd.DataFrame:
    dep = pd.read_csv(deposit_csv, usecols=["product_id", "deposit_utility", "U_rate", "U_bonus", "U_feasibility", "U_liquidity"])
    dep = dep.sort_values(["product_id", "deposit_utility"], ascending=[True, False]).drop_duplicates("product_id")
    dep = dep.rename(columns={"deposit_utility": "item_utility_prior"})
    dep["product_family"] = "deposit"

    fnd = pd.read_csv(fund_csv, usecols=["product_id", "fund_utility", "U_return", "U_risk_eff", "U_cost_eff", "U_liquidity", "U_simplicity"])
    fnd = fnd.sort_values(["product_id", "fund_utility"], ascending=[True, False]).drop_duplicates("product_id")
    fnd = fnd.rename(columns={"fund_utility": "item_utility_prior"})
    fnd["product_family"] = "fund"

    pri = pd.concat([dep, fnd], ignore_index=True)
    if family in {"deposit", "fund"}:
        pri = pri[pri["product_family"].eq(family)].copy()
    pri["product_id"] = pri["product_id"].astype(str)
    return pri


def clip01(s: pd.Series) -> pd.Series:
    return s.fillna(0.0).clip(0.0, 1.0)


def add_hybrid_features(pair: pd.DataFrame, params: UtilityParams) -> pd.DataFrame:
    pair = pair.copy()
    w_fit = params.fit_core_weights
    w_dep_real = params.dep_real_weights
    w_fund_real = params.fund_real_weights
    w_hybrid = params.hybrid_weights

    # fitness from pair matches
    fit_core = (
        w_fit[0] * pair["risk_match"]
        + w_fit[1] * pair["liquidity_match"]
        + w_fit[2] * pair["horizon_match"]
        + w_fit[3] * pair["complexity_match"]
        + w_fit[4] * pair["amount_feasibility"]
    )

    # realizability: deposit/fund rules differ
    dep_real = clip01(
        w_dep_real[0] * pair["amount_feasibility"]
        + w_dep_real[1] * pair["digital_match"]
        + w_dep_real[2] * (1.0 - pair["complexity"] / 2.0)
        + w_dep_real[3] * pair["liquidity_match"]
    )
    fund_real = clip01(
        w_fund_real[0] * pair["risk_match"]
        + w_fund_real[1] * pair["family_match"]
        + w_fund_real[2] * pair["horizon_match"]
        + w_fund_real[3] * pair["amount_feasibility"]
    )
    pair["realizability"] = np.where(pair["product_family"].eq("deposit"), dep_real, fund_real)

    pair["item_utility_prior"] = clip01(pair.get("item_utility_prior", pd.Series(0.5, index=pair.index)))

    # mild rate factor to keep financial attractiveness
    maxr = pd.to_numeric(pair.get("max_rate", 0), errors="coerce").fillna(0)
    denom = float(np.nanpercentile(np.abs(maxr), 95) + 1e-6)
    pair["rate_factor"] = clip01(maxr / max(denom, 1e-6))

    pair["hybrid_utility_score"] = clip01(
        w_hybrid[0] * fit_core
        + w_hybrid[1] * pair["item_utility_prior"]
        + w_hybrid[2] * pair["realizability"]
        + w_hybrid[3] * pair["rate_factor"]
    )

    return pair


def build_labels_from_hybrid(pair: pd.DataFrame) -> pd.Series:
    u = pair["hybrid_utility_score"]
    if len(u) < 6:
        return pd.Series(np.where(u >= u.median(), 2, 1), index=pair.index, dtype="int64")
    q80, q55, q30 = u.quantile([0.80, 0.55, 0.30]).tolist()
    y = np.select([u >= q80, u >= q55, u >= q30], [3, 2, 1], default=0)
    return pd.Series(y, index=pair.index, dtype="int64")


def build_proxy_label_independent(pair: pd.DataFrame, params: UtilityParams) -> pd.Series:
    pair = pair.copy()
    rate = pd.to_numeric(pair.get("max_rate", 0), errors="coerce").fillna(0.0)
    rmin, rmax = float(rate.min()), float(rate.max())
    rate_norm = (rate - rmin) / (rmax - rmin + 1e-9)

    risk_adj_rate = rate / (1.0 + pd.to_numeric(pair.get("risk_level", 0), errors="coerce").fillna(0.0))
    ra_min, ra_max = float(risk_adj_rate.min()), float(risk_adj_rate.max())
    risk_adj_norm = (risk_adj_rate - ra_min) / (ra_max - ra_min + 1e-9)

    item = clip01(pd.to_numeric(pair.get("item_utility_prior", 0.5), errors="coerce").fillna(0.5))
    feas = clip01(pd.to_numeric(pair.get("amount_feasibility", 0), errors="coerce").fillna(0))
    digital = clip01(pd.to_numeric(pair.get("digital_match", 0), errors="coerce").fillna(0))
    horizon = clip01(pd.to_numeric(pair.get("horizon_match", 0), errors="coerce").fillna(0))
    family = clip01(pd.to_numeric(pair.get("family_match", 0), errors="coerce").fillna(0))
    risk_match = clip01(pd.to_numeric(pair.get("risk_match", 0), errors="coerce").fillna(0))
    complexity = clip01(1.0 - pd.to_numeric(pair.get("complexity", 0), errors="coerce").fillna(0) / 2.0)
    liquidity = clip01(pd.to_numeric(pair.get("liquidity_match", 0), errors="coerce").fillna(0))
    low_risk_penalty = (
        (pd.to_numeric(pair.get("risk_tol", 1.0), errors="coerce").fillna(1.0) < 1.25)
        & (pd.to_numeric(pair.get("principal_variation", 0), errors="coerce").fillna(0) > 0)
    ).astype(float)

    w_dep = params.dep_label_weights
    w_fund = params.fund_label_weights
    dep_score = (
        w_dep[0] * item
        + w_dep[1] * rate_norm
        + w_dep[2] * feas
        + w_dep[3] * complexity
        + w_dep[4] * liquidity
    )
    fund_score = (
        w_fund[0] * item
        + w_fund[1] * risk_adj_norm
        + w_fund[2] * family
        + w_fund[3] * horizon
        + w_fund[4] * digital
        - params.fund_low_risk_penalty * low_risk_penalty
    )
    is_deposit = pair["product_family"].eq("deposit")
    score = np.where(is_deposit, dep_score, fund_score)
    score = clip01(pd.Series(score, index=pair.index))

    # Family-specific gating:
    # - deposit: keep strict amount feasibility gate
    # - fund: do not collapse by amount_feasibility (often non-informative for funds)
    #         use risk compatibility + family consistency as realizability gate.
    fund_gate = ((risk_match >= params.fund_gate_risk_min) & (family >= params.fund_gate_family_min)).astype(float)
    gate = np.where(is_deposit, (feas > 0).astype(float), fund_gate)
    score = score.where(gate > 0, 0.0)

    if len(score) < 6:
        y = np.where(score >= score.median(), 2, 1)
        y = np.where(gate <= 0, 0, y)
        return pd.Series(y, index=pair.index, dtype="int64")

    q80, q55, q30 = score.quantile([0.80, 0.55, 0.30]).tolist()
    y = np.select([score >= q80, score >= q55, score >= q30], [3, 2, 1], default=0)
    y = np.where(gate <= 0, 0, y)
    return pd.Series(y, index=pair.index, dtype="int64")


def apply_item_utility_prior(pair: pd.DataFrame, params: UtilityParams) -> pd.DataFrame:
    pair = pair.copy()
    is_deposit = pair["product_family"].eq("deposit")

    dep_parts = []
    for col in DEP_ITEM_UTILITY_TERMS:
        dep_parts.append(pd.to_numeric(pair.get(col, 0.0), errors="coerce").fillna(0.0))
    dep_prior = (
        params.dep_item_weights[0] * dep_parts[0]
        + params.dep_item_weights[1] * dep_parts[1]
        + params.dep_item_weights[2] * dep_parts[2]
        + params.dep_item_weights[3] * dep_parts[3]
    )

    fund_parts = []
    for col in FUND_ITEM_UTILITY_TERMS:
        fund_parts.append(pd.to_numeric(pair.get(col, 0.0), errors="coerce").fillna(0.0))
    fund_prior = (
        params.fund_item_weights[0] * fund_parts[0]
        + params.fund_item_weights[1] * fund_parts[1]
        + params.fund_item_weights[2] * fund_parts[2]
        + params.fund_item_weights[3] * fund_parts[3]
        + params.fund_item_weights[4] * fund_parts[4]
    )

    prior = np.where(is_deposit, dep_prior, fund_prior)
    pair["item_utility_prior"] = clip01(pd.Series(prior, index=pair.index))
    return pair


def summarize_label_diagnostics(train_data: pd.DataFrame, eval_data: pd.DataFrame) -> Dict[str, object]:
    diag: Dict[str, object] = {}

    fund_eval = eval_data[eval_data["product_family"].eq("fund")].copy()
    if not fund_eval.empty:
        diag["fund_eval_amount_feasibility_dist"] = (
            fund_eval["amount_feasibility"].value_counts(dropna=False).sort_index().to_dict()
        )
        diag["fund_eval_ind_proxy_label_dist"] = (
            fund_eval["ind_proxy_label"].value_counts(dropna=False).sort_index().to_dict()
        )
        diag["fund_eval_positive_rate_label_ge2"] = float((fund_eval["ind_proxy_label"] >= 2).mean())

    dep_train = train_data[train_data["product_family"].eq("deposit")].copy()
    if not dep_train.empty:
        pos_per_query = dep_train.groupby("query_id")["label"].apply(lambda x: int((x >= 2).sum()))
        diag["deposit_train_positive_per_query"] = {
            "count": float(pos_per_query.count()),
            "mean": float(pos_per_query.mean()),
            "min": float(pos_per_query.min()),
            "p25": float(pos_per_query.quantile(0.25)),
            "p50": float(pos_per_query.quantile(0.50)),
            "p75": float(pos_per_query.quantile(0.75)),
            "max": float(pos_per_query.max()),
            "zero_positive_query_rate": float((pos_per_query == 0).mean()),
        }

    return diag


def apply_utility_features_and_labels(
    pair: pd.DataFrame,
    rec: ThinFilerRecommender,
    params: UtilityParams,
) -> pd.DataFrame:
    pair = apply_item_utility_prior(pair, params)
    pair = add_hybrid_features(pair, params)
    pair["hybrid_label"] = build_labels_from_hybrid(pair)
    pair["ind_proxy_label"] = build_proxy_label_independent(pair, params)
    pair["label"] = pair["ind_proxy_label"].astype("int64")
    if "proxy_label" not in pair.columns:
        pair["proxy_label"] = rec._build_labels(pair)
    return pair


def candidate_pair_for_user_base(rec: ThinFilerRecommender, user_row: pd.Series, priors: pd.DataFrame, candidate_max: int) -> pd.DataFrame:
    cands = rec.generate_candidates(user_row, max_candidates=candidate_max)
    pair = rec._add_pair_features(pd.DataFrame([user_row]), cands)
    pair["product_id"] = pair["product_id"].astype(str)
    prior_cols = [
        "product_id",
        "item_utility_prior",
        "U_rate",
        "U_bonus",
        "U_feasibility",
        "U_liquidity",
        "U_return",
        "U_risk_eff",
        "U_cost_eff",
        "U_simplicity",
    ]
    prior_cols = [c for c in prior_cols if c in priors.columns]
    pair = pair.merge(priors[prior_cols], on="product_id", how="left")
    pair["proxy_label"] = rec._build_labels(pair)
    return pair


def build_base_dataset(
    rec: ThinFilerRecommender,
    snapshots: pd.DataFrame,
    priors: pd.DataFrame,
    candidate_max: int,
    max_users: int,
) -> Tuple[pd.DataFrame, List[int]]:
    users = snapshots[rec.config.user_key_11].drop_duplicates()
    if len(users) > max_users:
        keep = users.sample(n=max_users, random_state=42)
        snapshots = snapshots[snapshots[rec.config.user_key_11].isin(keep)].copy()

    groups: List[int] = []
    rows: List[pd.DataFrame] = []
    for _, user_row in snapshots.iterrows():
        pair = candidate_pair_for_user_base(rec, user_row, priors, candidate_max)
        rows.append(pair)
        groups.append(len(pair))

    data = pd.concat(rows, ignore_index=True)
    return data, groups


def _ndcg_metric(data: pd.DataFrame, label_col: str, score_col: str, k: int) -> float:
    scores: List[float] = []
    for _, g in data.groupby("query_id"):
        y = g[label_col].to_numpy(dtype=float)
        s = g[score_col].to_numpy(dtype=float)
        scores.append(_ndcg_at_k(y, s, int(k)))
    return float(np.mean(scores)) if scores else 0.0


def _dirichlet_around(base_weights: Sequence[float], rng: np.random.Generator, concentration: float = 24.0) -> List[float]:
    base = np.asarray([max(1e-6, float(x)) for x in base_weights], dtype=float)
    base = base / base.sum()
    alpha = np.maximum(base * concentration, 1e-3)
    return rng.dirichlet(alpha).tolist()


def sample_utility_params(base: UtilityParams, rng: np.random.Generator) -> UtilityParams:
    return UtilityParams(
        fit_core_weights=_dirichlet_around(base.fit_core_weights, rng),
        dep_real_weights=_dirichlet_around(base.dep_real_weights, rng),
        fund_real_weights=_dirichlet_around(base.fund_real_weights, rng),
        hybrid_weights=_dirichlet_around(base.hybrid_weights, rng),
        dep_label_weights=_dirichlet_around(base.dep_label_weights, rng),
        fund_label_weights=_dirichlet_around(base.fund_label_weights, rng),
        dep_item_weights=_dirichlet_around(base.dep_item_weights, rng),
        fund_item_weights=_dirichlet_around(base.fund_item_weights, rng),
        fund_low_risk_penalty=float(rng.uniform(0.03, 0.20)),
        fund_gate_risk_min=float(rng.uniform(0.20, 0.60)),
        fund_gate_family_min=float(rng.uniform(0.30, 0.80)),
        normalized=True,
    )


def sample_item_utility_only_params(base: UtilityParams, rng: np.random.Generator) -> UtilityParams:
    return UtilityParams(
        fit_core_weights=list(base.fit_core_weights),
        dep_real_weights=list(base.dep_real_weights),
        fund_real_weights=list(base.fund_real_weights),
        hybrid_weights=list(base.hybrid_weights),
        dep_label_weights=list(base.dep_label_weights),
        fund_label_weights=list(base.fund_label_weights),
        dep_item_weights=_dirichlet_around(base.dep_item_weights, rng),
        fund_item_weights=_dirichlet_around(base.fund_item_weights, rng),
        fund_low_risk_penalty=float(base.fund_low_risk_penalty),
        fund_gate_risk_min=float(base.fund_gate_risk_min),
        fund_gate_family_min=float(base.fund_gate_family_min),
        normalized=True,
    )


def tune_utility_params(
    rec: ThinFilerRecommender,
    eval_base: pd.DataFrame,
    base_params: UtilityParams,
    tune_trials: int,
    tune_seed: int,
    tune_k: int,
    tune_item_only: bool = False,
) -> Tuple[UtilityParams, pd.DataFrame]:
    rng = np.random.default_rng(tune_seed)
    trials: List[Dict[str, Any]] = []
    best_params = base_params
    best_score = -1.0

    for trial_idx in range(tune_trials + 1):
        if trial_idx == 0:
            params = base_params
            trial_name = "baseline"
        else:
            params = sample_item_utility_only_params(base_params, rng) if tune_item_only else sample_utility_params(base_params, rng)
            trial_name = f"trial_{trial_idx}"

        eval_data = apply_utility_features_and_labels(eval_base.copy(), rec, params)
        score_proxy = _ndcg_metric(eval_data, "proxy_label", "hybrid_utility_score", k=tune_k)
        score_ind = _ndcg_metric(eval_data, "ind_proxy_label", "hybrid_utility_score", k=tune_k)
        objective = 0.7 * score_proxy + 0.3 * score_ind

        row: Dict[str, Any] = {
            "trial": trial_name,
            "objective": float(objective),
            "hybrid_vs_proxy_ndcg": float(score_proxy),
            "hybrid_vs_ind_proxy_ndcg": float(score_ind),
            "fund_low_risk_penalty": float(params.fund_low_risk_penalty),
            "fund_gate_risk_min": float(params.fund_gate_risk_min),
            "fund_gate_family_min": float(params.fund_gate_family_min),
            "dep_item_weights": json.dumps(params.dep_item_weights, ensure_ascii=False),
            "fund_item_weights": json.dumps(params.fund_item_weights, ensure_ascii=False),
            "fit_core_weights": json.dumps(params.fit_core_weights, ensure_ascii=False),
            "dep_real_weights": json.dumps(params.dep_real_weights, ensure_ascii=False),
            "fund_real_weights": json.dumps(params.fund_real_weights, ensure_ascii=False),
            "hybrid_weights": json.dumps(params.hybrid_weights, ensure_ascii=False),
            "dep_label_weights": json.dumps(params.dep_label_weights, ensure_ascii=False),
            "fund_label_weights": json.dumps(params.fund_label_weights, ensure_ascii=False),
        }
        trials.append(row)
        if objective > best_score:
            best_score = objective
            best_params = params

    trial_df = pd.DataFrame(trials).sort_values("objective", ascending=False).reset_index(drop=True)
    return best_params, trial_df


def _weighted_formula(weights: Sequence[float], terms: Sequence[str]) -> str:
    parts = [f"{float(w):.4f}*{t}" for w, t in zip(weights, terms)]
    return " + ".join(parts)


def eval_methods(data: pd.DataFrame, ks: Sequence[int], label_col: str) -> Dict[str, float]:
    out: Dict[str, List[float]] = {}
    for m in ["baseline_score", "hybrid_utility_score", "model_score"]:
        for k in ks:
            out[f"{m}_ndcg@{k}"] = []

    for _, g in data.groupby("query_id"):
        y = g[label_col].to_numpy(dtype=float)
        for m in ["baseline_score", "hybrid_utility_score", "model_score"]:
            s = g[m].to_numpy(dtype=float)
            for k in ks:
                out[f"{m}_ndcg@{k}"].append(_ndcg_at_k(y, s, int(k)))

    return {k: float(np.mean(v)) if v else 0.0 for k, v in out.items()}


def plot_metric_bars(metrics: Dict[str, float], ks: Sequence[int], out_path: Path) -> None:
    rows = []
    for k in ks:
        rows.append({"method": "baseline", "k": f"@{k}", "ndcg": metrics[f"baseline_score_ndcg@{k}"]})
        rows.append({"method": "hybrid_rule", "k": f"@{k}", "ndcg": metrics[f"hybrid_utility_score_ndcg@{k}"]})
        rows.append({"method": "lgbm_model", "k": f"@{k}", "ndcg": metrics[f"model_score_ndcg@{k}"]})
    mdf = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=mdf, x="k", y="ndcg", hue="method", ax=ax)
    ax.set_title("개선 추천시스템 성능 비교 (NDCG)")
    ax.set_xlabel("k")
    ax.set_ylabel("NDCG")
    for p in ax.patches:
        h = p.get_height()
        ax.text(p.get_x() + p.get_width() / 2, h + 0.003, f"{h:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_score_distributions(data: pd.DataFrame, out_path: Path) -> None:
    d = pd.DataFrame(
        {
            "baseline_score": data["baseline_score"],
            "hybrid_utility_score": data["hybrid_utility_score"],
            "model_score": data["model_score"],
        }
    ).melt(var_name="method", value_name="score")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.kdeplot(data=d, x="score", hue="method", linewidth=2, ax=ax)
    ax.set_title("점수 분포 비교 (baseline vs hybrid vs model)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_feature_importance(model: LGBMRanker, feature_cols: List[str], out_path: Path) -> None:
    imp = pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_})
    imp = imp.sort_values("importance", ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(data=imp, x="importance", y="feature", color="#2E86AB", ax=ax)
    ax.set_title("Top 20 Feature Importance (LGBMRanker)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_utility_feature_importance(model: LGBMRanker, feature_cols: List[str], out_path: Path) -> pd.DataFrame:
    imp = pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_})
    uimp = imp[imp["feature"].isin(UTILITY_FOCUS_FEATURES)].copy()
    if uimp.empty:
        return uimp
    uimp = uimp.sort_values("importance", ascending=False)
    total = float(uimp["importance"].sum())
    uimp["importance_share"] = np.where(total > 0, uimp["importance"] / total, 0.0)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(data=uimp, x="importance_share", y="feature", color="#1F77B4", ax=ax)
    ax.set_title("Utility 관련 Pair-Feature Importance 비중")
    ax.set_xlabel("share within utility-focused features")
    ax.set_ylabel("feature")
    for i, r in uimp.reset_index(drop=True).iterrows():
        ax.text(r["importance_share"] + 0.005, i, f"{r['importance_share']:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return uimp


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.tune_out_csv.parent.mkdir(parents=True, exist_ok=True)
    font_name = set_korean_font()
    utility_params = build_utility_params(args)

    cfg = RecommenderConfig(data_root=args.data_root, recommender_family=args.family)
    rec = ThinFilerRecommender(cfg)

    snapshots = rec.build_user_snapshots(sample_users=args.sample_users)
    rec.load_products()

    priors = load_item_priors(
        Path("data/processed/product12_deposit_utility_index.csv"),
        Path("data/processed/product12_fund_utility_index.csv"),
        family=args.family,
    )

    train_snap, eval_snap = split_users(snapshots, cfg.user_key_11, train_ratio=0.8)

    train_base, train_group = build_base_dataset(rec, train_snap, priors, args.candidate_max, args.max_train_users)
    eval_base, _ = build_base_dataset(rec, eval_snap, priors, args.candidate_max, args.max_eval_users)

    # query ids for grouped evaluation
    train_base = train_base.reset_index(drop=True)
    eval_base = eval_base.reset_index(drop=True)
    train_base["query_id"] = train_base[cfg.user_key_11].astype(str) + "::" + train_base["as_of_date"].astype(str)
    eval_base["query_id"] = eval_base[cfg.user_key_11].astype(str) + "::" + eval_base["as_of_date"].astype(str)

    tuning_trials_df: pd.DataFrame | None = None
    if args.tune_trials > 0:
        utility_params, tuning_trials_df = tune_utility_params(
            rec=rec,
            eval_base=eval_base,
            base_params=utility_params,
            tune_trials=int(args.tune_trials),
            tune_seed=int(args.tune_seed),
            tune_k=int(args.tune_k),
            tune_item_only=bool(args.tune_item_only),
        )
        tuning_trials_df.to_csv(args.tune_out_csv, index=False, encoding="utf-8-sig")

    train_data = apply_utility_features_and_labels(train_base.copy(), rec, utility_params)
    eval_data = apply_utility_features_and_labels(eval_base.copy(), rec, utility_params)

    feature_cols = [
        "risk_match", "liquidity_match", "horizon_match", "complexity_match", "amount_feasibility",
        "family_match", "digital_match", "risk_level", "liquidity_level", "complexity", "min_amount_bin",
        "principal_variation", "max_rate", "risk_tol", "liquidity_need", "complexity_tol", "amount_bin",
        "investment_possible", "credit_depth", "credit_recency", "telecom_payment_consistency",
        "card_usage_stability", "spending_vs_balance_ratio", "digital_behavior_freq",
        "item_utility_prior", "realizability", "rate_factor", "hybrid_utility_score",
    ]
    feature_cols = [c for c in feature_cols if c in train_data.columns]

    if LGBMRanker is None:
        raise ImportError("lightgbm is required for improved recommender run.")

    model = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=240,
        learning_rate=0.05,
        num_leaves=64,
        random_state=42,
        verbose=-1,
    )

    model.fit(
        train_data[feature_cols].fillna(0.0),
        train_data["label"].astype(int),
        group=train_group,
    )

    eval_data["model_score"] = model.predict(eval_data[feature_cols].fillna(0.0))

    metrics_train_label = eval_methods(eval_data, ks=args.ks, label_col="label")
    metrics_proxy_label = eval_methods(eval_data, ks=args.ks, label_col="proxy_label")
    metrics_hybrid_label = eval_methods(eval_data, ks=args.ks, label_col="hybrid_label")
    diagnostics = summarize_label_diagnostics(train_data, eval_data)

    # figures
    plot_metric_bars(metrics_proxy_label, args.ks, args.out_dir / "01_ndcg_comparison.png")
    plot_score_distributions(eval_data, args.out_dir / "02_score_distribution.png")
    plot_feature_importance(model, feature_cols, args.out_dir / "03_feature_importance.png")
    utility_imp_df = plot_utility_feature_importance(model, feature_cols, args.out_dir / "05_utility_feature_importance.png")

    # top recommendation diversity check
    tops: List[pd.DataFrame] = []
    for qid, g in eval_data.groupby("query_id"):
        tops.append(g.sort_values("model_score", ascending=False).head(5))
    top_df = pd.concat(tops, ignore_index=True)
    fam = top_df["product_family"].value_counts(normalize=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=fam.index, y=fam.values, color="#16A085", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Top-5 추천 상품군 비중")
    ax.set_ylabel("ratio")
    for i, v in enumerate(fam.values):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center")
    fig.tight_layout()
    fig.savefig(args.out_dir / "04_top5_family_mix.png", dpi=180)
    plt.close(fig)

    summary = {
        "font": font_name,
        "family": args.family,
        "utility_params": utility_params.to_dict(),
        "utility_tuning": {
            "enabled": bool(args.tune_trials > 0),
            "trials": int(args.tune_trials),
            "seed": int(args.tune_seed),
            "k": int(args.tune_k),
            "out_csv": str(args.tune_out_csv),
            "tune_item_only": bool(args.tune_item_only),
        },
        "snapshot_quality": rec.snapshot_quality_report(snapshots),
        "data": {
            "sample_users_arg": args.sample_users,
            "train_rows": int(len(train_data)),
            "eval_rows": int(len(eval_data)),
            "train_queries": int(train_data["query_id"].nunique()),
            "eval_queries": int(eval_data["query_id"].nunique()),
            "candidate_max": int(args.candidate_max),
            "feature_count": int(len(feature_cols)),
        },
        "metrics_train_label": metrics_train_label,
        "metrics_proxy_label": metrics_proxy_label,
        "metrics_hybrid_label": metrics_hybrid_label,
        "label_diagnostics": diagnostics,
        "top5_family_mix": {k: float(v) for k, v in fam.to_dict().items()},
        "formulae": {
            "item_utility_deposit": _weighted_formula(utility_params.dep_item_weights, DEP_ITEM_UTILITY_TERMS),
            "item_utility_fund": _weighted_formula(utility_params.fund_item_weights, FUND_ITEM_UTILITY_TERMS),
            "fit_core": _weighted_formula(utility_params.fit_core_weights, FIT_CORE_TERMS),
            "realizability_deposit": _weighted_formula(utility_params.dep_real_weights, DEP_REAL_TERMS),
            "realizability_fund": _weighted_formula(utility_params.fund_real_weights, FUND_REAL_TERMS),
            "hybrid_utility_score": _weighted_formula(utility_params.hybrid_weights, HYBRID_TERMS),
            "ind_proxy_label_deposit_score": _weighted_formula(utility_params.dep_label_weights, DEP_LABEL_TERMS),
            "ind_proxy_label_fund_score": _weighted_formula(utility_params.fund_label_weights, FUND_LABEL_TERMS)
            + f" - {utility_params.fund_low_risk_penalty:.4f}*low_risk_penalty",
        },
        "tuning_method": {
            "search": "random search around baseline via Dirichlet sampling",
            "dirichlet_concentration": 24.0,
            "objective": "0.7*NDCG(hybrid_utility_score, proxy_label) + 0.3*NDCG(hybrid_utility_score, ind_proxy_label)",
            "selection": "best objective on held-out eval queries",
            "scope": "item utility only" if args.tune_item_only else "item + pair utility params",
        },
        "notes": [
            "Applied split utility priors: deposit_utility for deposits, fund_utility for funds.",
            "Added pair-level realizability to prevent over-rewarding hard-to-achieve products.",
            "Used hybrid utility weak labels for ranking supervision.",
            "Primary metric for model fit is ind_proxy_label on held-out users; proxy/hybrid metrics are auxiliary and may be circular.",
        ],
    }
    if not utility_imp_df.empty:
        summary["utility_feature_importance"] = utility_imp_df.to_dict(orient="records")
    if tuning_trials_df is not None and not tuning_trials_df.empty:
        summary["utility_tuning"]["best_objective"] = float(tuning_trials_df.iloc[0]["objective"])
        summary["utility_tuning"]["best_trial"] = str(tuning_trials_df.iloc[0]["trial"])

    warnings: List[str] = []
    fund_dist = diagnostics.get("fund_eval_ind_proxy_label_dist", {})
    if fund_dist:
        non_zero = sum(v for k, v in fund_dist.items() if int(k) > 0)
        total = sum(fund_dist.values())
        if total > 0 and non_zero / total < 0.01:
            warnings.append("Fund ind_proxy_label is near-collapsed (non-zero labels <1%). Review fund gating/rules.")

    dep_pos = diagnostics.get("deposit_train_positive_per_query", {})
    if dep_pos:
        if float(dep_pos.get("zero_positive_query_rate", 1.0)) > 0.2:
            warnings.append("High zero-positive-query rate in deposit train set (>20%). NDCG reliability may be weak.")

    summary["warnings"] = warnings

    args.out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report_lines = [
        "# 개선 추천시스템 리포트",
        "",
        f"- family mode: `{args.family}`",
        "",
        "## 적용한 개선",
        "- 수신상품/펀드 utility prior를 분리 반영",
        "- 사용자-상품 쌍 실현가능성(realizability) 피처 추가",
        "- 독립 proxy label(ind_proxy_label) 기반 LTR 재학습",
        "",
        "## 데이터",
        f"- train rows: {summary['data']['train_rows']:,}",
        f"- eval rows: {summary['data']['eval_rows']:,}",
        f"- train queries: {summary['data']['train_queries']:,}",
        f"- eval queries: {summary['data']['eval_queries']:,}",
        f"- feature count: {summary['data']['feature_count']}",
        "",
        "## Utility 파라미터",
        f"- fit_core_weights: {utility_params.fit_core_weights}",
        f"- dep_real_weights: {utility_params.dep_real_weights}",
        f"- fund_real_weights: {utility_params.fund_real_weights}",
        f"- hybrid_weights: {utility_params.hybrid_weights}",
        f"- dep_label_weights: {utility_params.dep_label_weights}",
        f"- fund_label_weights: {utility_params.fund_label_weights}",
        f"- dep_item_weights: {utility_params.dep_item_weights}",
        f"- fund_item_weights: {utility_params.fund_item_weights}",
        f"- fund_low_risk_penalty: {utility_params.fund_low_risk_penalty:.4f}",
        f"- fund_gate_risk_min: {utility_params.fund_gate_risk_min:.4f}",
        f"- fund_gate_family_min: {utility_params.fund_gate_family_min:.4f}",
        "",
        "## Utility 튜닝",
        f"- enabled: {bool(args.tune_trials > 0)}",
        f"- trials: {int(args.tune_trials)}",
        f"- tune_k: {int(args.tune_k)}",
        f"- tune_item_only: {bool(args.tune_item_only)}",
    ]
    if tuning_trials_df is not None and not tuning_trials_df.empty:
        report_lines.append(f"- best objective: {float(tuning_trials_df.iloc[0]['objective']):.4f}")
        report_lines.append(f"- best trial: {str(tuning_trials_df.iloc[0]['trial'])}")
    report_lines += [
        "",
        "## 식 정의 (현재 반영값)",
        f"- item_utility (deposit) = {summary['formulae']['item_utility_deposit']}",
        f"- item_utility (fund) = {summary['formulae']['item_utility_fund']}",
        f"- fit_core = {summary['formulae']['fit_core']}",
        f"- realizability (deposit) = {summary['formulae']['realizability_deposit']}",
        f"- realizability (fund) = {summary['formulae']['realizability_fund']}",
        f"- hybrid_utility_score = {summary['formulae']['hybrid_utility_score']}",
        f"- ind_proxy_label score (deposit) = {summary['formulae']['ind_proxy_label_deposit_score']}",
        f"- ind_proxy_label score (fund) = {summary['formulae']['ind_proxy_label_fund_score']}",
        "",
        "## 식 의미",
        "- fit_core: 사용자-상품 기본 적합도(위험/유동성/기간/복잡도/금액충족) 요약값",
        "- realizability: '이론상 좋음'이 아니라 실제 가입/달성 가능성을 반영한 보정값",
        "- item_utility_prior: 상품 자체의 사전 utility 인덱스(수신/펀드 분리 계산, 본 튜닝의 핵심 대상)",
        "- rate_factor: 금리 매력도를 약한 tie-breaker로 반영(과대영향 방지)",
        "- hybrid_utility_score: 적합도 + 상품사전점수 + 실현가능성 + 금리요인 결합 점수",
        "- ind_proxy_label: 학습용 약라벨(독립 규칙)로 쿼리 내 상대 순위를 0/1/2/3으로 변환",
        "",
        "## 튜닝 방법",
        "- baseline 가중치 주변을 Dirichlet 샘플링으로 랜덤 탐색",
        "- 탐색 단위: 가중치 세트 1개당 전체 eval query에서 NDCG 계산",
        "- 목적함수: 0.7*NDCG(hybrid, proxy_label) + 0.3*NDCG(hybrid, ind_proxy_label)",
        "- 최적 선택: 목적함수 최대 trial 채택",
        "",
        "## 성능 (NDCG)",
    ]
    for k in args.ks:
        report_lines.append(
            f"- @ {k} [proxy_label]: baseline={metrics_proxy_label[f'baseline_score_ndcg@{k}']:.4f}, hybrid_rule={metrics_proxy_label[f'hybrid_utility_score_ndcg@{k}']:.4f}, model={metrics_proxy_label[f'model_score_ndcg@{k}']:.4f}"
        )
    report_lines.append("")
    report_lines.append("## 학습 라벨 기준 (ind_proxy_label)")
    for k in args.ks:
        report_lines.append(
            f"- @ {k}: baseline={metrics_train_label[f'baseline_score_ndcg@{k}']:.4f}, hybrid_rule={metrics_train_label[f'hybrid_utility_score_ndcg@{k}']:.4f}, model={metrics_train_label[f'model_score_ndcg@{k}']:.4f}"
        )
    report_lines.append("")
    report_lines.append("## 내부 일치도 (hybrid_label)")
    for k in args.ks:
        report_lines.append(
            f"- @ {k}: baseline={metrics_hybrid_label[f'baseline_score_ndcg@{k}']:.4f}, hybrid_rule={metrics_hybrid_label[f'hybrid_utility_score_ndcg@{k}']:.4f}, model={metrics_hybrid_label[f'model_score_ndcg@{k}']:.4f}"
        )
    report_lines += [
        "",
        "## 라벨 진단",
    ]
    if "fund_eval_amount_feasibility_dist" in diagnostics:
        report_lines.append(f"- fund eval amount_feasibility dist: {diagnostics['fund_eval_amount_feasibility_dist']}")
    if "fund_eval_ind_proxy_label_dist" in diagnostics:
        report_lines.append(f"- fund eval ind_proxy_label dist: {diagnostics['fund_eval_ind_proxy_label_dist']}")
    if "fund_eval_positive_rate_label_ge2" in diagnostics:
        report_lines.append(f"- fund eval positive rate (label>=2): {diagnostics['fund_eval_positive_rate_label_ge2']:.4f}")
    if "deposit_train_positive_per_query" in diagnostics:
        report_lines.append(
            f"- deposit train positive-per-query stats: {diagnostics['deposit_train_positive_per_query']}"
        )

    if warnings:
        report_lines += [
            "",
            "## 경고",
        ]
        for w in warnings:
            report_lines.append(f"- {w}")

    report_lines += [
        "",
        "## Utility Feature Importance",
    ]
    if "utility_feature_importance" in summary:
        for r in summary["utility_feature_importance"][:8]:
            report_lines.append(f"- {r['feature']}: share={float(r['importance_share']):.4f}")
    report_lines += [
        "",
        "## Top-5 상품군 비중",
    ]
    for fam_name, ratio in summary["top5_family_mix"].items():
        report_lines.append(f"- {fam_name}: {ratio:.3f}")

    report_lines += [
        "",
        "## 산출물",
        f"- raw json: `{args.out_json}`",
        f"- figure: `{args.out_dir / '01_ndcg_comparison.png'}`",
        f"- figure: `{args.out_dir / '02_score_distribution.png'}`",
        f"- figure: `{args.out_dir / '03_feature_importance.png'}`",
        f"- figure: `{args.out_dir / '04_top5_family_mix.png'}`",
        f"- figure: `{args.out_dir / '05_utility_feature_importance.png'}`",
    ]
    if args.tune_trials > 0:
        report_lines.append(f"- tuning csv: `{args.tune_out_csv}`")

    (args.out_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"saved report: {args.out_dir / 'report.md'}")
    print(f"saved json: {args.out_json}")


if __name__ == "__main__":
    main()
