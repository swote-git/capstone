#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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
    return parse_args_with_config(p, section="improve_recommender_with_utility")


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


def add_hybrid_features(pair: pd.DataFrame) -> pd.DataFrame:
    pair = pair.copy()
    # fitness from pair matches
    fit_core = (
        0.30 * pair["risk_match"]
        + 0.25 * pair["liquidity_match"]
        + 0.20 * pair["horizon_match"]
        + 0.15 * pair["complexity_match"]
        + 0.10 * pair["amount_feasibility"]
    )

    # realizability: deposit/fund rules differ
    dep_real = clip01(
        0.45 * pair["amount_feasibility"]
        + 0.25 * pair["digital_match"]
        + 0.20 * (1.0 - pair["complexity"] / 2.0)
        + 0.10 * pair["liquidity_match"]
    )
    fund_real = clip01(
        0.40 * pair["risk_match"]
        + 0.30 * pair["family_match"]
        + 0.20 * pair["horizon_match"]
        + 0.10 * pair["amount_feasibility"]
    )
    pair["realizability"] = np.where(pair["product_family"].eq("deposit"), dep_real, fund_real)

    pair["item_utility_prior"] = clip01(pair.get("item_utility_prior", pd.Series(0.5, index=pair.index)))

    # mild rate factor to keep financial attractiveness
    maxr = pd.to_numeric(pair.get("max_rate", 0), errors="coerce").fillna(0)
    denom = float(np.nanpercentile(np.abs(maxr), 95) + 1e-6)
    pair["rate_factor"] = clip01(maxr / max(denom, 1e-6))

    pair["hybrid_utility_score"] = clip01(
        0.40 * fit_core
        + 0.35 * pair["item_utility_prior"]
        + 0.20 * pair["realizability"]
        + 0.05 * pair["rate_factor"]
    )

    return pair


def build_labels_from_hybrid(pair: pd.DataFrame) -> pd.Series:
    u = pair["hybrid_utility_score"]
    if len(u) < 6:
        return pd.Series(np.where(u >= u.median(), 2, 1), index=pair.index, dtype="int64")
    q80, q55, q30 = u.quantile([0.80, 0.55, 0.30]).tolist()
    y = np.select([u >= q80, u >= q55, u >= q30], [3, 2, 1], default=0)
    return pd.Series(y, index=pair.index, dtype="int64")


def build_proxy_label_independent(pair: pd.DataFrame) -> pd.Series:
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

    dep_score = (
        0.35 * item
        + 0.20 * rate_norm
        + 0.20 * feas
        + 0.15 * complexity
        + 0.10 * liquidity
    )
    fund_score = (
        0.35 * item
        + 0.25 * risk_adj_norm
        + 0.15 * family
        + 0.15 * horizon
        + 0.10 * digital
        - 0.10 * low_risk_penalty
    )
    is_deposit = pair["product_family"].eq("deposit")
    score = np.where(is_deposit, dep_score, fund_score)
    score = clip01(pd.Series(score, index=pair.index))

    # Family-specific gating:
    # - deposit: keep strict amount feasibility gate
    # - fund: do not collapse by amount_feasibility (often non-informative for funds)
    #         use risk compatibility + family consistency as realizability gate.
    fund_gate = ((risk_match >= 0.35) & (family >= 0.5)).astype(float)
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


def candidate_pair_for_user(rec: ThinFilerRecommender, user_row: pd.Series, priors: pd.DataFrame, candidate_max: int) -> pd.DataFrame:
    cands = rec.generate_candidates(user_row, max_candidates=candidate_max)
    pair = rec._add_pair_features(pd.DataFrame([user_row]), cands)
    pair["product_id"] = pair["product_id"].astype(str)
    pair = pair.merge(priors[["product_id", "item_utility_prior"]], on="product_id", how="left")
    pair = add_hybrid_features(pair)
    pair["hybrid_label"] = build_labels_from_hybrid(pair)
    pair["proxy_label"] = rec._build_labels(pair)
    pair["ind_proxy_label"] = build_proxy_label_independent(pair)
    pair["label"] = pair["ind_proxy_label"].astype("int64")
    return pair


def build_dataset(rec: ThinFilerRecommender, snapshots: pd.DataFrame, priors: pd.DataFrame, candidate_max: int, max_users: int) -> Tuple[pd.DataFrame, List[int]]:
    users = snapshots[rec.config.user_key_11].drop_duplicates()
    if len(users) > max_users:
        keep = users.sample(n=max_users, random_state=42)
        snapshots = snapshots[snapshots[rec.config.user_key_11].isin(keep)].copy()

    groups: List[int] = []
    rows: List[pd.DataFrame] = []
    for _, user_row in snapshots.iterrows():
        pair = candidate_pair_for_user(rec, user_row, priors, candidate_max)
        rows.append(pair)
        groups.append(len(pair))

    data = pd.concat(rows, ignore_index=True)
    return data, groups


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


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    font_name = set_korean_font()

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

    train_data, train_group = build_dataset(rec, train_snap, priors, args.candidate_max, args.max_train_users)
    eval_data, _ = build_dataset(rec, eval_snap, priors, args.candidate_max, args.max_eval_users)

    # query ids for grouped evaluation
    train_data = train_data.reset_index(drop=True)
    eval_data = eval_data.reset_index(drop=True)
    train_data["query_id"] = train_data[cfg.user_key_11].astype(str) + "::" + train_data["as_of_date"].astype(str)
    eval_data["query_id"] = eval_data[cfg.user_key_11].astype(str) + "::" + eval_data["as_of_date"].astype(str)

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
        "notes": [
            "Applied split utility priors: deposit_utility for deposits, fund_utility for funds.",
            "Added pair-level realizability to prevent over-rewarding hard-to-achieve products.",
            "Used hybrid utility weak labels for ranking supervision.",
            "Primary metric for model fit is ind_proxy_label on held-out users; proxy/hybrid metrics are auxiliary and may be circular.",
        ],
    }

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
    ]

    (args.out_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"saved report: {args.out_dir / 'report.md'}")
    print(f"saved json: {args.out_json}")


if __name__ == "__main__":
    main()
