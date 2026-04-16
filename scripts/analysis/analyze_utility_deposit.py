#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from common.config import RecommenderConfig
from recommender.engine import ThinFilerRecommender


def set_korean_font() -> str:
    candidates = ["NanumGothic", "Noto Sans CJK KR", "Noto Sans KR", "AppleGothic", "Malgun Gothic"]
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = next((c for c in candidates if c in available), "DejaVu Sans")
    plt.rcParams["font.family"] = chosen
    plt.rcParams["axes.unicode_minus"] = False
    return chosen


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Customer-product utility analysis for deposit products")
    p.add_argument("--sample-users", type=int, default=300)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--deposit-csv", type=Path, default=Path("data/processed/product12_deposit_catalog.csv"))
    p.add_argument("--raw-deposit-csv", type=Path, default=Path("data/12.금융상품정보/은행수신상품.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("reports/utility"))
    p.add_argument("--out-pair-csv", type=Path, default=Path("data/processed/deposit_utility_pairs_sample.csv"))
    return p.parse_args()


def cond_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("우대금리조건_") and c.endswith("_여부")]
    return [c for c in cols if "기타" not in c]


def short_name(col: str) -> str:
    return col.replace("우대금리조건_", "").replace("_여부", "")


def clean_binary(s: pd.Series) -> pd.Series:
    out = s.replace({"Y": 1, "N": 0, "y": 1, "n": 0, True: 1, False: 0})
    out = pd.to_numeric(out, errors="coerce").fillna(0)
    return (out > 0).astype(float)


def build_condition_product_table(raw_deposit: pd.DataFrame) -> pd.DataFrame:
    yn = cond_cols(raw_deposit)
    frames = [raw_deposit[["상품코드"]].rename(columns={"상품코드": "product_id"}).astype(str)]

    for c in yn:
        bonus_c = c.replace("_여부", "_우대금리")
        cname = short_name(c)
        active = clean_binary(raw_deposit[c]).rename(f"cond_active::{cname}")
        bonus = pd.to_numeric(raw_deposit.get(bonus_c, 0), errors="coerce").fillna(0).rename(f"cond_bonus::{cname}")
        frames.extend([active, bonus])

    tmp = pd.concat(frames, axis=1)
    agg_dict = {col: "max" for col in tmp.columns if col != "product_id"}
    pt = tmp.groupby("product_id", as_index=False).agg(agg_dict)
    return pt


def user_condition_propensity(user: pd.Series, cond_name: str) -> float:
    digital = float(np.clip(user.get("digital_behavior_freq", 0.0), 0, 1))
    card = float(np.clip(0.5 * user.get("card_usage_stability", 0.0) + 0.5 * np.clip(user.get("spending_vs_balance_ratio", 0) / 2, 0, 1), 0, 1))
    liquidity_need = float(np.clip(user.get("liquidity_need", 1.0) / 3.0, 0, 1))
    complexity = float(np.clip(user.get("complexity_tolerance", 0.0) / 2.0, 0, 1))

    n = cond_name
    if any(k in n for k in ["모바일", "비대면", "인터넷", "오픈뱅킹", "마이데이터"]):
        return float(np.clip(0.2 + 0.8 * digital, 0, 1))
    if any(k in n for k in ["카드", "신용카드", "체크카드", "가맹점"]):
        return float(np.clip(0.2 + 0.8 * card, 0, 1))
    if any(k in n for k in ["급여이체", "공과금이체", "예금자동이체", "보험료자동이체", "연금자동이체"]):
        return float(np.clip(0.25 + 0.5 * card + 0.25 * liquidity_need, 0, 1))
    if any(k in n for k in ["첫거래", "상품미보유"]):
        return 0.3
    if any(k in n for k in ["고객연령", "고객특성", "어린이", "학생", "장병", "농어민"]):
        return float(np.clip(0.2 + 0.6 * (1 - complexity), 0, 1))
    return 0.35


def normalize01(x: np.ndarray) -> np.ndarray:
    mn = float(np.nanmin(x))
    mx = float(np.nanmax(x))
    if mx - mn < 1e-12:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def main() -> None:
    args = parse_args()
    font_name = set_korean_font()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.out_pair_csv.parent.mkdir(parents=True, exist_ok=True)

    # user snapshot
    rec = ThinFilerRecommender(RecommenderConfig())
    users = rec.build_user_snapshots(sample_users=args.sample_users)
    snapshot_rows = len(users)

    # deposit products + condition table
    dep = pd.read_csv(args.deposit_csv)
    raw_dep = pd.read_csv(args.raw_deposit_csv, low_memory=False)
    cond_pt = build_condition_product_table(raw_dep)

    prod = dep.merge(cond_pt, on="product_id", how="left")
    prod = prod.fillna(0)
    prod["horizon_code"] = (
        prod["horizon"]
        .map({"short": 0, "mid": 1, "long": 2})
        .fillna(1)
        .astype("int64")
    )
    for c in ["risk_level", "liquidity_level", "complexity", "min_amount_bin", "max_rate", "base_rate"]:
        if c in prod.columns:
            prod[c] = pd.to_numeric(prod[c], errors="coerce").fillna(0)

    cond_names = [c.split("::", 1)[1] for c in prod.columns if c.startswith("cond_active::")]
    active_cols = [f"cond_active::{n}" for n in cond_names]
    bonus_cols = [f"cond_bonus::{n}" for n in cond_names]

    active_mat = prod[active_cols].to_numpy(dtype=float)
    bonus_mat = prod[bonus_cols].to_numpy(dtype=float)
    weight_mat = active_mat * bonus_mat

    base_rate = pd.to_numeric(prod["base_rate"], errors="coerce").fillna(0).to_numpy(dtype=float)
    max_rate = pd.to_numeric(prod["max_rate"], errors="coerce").fillna(0).to_numpy(dtype=float)
    min_amount_bin = pd.to_numeric(prod["min_amount_bin"], errors="coerce").fillna(0).to_numpy(dtype=float)
    cond_count = active_mat.sum(axis=1)

    all_rows = []

    for _, u in users.iterrows():
        p = np.array([user_condition_propensity(u, n) for n in cond_names], dtype=float)

        achievable_bonus = (weight_mat * p).sum(axis=1)
        realized_rate = base_rate + achievable_bonus
        U1 = normalize01(realized_rate)

        den = weight_mat.sum(axis=1)
        U2 = np.where(den > 0, (weight_mat * p).sum(axis=1) / (den + 1e-12), 1.0)

        # 접근성 hard filter (가입금액)
        user_amount_bin = float(u.get("amount_bin", 0))
        U3 = (user_amount_bin >= min_amount_bin).astype(float)

        # 적합성 (risk/liquidity/horizon/complexity match)
        pair = rec._add_pair_features(pd.DataFrame([u]), prod)
        U4 = (
            0.35 * pair["risk_match"].to_numpy()
            + 0.35 * pair["liquidity_match"].to_numpy()
            + 0.15 * pair["horizon_match"].to_numpy()
            + 0.15 * pair["complexity_match"].to_numpy()
        )

        # 부가혜택
        U5 = 0.5 * normalize01(cond_count) + 0.5 * normalize01(max_rate)

        utility = 0.40 * U1 + 0.25 * U2 + 0.20 * U3 + 0.10 * U4 + 0.05 * U5

        label = ((utility >= 0.65) & (U1 >= 0.60) & (U3 > 0)).astype(int)
        hard_negative = (U3 == 0).astype(int)

        user_rows = pd.DataFrame(
            {
                "user_id": str(u[rec.config.user_key_11]),
                "as_of_date": str(u.get("as_of_date", "")),
                "product_id": prod["product_id"].astype(str),
                "U1_realized_rate": U1,
                "U2_achievability": U2,
                "U3_accessibility": U3,
                "U4_fit": U4,
                "U5_extra": U5,
                "utility": utility,
                "label_positive": label,
                "hard_negative": hard_negative,
                "realized_rate": realized_rate,
                "achievable_bonus": achievable_bonus,
                "base_rate": base_rate,
            }
        )

        top_user = user_rows.sort_values("utility", ascending=False).head(args.top_k)
        all_rows.append(top_user)

    out = pd.concat(all_rows, ignore_index=True)
    out.to_csv(args.out_pair_csv, index=False, encoding="utf-8-sig")

    # snapshot-level utility KPI
    out["snapshot_key"] = out["user_id"].astype(str) + "::" + out["as_of_date"].astype(str)
    snap_kpi = (
        out.groupby("snapshot_key", as_index=False)
        .agg(
            user_id=("user_id", "first"),
            as_of_date=("as_of_date", "first"),
            utility_at_k=("utility", "mean"),
            top1_utility=("utility", "max"),
            positive_rate_at_k=("label_positive", "mean"),
            positive_hit_at_k=("label_positive", lambda x: float((x > 0).any())),
            expected_realized_rate=("realized_rate", "mean"),
            expected_bonus=("achievable_bonus", "mean"),
        )
    )
    user_kpi = (
        snap_kpi.groupby("user_id", as_index=False)
        .agg(
            mean_utility_at_k=("utility_at_k", "mean"),
            mean_top1_utility=("top1_utility", "mean"),
            mean_positive_rate_at_k=("positive_rate_at_k", "mean"),
            hit_rate_at_k=("positive_hit_at_k", "mean"),
            expected_realized_rate=("expected_realized_rate", "mean"),
            expected_bonus=("expected_bonus", "mean"),
            snapshot_count=("as_of_date", "count"),
        )
    )
    user_kpi_path = args.out_dir / "user_utility_summary.csv"
    user_kpi.to_csv(user_kpi_path, index=False, encoding="utf-8-sig")

    # Summary
    summary = {
        "sample_users_unique": int(users[rec.config.user_key_11].nunique()),
        "sample_snapshots": int(snapshot_rows),
        "topk_per_user": int(args.top_k),
        "rows": int(len(out)),
        "mean_utility": float(out["utility"].mean()),
        "median_utility": float(out["utility"].median()),
        "positive_ratio": float(out["label_positive"].mean()),
        "hard_negative_ratio": float(out["hard_negative"].mean()),
        "mean_U1": float(out["U1_realized_rate"].mean()),
        "mean_U2": float(out["U2_achievability"].mean()),
        "mean_U3": float(out["U3_accessibility"].mean()),
        "mean_U4": float(out["U4_fit"].mean()),
        "mean_U5": float(out["U5_extra"].mean()),
        "mean_utility_at_k": float(snap_kpi["utility_at_k"].mean()),
        "positive_hit_at_k": float(snap_kpi["positive_hit_at_k"].mean()),
        "mean_positive_rate_at_k": float(snap_kpi["positive_rate_at_k"].mean()),
        "expected_realized_rate": float(snap_kpi["expected_realized_rate"].mean()),
        "expected_bonus": float(snap_kpi["expected_bonus"].mean()),
    }

    # figures
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(out["utility"], bins=30, ax=ax, color="#2E86AB")
    ax.set_title("Top-k 추천 샘플 Utility 분포")
    ax.set_xlabel("utility")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(args.out_dir / "utility_hist.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(data=out, x="U1_realized_rate", y="utility", hue="label_positive", palette="Set2", alpha=0.7, ax=ax)
    ax.set_title("U1(실현금리) vs Utility")
    fig.tight_layout()
    fig.savefig(args.out_dir / "u1_vs_utility_scatter.png", dpi=160)
    plt.close(fig)

    comp = out[["U1_realized_rate", "U2_achievability", "U3_accessibility", "U4_fit", "U5_extra"]].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=comp.index, y=comp.values, ax=ax, color="#74c476")
    ax.set_title("Utility 컴포넌트 평균 (Top-k 샘플)")
    ax.set_xlabel("component")
    ax.set_ylabel("mean")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(args.out_dir / "utility_component_mean.png", dpi=160)
    plt.close(fig)

    # markdown report
    lines = []
    lines.append("# 수신상품 고객별 Utility 분석 보고서")
    lines.append("")
    lines.append(f"- 적용 폰트: {font_name}")
    lines.append(f"- 사용자 샘플 수(고유): {summary['sample_users_unique']}")
    lines.append(f"- 사용자 스냅샷 행 수: {summary['sample_snapshots']}")
    lines.append(f"- 사용자별 top-k: {summary['topk_per_user']}")
    lines.append(f"- 총 샘플 행: {summary['rows']}")
    lines.append("")
    lines.append("## Utility 정의")
    lines.append("- Utility = 0.40*U1 + 0.25*U2 + 0.20*U3 + 0.10*U4 + 0.05*U5")
    lines.append("- Positive label: Utility >= 0.65 AND U1 >= 0.60 AND U3 > 0")
    lines.append("- Hard negative: U3 == 0")
    lines.append("")
    lines.append("## 요약 지표")
    lines.append(f"- mean_utility: {summary['mean_utility']:.4f}")
    lines.append(f"- median_utility: {summary['median_utility']:.4f}")
    lines.append(f"- positive_ratio: {summary['positive_ratio']:.4f}")
    lines.append(f"- hard_negative_ratio: {summary['hard_negative_ratio']:.4f}")
    lines.append(f"- mean_U1(realized_rate): {summary['mean_U1']:.4f}")
    lines.append(f"- mean_U2(achievability): {summary['mean_U2']:.4f}")
    lines.append(f"- mean_U3(accessibility): {summary['mean_U3']:.4f}")
    lines.append(f"- mean_U4(fit): {summary['mean_U4']:.4f}")
    lines.append(f"- mean_U5(extra): {summary['mean_U5']:.4f}")
    lines.append("")
    lines.append("## 고객 유용성 KPI (@K)")
    lines.append(f"- mean_utility_at_k: {summary['mean_utility_at_k']:.4f}")
    lines.append(f"- positive_hit_at_k: {summary['positive_hit_at_k']:.4f}")
    lines.append(f"- mean_positive_rate_at_k: {summary['mean_positive_rate_at_k']:.4f}")
    lines.append(f"- expected_realized_rate: {summary['expected_realized_rate']:.4f}")
    lines.append(f"- expected_bonus: {summary['expected_bonus']:.4f}")
    lines.append("")
    lines.append("## 진단 노트")
    if summary["mean_U3"] >= 0.999:
        lines.append("- U3(accessibility)가 거의 1.0으로 고정: 가입금액/대상 접근성 분해능이 낮을 수 있습니다.")
    if summary["mean_U2"] >= 0.95:
        lines.append("- U2(달성가능성)가 매우 높음: 조건 매핑 규칙이 낙관적으로 설정되었을 수 있습니다.")
    if summary["positive_hit_at_k"] >= 0.95:
        lines.append("- Positive hit@k가 높음: threshold가 완만하거나 후보군 난이도가 낮을 가능성이 있습니다.")
    lines.append("")
    lines.append("## 산출물")
    lines.append(f"- pair sample csv: `{args.out_pair_csv}`")
    lines.append(f"- user utility summary csv: `{user_kpi_path}`")
    lines.append(f"- figure: `{args.out_dir / 'utility_hist.png'}`")
    lines.append(f"- figure: `{args.out_dir / 'u1_vs_utility_scatter.png'}`")
    lines.append(f"- figure: `{args.out_dir / 'utility_component_mean.png'}`")

    (args.out_dir / "utility_report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"saved pair csv: {args.out_pair_csv}")
    print(f"saved report: {args.out_dir / 'utility_report.md'}")
    print(f"saved figures: {args.out_dir}")


if __name__ == "__main__":
    main()
