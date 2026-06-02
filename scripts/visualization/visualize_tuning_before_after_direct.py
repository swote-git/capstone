#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np


def set_korean_font() -> str:
    candidates = ["NanumGothic", "Noto Sans CJK KR", "Noto Sans KR", "AppleGothic", "Malgun Gothic"]
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = next((c for c in candidates if c in available), "DejaVu Sans")
    plt.rcParams["font.family"] = chosen
    plt.rcParams["axes.unicode_minus"] = False
    return chosen


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Direct before/after chart from tuning trials")
    p.add_argument("--deposit-csv", type=Path, default=Path("reports/raw/utility_tuning_trials_deposit.csv"))
    p.add_argument("--fund-csv", type=Path, default=Path("reports/raw/utility_tuning_trials_fund.csv"))
    p.add_argument("--out-png", type=Path, default=Path("reports/e2e/utility_tuning/direct_before_after_tuning_20260508.png"))
    p.add_argument("--out-csv", type=Path, default=Path("reports/e2e/utility_tuning/direct_before_after_tuning_20260508.csv"))
    return p.parse_args()


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def get_baseline_and_best(rows: List[Dict[str, str]]) -> Tuple[Dict[str, str], Dict[str, str]]:
    baseline = None
    best = None
    best_obj = -1e18
    for r in rows:
        if r.get("trial") == "baseline":
            baseline = r
        obj = float(r.get("objective", 0.0))
        if obj > best_obj:
            best_obj = obj
            best = r
    if baseline is None or best is None:
        raise ValueError("baseline/best row not found")
    return baseline, best


def pick_metrics(row: Dict[str, str]) -> Dict[str, float]:
    return {
        "objective": float(row.get("objective", 0.0)),
        "hybrid_vs_proxy_ndcg": float(row.get("hybrid_vs_proxy_ndcg", 0.0)),
        "hybrid_vs_ind_proxy_ndcg": float(row.get("hybrid_vs_ind_proxy_ndcg", 0.0)),
    }


def plot_family(ax: plt.Axes, family: str, before: Dict[str, float], after: Dict[str, float]) -> None:
    labels = ["Objective", "Hybrid vs Proxy", "Hybrid vs IndProxy"]
    b = [before["objective"], before["hybrid_vs_proxy_ndcg"], before["hybrid_vs_ind_proxy_ndcg"]]
    a = [after["objective"], after["hybrid_vs_proxy_ndcg"], after["hybrid_vs_ind_proxy_ndcg"]]
    x = np.arange(len(labels))
    w = 0.34

    bars1 = ax.bar(x - w / 2, b, width=w, color="#9E9E9E", label="Before (baseline trial)")
    bars2 = ax.bar(x + w / 2, a, width=w, color="#2E86AB", label="After (best trial)")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_title(f"{family} (동일 tuning run 내 전후 비교)", fontsize=15)
    ax.grid(axis="y", linestyle="--", alpha=0.25)

    for p in list(bars1) + list(bars2):
        h = p.get_height()
        ax.text(p.get_x() + p.get_width() / 2, h + 0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=9)

    for i, (bv, av) in enumerate(zip(b, a)):
        d = av - bv
        ax.text(i, max(bv, av) + 0.04, f"Δ {d:+.3f}", ha="center", va="bottom", fontsize=10, color=("#1E8449" if d >= 0 else "#B03A2E"))


def main() -> None:
    args = parse_args()
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    font = set_korean_font()

    dep_rows = read_rows(args.deposit_csv)
    fund_rows = read_rows(args.fund_csv)
    dep_before_row, dep_after_row = get_baseline_and_best(dep_rows)
    fund_before_row, fund_after_row = get_baseline_and_best(fund_rows)

    dep_before = pick_metrics(dep_before_row)
    dep_after = pick_metrics(dep_after_row)
    fund_before = pick_metrics(fund_before_row)
    fund_after = pick_metrics(fund_after_row)

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    plot_family(axes[0], "Deposit", dep_before, dep_after)
    plot_family(axes[1], "Fund", fund_before, fund_after)
    axes[0].set_ylabel("score")
    axes[1].legend(loc="lower right", fontsize=10)

    fig.suptitle("Utility 튜닝 전/후 직접 비교 (동일 run: baseline trial vs best trial)", fontsize=20, y=0.965)

    caption = (
        "지표 설명\n"
        "- Objective: 0.7×NDCG(Hybrid vs Proxy) + 0.3×NDCG(Hybrid vs IndProxy)\n"
        "- Hybrid vs Proxy: hybrid_utility_score 순위를 proxy_label 기준으로 평가한 NDCG\n"
        "- Hybrid vs IndProxy: hybrid_utility_score 순위를 ind_proxy_label 기준으로 평가한 NDCG\n"
        "- Proxy: 기존 약라벨(rec._build_labels), IndProxy: 독립 규칙 약라벨"
    )
    fig.text(
        0.5,
        0.03,
        caption,
        ha="center",
        va="bottom",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#F8F9FA", "edgecolor": "#B0B0B0"},
    )
    fig.text(0.01, 0.01, f"source: utility_tuning_trials_deposit/fund.csv | font={font}", fontsize=9, color="#666")

    fig.subplots_adjust(top=0.88, bottom=0.30, wspace=0.18)
    fig.savefig(args.out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    lines = [
        "family,phase,objective,hybrid_vs_proxy_ndcg,hybrid_vs_ind_proxy_ndcg,trial",
        f"deposit,before,{dep_before['objective']:.6f},{dep_before['hybrid_vs_proxy_ndcg']:.6f},{dep_before['hybrid_vs_ind_proxy_ndcg']:.6f},{dep_before_row.get('trial','')}",
        f"deposit,after,{dep_after['objective']:.6f},{dep_after['hybrid_vs_proxy_ndcg']:.6f},{dep_after['hybrid_vs_ind_proxy_ndcg']:.6f},{dep_after_row.get('trial','')}",
        f"fund,before,{fund_before['objective']:.6f},{fund_before['hybrid_vs_proxy_ndcg']:.6f},{fund_before['hybrid_vs_ind_proxy_ndcg']:.6f},{fund_before_row.get('trial','')}",
        f"fund,after,{fund_after['objective']:.6f},{fund_after['hybrid_vs_proxy_ndcg']:.6f},{fund_after['hybrid_vs_ind_proxy_ndcg']:.6f},{fund_after_row.get('trial','')}",
    ]
    args.out_csv.write_text("\n".join(lines), encoding="utf-8")

    print(f"saved: {args.out_png}")
    print(f"saved: {args.out_csv}")


if __name__ == "__main__":
    main()
