#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def set_korean_font() -> str:
    candidates = [
        "NanumGothic",
        "Noto Sans CJK KR",
        "Noto Sans KR",
        "Malgun Gothic",
        "AppleGothic",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = next((c for c in candidates if c in available), "DejaVu Sans")
    plt.rcParams["font.family"] = chosen
    plt.rcParams["axes.unicode_minus"] = False
    return chosen


def load_eval(path: Path) -> Dict[str, float]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    r = obj["ranking_evaluation"]
    v = obj.get("verifier_evaluation", {})
    u = obj.get("understanding_evaluation_summary", {})
    return {
        "baseline@5": float(r["baseline_ndcg@5"]),
        "model@5": float(r["model_ndcg@5"]),
        "baseline@10": float(r["baseline_ndcg@10"]),
        "model@10": float(r["model_ndcg@10"]),
        "pass_rate": float(v.get("pass_rate", 0.0)),
        "ug": float(u.get("understanding_gain", 0.0)),
        "mr": float(u.get("misinterpretation_rate", 0.0)),
    }


def add_value_labels(ax: plt.Axes) -> None:
    for p in ax.patches:
        h = p.get_height()
        if np.isnan(h):
            continue
        ax.text(
            p.get_x() + p.get_width() / 2.0,
            h + 0.01,
            f"{h:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )


def plot_deposit_before_after(dep_prev: Dict[str, float], dep_net: Dict[str, float], out: Path) -> None:
    labels = ["NDCG@5", "NDCG@10"]
    prev = [dep_prev["model@5"], dep_prev["model@10"]]
    net = [dep_net["model@5"], dep_net["model@10"]]
    x = np.arange(len(labels))
    w = 0.34

    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    ax.bar(x - w / 2, prev, width=w, label="deposit_real_llm", color="#8FB9A8")
    ax.bar(x + w / 2, net, width=w, label="deposit_real_llm_network", color="#2E8B57")
    ax.set_ylim(0, 1.05)
    ax.set_title("수신상품 NDCG 비교 (동일 사용자, 실행 버전별)", fontsize=20, pad=14)
    ax.set_ylabel("NDCG", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=13)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(fontsize=12, loc="lower right")
    add_value_labels(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def plot_family_compare(dep: Dict[str, float], fund: Dict[str, float], out: Path) -> None:
    labels = ["Baseline@5", "Model@5", "Baseline@10", "Model@10"]
    dep_vals = [dep["baseline@5"], dep["model@5"], dep["baseline@10"], dep["model@10"]]
    fund_vals = [fund["baseline@5"], fund["model@5"], fund["baseline@10"], fund["model@10"]]
    x = np.arange(len(labels))
    w = 0.34

    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    ax.bar(x - w / 2, dep_vals, width=w, label="Deposit", color="#4C78A8")
    ax.bar(x + w / 2, fund_vals, width=w, label="Fund", color="#F58518")
    ax.set_ylim(0, 1.05)
    ax.set_title("상품군별 NDCG 비교 (SYN_13673)", fontsize=20, pad=14)
    ax.set_ylabel("NDCG", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(fontsize=12, loc="lower right")
    add_value_labels(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def plot_model_trend(dep: Dict[str, float], fund: Dict[str, float], out: Path) -> None:
    ks = [5, 10]
    dep_vals = [dep["model@5"], dep["model@10"]]
    fund_vals = [fund["model@5"], fund["model@10"]]

    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    ax.plot(ks, dep_vals, marker="o", linewidth=3, markersize=9, color="#4C78A8", label="Deposit Model")
    ax.plot(ks, fund_vals, marker="o", linewidth=3, markersize=9, color="#F58518", label="Fund Model")
    ax.set_ylim(0, 1.05)
    ax.set_xticks([5, 10])
    ax.set_xlabel("k", fontsize=14)
    ax.set_ylabel("NDCG", fontsize=14)
    ax.set_title("Model NDCG 추이 (@5 vs @10)", fontsize=20, pad=14)
    ax.grid(linestyle="--", alpha=0.3)
    ax.legend(fontsize=12, loc="lower left")
    for x, y in zip(ks, dep_vals):
        ax.text(x, y + 0.02, f"{y:.3f}", color="#4C78A8", ha="center", fontsize=11)
    for x, y in zip(ks, fund_vals):
        ax.text(x, y + 0.02, f"{y:.3f}", color="#F58518", ha="center", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def plot_eval_aux(dep: Dict[str, float], fund: Dict[str, float], out: Path) -> None:
    metrics = ["Pass Rate", "UG", "MR"]
    dep_vals = [dep["pass_rate"], dep["ug"], dep["mr"]]
    fund_vals = [fund["pass_rate"], fund["ug"], fund["mr"]]
    x = np.arange(len(metrics))
    w = 0.34

    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    ax.bar(x - w / 2, dep_vals, width=w, label="Deposit", color="#4C78A8")
    ax.bar(x + w / 2, fund_vals, width=w, label="Fund", color="#F58518")
    ax.set_ylim(0, 1.05)
    ax.set_title("설명 평가 지표 비교 (Verifier/Effect)", fontsize=20, pad=14)
    ax.set_ylabel("Score", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(fontsize=12, loc="upper right")
    add_value_labels(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PPT-friendly chart pack for SYN_13673 E2E results")
    p.add_argument(
        "--deposit-prev-json",
        type=Path,
        default=Path("reports/e2e/syn_13673_presentation_bundle/deposit_real_llm/20260508_025455/05_evaluation_result.json"),
    )
    p.add_argument(
        "--deposit-network-json",
        type=Path,
        default=Path("reports/e2e/syn_13673_presentation_bundle/deposit_real_llm_network/20260508_030043/05_evaluation_result.json"),
    )
    p.add_argument(
        "--fund-network-json",
        type=Path,
        default=Path("reports/e2e/syn_13673_presentation_bundle/fund_real_llm_network/20260508_040159/05_evaluation_result.json"),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports/e2e/syn_13673_presentation_bundle/ppt_charts_20260508"),
    )
    return p.parse_args()


def main() -> None:
    matplotlib.use("Agg")
    sns.set_theme(style="whitegrid")
    font = set_korean_font()

    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dep_prev = load_eval(args.deposit_prev_json)
    dep_net = load_eval(args.deposit_network_json)
    fund_net = load_eval(args.fund_network_json)

    plot_deposit_before_after(dep_prev, dep_net, args.out_dir / "01_deposit_model_ndcg_before_after.png")
    plot_family_compare(dep_net, fund_net, args.out_dir / "02_family_baseline_model_ndcg.png")
    plot_model_trend(dep_net, fund_net, args.out_dir / "03_family_model_ndcg_trend.png")
    plot_eval_aux(dep_net, fund_net, args.out_dir / "04_family_eval_aux_metrics.png")

    readme = [
        "# SYN_13673 PPT Chart Pack",
        "",
        f"- font: `{font}`",
        "- target: PPT insertion (16:9, 300dpi PNG)",
        "",
        "## Files",
        "- 01_deposit_model_ndcg_before_after.png",
        "- 02_family_baseline_model_ndcg.png",
        "- 03_family_model_ndcg_trend.png",
        "- 04_family_eval_aux_metrics.png",
    ]
    (args.out_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print(f"saved: {args.out_dir}")


if __name__ == "__main__":
    main()
