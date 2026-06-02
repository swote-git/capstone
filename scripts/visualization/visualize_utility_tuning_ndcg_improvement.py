#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
    p = argparse.ArgumentParser(description="Visualize utility-tuning NDCG improvements (single chart)")
    p.add_argument(
        "--deposit-json",
        type=Path,
        default=Path("reports/raw/utility_tuning_deposit.json"),
    )
    p.add_argument(
        "--fund-json",
        type=Path,
        default=Path("reports/raw/utility_tuning_fund.json"),
    )
    p.add_argument(
        "--out-png",
        type=Path,
        default=Path("reports/e2e/utility_tuning/ndcg_improvement_summary_20260508.png"),
    )
    return p.parse_args()


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_delta_rows(obj: Dict, family_name: str) -> List[Tuple[str, float, float]]:
    rows: List[Tuple[str, float, float]] = []
    metric_sections = [
        ("metrics_proxy_label", "Proxy"),
        ("metrics_train_label", "IndProxy"),
        ("metrics_hybrid_label", "Hybrid"),
    ]
    for sec_key, sec_name in metric_sections:
        sec = obj.get(sec_key, {})
        for k in (5, 10):
            baseline = float(sec.get(f"baseline_score_ndcg@{k}", 0.0))
            hybrid = float(sec.get(f"hybrid_utility_score_ndcg@{k}", 0.0))
            model = float(sec.get(f"model_score_ndcg@{k}", 0.0))
            rows.append((f"{family_name}-{sec_name}@{k}", hybrid - baseline, model - baseline))
    return rows


def main() -> None:
    args = parse_args()
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    font_name = set_korean_font()

    dep = load_json(args.deposit_json)
    fund = load_json(args.fund_json)

    rows = build_delta_rows(dep, "Deposit") + build_delta_rows(fund, "Fund")
    labels = [r[0] for r in rows]
    d_hybrid = np.array([r[1] for r in rows], dtype=float)
    d_model = np.array([r[2] for r in rows], dtype=float)
    x = np.arange(len(labels))
    w = 0.38

    fig, ax = plt.subplots(figsize=(16, 9))
    b1 = ax.bar(x - w / 2, d_hybrid, width=w, label="Hybrid - Baseline", color="#5DA5DA")
    b2 = ax.bar(x + w / 2, d_model, width=w, label="Model - Baseline", color="#F17CB0")
    ax.axhline(0.0, color="#444444", linewidth=1)

    ax.set_title("Utility Tuning NDCG 개선폭 요약 (ΔNDCG, baseline 대비)", fontsize=20, pad=14)
    ax.set_ylabel("ΔNDCG", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(loc="best", fontsize=11)

    for bars in (b1, b2):
        for p in bars:
            h = p.get_height()
            ax.text(
                p.get_x() + p.get_width() / 2,
                h + (0.01 if h >= 0 else -0.03),
                f"{h:+.3f}",
                ha="center",
                va="bottom" if h >= 0 else "top",
                fontsize=9,
            )

    fig.text(
        0.01,
        0.01,
        f"source: {args.deposit_json.name}, {args.fund_json.name} | font={font_name}",
        fontsize=9,
        color="#666666",
    )
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=300)
    plt.close(fig)
    print(f"saved: {args.out_png}")


if __name__ == "__main__":
    main()
