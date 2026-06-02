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
    p = argparse.ArgumentParser(description="Compare old(v3) vs tuned NDCG")
    p.add_argument("--old-deposit-json", type=Path, default=Path("reports/raw/e2e_improved_recommender_deposit_v3.json"))
    p.add_argument("--old-fund-json", type=Path, default=Path("reports/raw/e2e_improved_recommender_fund_v3.json"))
    p.add_argument("--tuned-deposit-json", type=Path, default=Path("reports/raw/utility_tuning_deposit.json"))
    p.add_argument("--tuned-fund-json", type=Path, default=Path("reports/raw/utility_tuning_fund.json"))
    p.add_argument("--out-png", type=Path, default=Path("reports/e2e/utility_tuning/old_vs_tuned_ndcg_20260508.png"))
    p.add_argument("--out-csv", type=Path, default=Path("reports/e2e/utility_tuning/old_vs_tuned_ndcg_20260508.csv"))
    return p.parse_args()


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_rows(old_obj: Dict, tuned_obj: Dict, family: str) -> List[Tuple[str, float, float, float]]:
    rows: List[Tuple[str, float, float, float]] = []
    sections = [("metrics_proxy_label", "Proxy"), ("metrics_train_label", "IndProxy")]
    for key, label in sections:
        old_sec = old_obj.get(key, {})
        new_sec = tuned_obj.get(key, {})
        for k in (5, 10):
            old_m = float(old_sec.get(f"model_score_ndcg@{k}", 0.0))
            new_m = float(new_sec.get(f"model_score_ndcg@{k}", 0.0))
            rows.append((f"{family}-{label}@{k}", old_m, new_m, new_m - old_m))
    return rows


def main() -> None:
    args = parse_args()
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    font = set_korean_font()

    old_dep = load_json(args.old_deposit_json)
    old_fund = load_json(args.old_fund_json)
    tuned_dep = load_json(args.tuned_deposit_json)
    tuned_fund = load_json(args.tuned_fund_json)

    rows = collect_rows(old_dep, tuned_dep, "Deposit") + collect_rows(old_fund, tuned_fund, "Fund")
    labels = [r[0] for r in rows]
    old_vals = np.array([r[1] for r in rows], dtype=float)
    new_vals = np.array([r[2] for r in rows], dtype=float)
    delta_vals = np.array([r[3] for r in rows], dtype=float)
    x = np.arange(len(labels))
    w = 0.35

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={"height_ratios": [2.2, 1.2]})

    b1 = ax1.bar(x - w / 2, old_vals, width=w, label="Old baseline (v3)", color="#A0A0A0")
    b2 = ax1.bar(x + w / 2, new_vals, width=w, label="Tuned (2026-05-07)", color="#2E86AB")
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Model NDCG")
    ax1.set_title("옛 Baseline(v3) 대비 Utility Tuning 후 Model NDCG 비교", fontsize=20, pad=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right", fontsize=10)
    ax1.grid(axis="y", linestyle="--", alpha=0.25)
    ax1.legend(loc="lower right")

    for bars in (b1, b2):
        for p in bars:
            h = p.get_height()
            ax1.text(p.get_x() + p.get_width() / 2, h + 0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=9)

    colors = np.where(delta_vals >= 0, "#27AE60", "#C0392B")
    b3 = ax2.bar(x, delta_vals, width=0.55, color=colors)
    ax2.axhline(0.0, color="#444444", linewidth=1)
    ymin = min(-0.25, float(delta_vals.min() - 0.05))
    ymax = max(0.25, float(delta_vals.max() + 0.05))
    ax2.set_ylim(ymin, ymax)
    ax2.set_ylabel("ΔNDCG (Tuned - Old)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=30, ha="right", fontsize=10)
    ax2.grid(axis="y", linestyle="--", alpha=0.25)

    for p in b3:
        h = p.get_height()
        ax2.text(
            p.get_x() + p.get_width() / 2,
            h + (0.01 if h >= 0 else -0.01),
            f"{h:+.3f}",
            ha="center",
            va="bottom" if h >= 0 else "top",
            fontsize=9,
        )

    fig.text(
        0.01,
        0.01,
        f"old=deposit_v3+fund_v3, tuned=utility_tuning_deposit+fund | font={font}",
        fontsize=9,
        color="#666666",
    )
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=300)
    plt.close(fig)

    lines = ["label,old_model_ndcg,tuned_model_ndcg,delta"]
    for label, old_m, new_m, d in rows:
        lines.append(f"{label},{old_m:.6f},{new_m:.6f},{d:.6f}")
    args.out_csv.write_text("\n".join(lines), encoding="utf-8")

    print(f"saved: {args.out_png}")
    print(f"saved: {args.out_csv}")


if __name__ == "__main__":
    main()
