#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


def _first_available_score20(
    df: pd.DataFrame,
    direct_col: str,
    fallback_col_01: str,
    score_mode: str = "auto",
) -> pd.Series:
    if score_mode == "internal":
        if fallback_col_01 in df.columns:
            return (pd.to_numeric(df[fallback_col_01], errors="coerce") * 20.0).clip(0.0, 20.0)
        return pd.Series(np.nan, index=df.index)
    if score_mode == "llm_direct":
        if direct_col in df.columns:
            return pd.to_numeric(df[direct_col], errors="coerce").clip(0.0, 20.0)
        return pd.Series(np.nan, index=df.index)

    if direct_col in df.columns and df[direct_col].notna().any():
        return pd.to_numeric(df[direct_col], errors="coerce").clip(0.0, 20.0)
    if fallback_col_01 in df.columns:
        return (pd.to_numeric(df[fallback_col_01], errors="coerce") * 20.0).clip(0.0, 20.0)
    return pd.Series(np.nan, index=df.index)


def build_score_table(df: pd.DataFrame, score_mode: str = "auto") -> pd.DataFrame:
    work = df.copy()
    work = work[work.get("status", "ok").eq("ok")].copy()
    if work.empty:
        return pd.DataFrame()

    out = pd.DataFrame(index=work.index)
    out["model"] = work["model_input"].astype(str)

    out["Personalization"] = _first_available_score20(
        work, "llm_personalization_score20", "personalization", score_mode=score_mode
    )
    out["Product Grounding"] = _first_available_score20(
        work, "llm_product_grounding_score20", "product_grounding", score_mode=score_mode
    )
    out["Terminology Clarity"] = _first_available_score20(
        work, "llm_terminology_clarity_score20", "terminology_clarity", score_mode=score_mode
    )
    if score_mode == "internal":
        if "compliance" in work.columns:
            out["Compliance"] = (pd.to_numeric(work["compliance"], errors="coerce") * 20.0).clip(0.0, 20.0)
        else:
            out["Compliance"] = pd.Series(np.nan, index=work.index)
    elif score_mode == "llm_direct":
        if "llm_compliance_score20" in work.columns:
            out["Compliance"] = pd.to_numeric(work["llm_compliance_score20"], errors="coerce").clip(0.0, 20.0)
        else:
            out["Compliance"] = pd.Series(np.nan, index=work.index)
    else:
        # auto: prefer internal for compliance, fallback to direct
        if "compliance" in work.columns and work["compliance"].notna().any():
            out["Compliance"] = (pd.to_numeric(work["compliance"], errors="coerce") * 20.0).clip(0.0, 20.0)
        else:
            out["Compliance"] = _first_available_score20(
                work, "llm_compliance_score20", "compliance", score_mode=score_mode
            )

    # UG: center 10 (no change), positive gain >10, negative gain <10
    if "mean_delta_total_100" in work.columns and work["mean_delta_total_100"].notna().any():
        ug20 = 10.0 + pd.to_numeric(work["mean_delta_total_100"], errors="coerce") / 10.0
    else:
        ug20 = 10.0 + pd.to_numeric(work.get("understanding_gain", 0.0), errors="coerce") * 10.0
    out["Understanding Gain"] = ug20.clip(0.0, 20.0)

    if "misinterpretation_rate_weighted" in work.columns and work["misinterpretation_rate_weighted"].notna().any():
        mr = pd.to_numeric(work.get("misinterpretation_rate_weighted", np.nan), errors="coerce")
    else:
        mr = pd.to_numeric(work.get("misinterpretation_rate", np.nan), errors="coerce")
    out["Misinterpretation Control"] = (1.0 - mr).clip(0.0, 1.0) * 20.0

    metric_cols = [
        "Personalization",
        "Product Grounding",
        "Terminology Clarity",
        "Compliance",
        "Understanding Gain",
        "Misinterpretation Control",
    ]
    out["Overall"] = out[metric_cols].mean(axis=1, skipna=True)

    out = out.set_index("model")
    out = out.sort_values("Overall", ascending=False)
    return out


def save_heatmap(score_table: pd.DataFrame, out_png: Path, title: str) -> None:
    if score_table.empty:
        return

    annot = score_table.copy()
    for c in annot.columns:
        annot[c] = annot[c].map(lambda x: "-" if pd.isna(x) else f"{float(x):.1f}")

    fig_h = max(4.5, 1.0 + 0.7 * len(score_table.index))
    fig, ax = plt.subplots(figsize=(16, fig_h))
    sns.heatmap(
        score_table,
        annot=annot,
        fmt="",
        cmap="YlGnBu",
        vmin=0,
        vmax=20,
        linewidths=0.5,
        cbar_kws={"label": "score (0~20)"},
        ax=ax,
    )
    ax.set_title(title, fontsize=20, pad=12)
    ax.set_xlabel("metric", fontsize=12)
    ax.set_ylabel("model", fontsize=12)
    ax.tick_params(axis="x", labelrotation=90)
    ax.tick_params(axis="y", labelrotation=0)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create model-wise LLM benchmark heatmap (0~20 scale)")
    p.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("reports/e2e/llm_model_benchmark/20260522_034605/model_benchmark_summary.csv"),
    )
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument(
        "--title",
        type=str,
        default="모델별 LLM 직접/환산 채점 지표 (0~20)",
    )
    p.add_argument(
        "--score-mode",
        choices=["auto", "internal", "llm_direct"],
        default="auto",
        help=(
            "auto: direct llm score 우선 + 내부환산 fallback, "
            "internal: 내부 evaluator(0~1)*20만 사용, "
            "llm_direct: llm_*_score20만 사용"
        ),
    )
    return p.parse_args()


def main() -> None:
    matplotlib.use("Agg")
    sns.set_theme(style="white")
    font_name = set_korean_font()

    args = parse_args()
    if args.out_dir is None:
        out_dir = args.summary_csv.parent
    else:
        out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.summary_csv)
    score_table = build_score_table(df, score_mode=str(args.score_mode))

    out_png = out_dir / "model_metric_score20_heatmap.png"
    out_csv = out_dir / "model_metric_score20_heatmap_table.csv"

    if score_table.empty:
        msg = "No usable rows (status=ok) for heatmap generation."
        (out_dir / "model_metric_score20_heatmap_README.md").write_text(msg + "\n", encoding="utf-8")
        print(msg)
        return

    score_table.to_csv(out_csv, encoding="utf-8-sig")
    save_heatmap(score_table, out_png, args.title)

    readme = [
        "# Model-wise LLM Benchmark Heatmap",
        "",
        f"- input: `{args.summary_csv}`",
        f"- score_mode: `{args.score_mode}`",
        f"- font: `{font_name}`",
        f"- figure: `{out_png}`",
        f"- table: `{out_csv}`",
        "",
        "## Metric Mapping",
        "- Personalization / Product Grounding / Terminology Clarity:",
        "  - prefer direct LLM score columns (`llm_*_score20`)",
        "  - fallback to evaluator 0~1 columns × 20",
        "- Compliance: default is internal evaluator score × 20 (if available).",
        "  - if internal compliance is missing, fallback to `llm_compliance_score20`.",
        "- Understanding Gain: `10 + understanding_gain*10` (or `10 + mean_delta_total_100/10`) clipped to 0~20",
        "- Misinterpretation Control: `(1 - misinterpretation_rate) * 20`",
        "- Overall: mean of six metrics",
    ]
    (out_dir / "model_metric_score20_heatmap_README.md").write_text("\n".join(readme), encoding="utf-8")
    print(f"saved: {out_png}")
    print(f"saved: {out_csv}")


if __name__ == "__main__":
    main()
