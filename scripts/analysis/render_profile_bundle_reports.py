#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


SECTION_SPECS = [
    ("상품 정보 요약", "[상품 정보 요약]"),
    ("추천 이유", "[추천 이유]"),
    ("유의사항", "[유의사항]"),
    ("대안 비교", "[대안 비교]"),
    ("한줄 요약", "[한줄 요약]"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render refreshed profile-bundle reports and LLM evaluation figures by user type")
    p.add_argument("--bundle-dir", type=Path, default=Path("reports/e2e/profile_bundle"))
    return p.parse_args()


def set_korean_font() -> str:
    candidates = ["NanumGothic", "Noto Sans CJK KR", "Noto Sans KR", "AppleGothic", "Malgun Gothic"]
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = next((c for c in candidates if c in available), "DejaVu Sans")
    plt.rcParams["font.family"] = chosen
    plt.rcParams["axes.unicode_minus"] = False
    return chosen


def safe_slug(text: str) -> str:
    t = re.sub(r"\s+", "_", str(text).strip())
    t = re.sub(r"[^0-9A-Za-z가-힣_]+", "", t)
    return t or "unknown"


def has_section(text: str, marker: str) -> int:
    return int(marker in (text or ""))


def flatten_records(sample_obj: Dict[str, Any], profile_obj: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    def push(scope: str, user_type: str, user_id: str, rank: int, rec: Dict[str, Any]) -> None:
        ver = rec.get("verification", {}) or {}
        llm_inter = rec.get("llm_intermediate")
        llm_ver = (llm_inter or {}).get("llm_verification", {}) if isinstance(llm_inter, dict) else {}
        eo = rec.get("explanation_object", {}) or {}
        rp = eo.get("recommended_product", {}) or {}
        text = rec.get("rendered_explanation", "") or ""

        row = {
            "scope": scope,
            "user_type": user_type,
            "user_id": user_id,
            "rank": int(rank),
            "product_id": str(rec.get("product_id", "")),
            "score": float(rec.get("score", 0.0)),
            "product_family": str(rp.get("family", "")),
            "product_risk": str(rp.get("risk", "")),
            "render_source": str(rec.get("render_source", "unknown")),
            "final_passed": int(bool(ver.get("passed", False))),
            "reason_alignment": float(ver.get("reason_alignment", 0.0)),
            "fact_consistency": int(bool(ver.get("fact_consistency", False))),
            "hallucination_rate": float(ver.get("hallucination_rate", 0.0)),
            "forbidden_claim_cnt": int(len(ver.get("forbidden_claims", []) or [])),
            "llm_attempted": int(llm_inter is not None or rec.get("render_source") == "llm"),
            "llm_passed": int(bool(llm_ver.get("passed", False))) if llm_inter is not None else int(rec.get("render_source") == "llm"),
            "text_len": int(len(text)),
        }
        for sec_name, marker in SECTION_SPECS:
            row[f"sec_{safe_slug(sec_name)}"] = has_section(text, marker)
        rows.append(row)

    # sample
    sample_user = sample_obj.get("sample_input_raw", {}).get("CUST_ID", "sample_user")
    for i, rec in enumerate(sample_obj.get("llm_pipeline_output", {}).get("recommendations", []), 1):
        push("sample", "샘플", str(sample_user), i, rec)

    # profile types
    for row in profile_obj:
        utype = str(row.get("user_type", "unknown"))
        uid = str(row.get("user_id", "unknown"))
        recs = row.get("explain", {}).get("recommendations", [])
        for i, rec in enumerate(recs, 1):
            push("profile", utype, uid, i, rec)

    return pd.DataFrame(rows)


def summarize_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {"overall": {}, "by_type": []}

    overall = {
        "n": int(len(df)),
        "llm_attempt_rate": float(df["llm_attempted"].mean()),
        "llm_pass_rate": float(df["llm_passed"].mean()),
        "fallback_rate": float((df["render_source"] == "template_fallback").mean()),
        "final_pass_rate": float(df["final_passed"].mean()),
        "mean_reason_alignment": float(df["reason_alignment"].mean()),
        "mean_hallucination_rate": float(df["hallucination_rate"].mean()),
        "mean_forbidden_claim_cnt": float(df["forbidden_claim_cnt"].mean()),
    }

    by_type = (
        df.groupby("user_type", as_index=False)
        .agg(
            n=("product_id", "count"),
            llm_attempt_rate=("llm_attempted", "mean"),
            llm_pass_rate=("llm_passed", "mean"),
            fallback_rate=("render_source", lambda s: float((s == "template_fallback").mean())),
            final_pass_rate=("final_passed", "mean"),
            mean_reason_alignment=("reason_alignment", "mean"),
            mean_hallucination_rate=("hallucination_rate", "mean"),
            mean_forbidden_claim_cnt=("forbidden_claim_cnt", "mean"),
        )
        .sort_values("user_type")
    )
    return {"overall": overall, "by_type": by_type}


def save_figures(df: pd.DataFrame, out_dir: Path) -> List[str]:
    figs: List[str] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    # Global source mix
    src = df["render_source"].value_counts().rename_axis("render_source").reset_index(name="n")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(data=src, x="render_source", y="n", palette="Set2", ax=ax)
    ax.set_title("전체 render_source 분포")
    for i, v in enumerate(src["n"].tolist()):
        ax.text(i, v + 0.05, str(int(v)), ha="center", fontsize=9)
    fig.tight_layout()
    p = out_dir / "00_overall_render_source.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    figs.append(str(p))

    # By-type figures
    for utype, g in df.groupby("user_type"):
        slug = safe_slug(utype)
        fig1, ax1 = plt.subplots(figsize=(6.5, 4.2))
        src_t = g["render_source"].value_counts().rename_axis("render_source").reset_index(name="n")
        sns.barplot(data=src_t, x="render_source", y="n", palette="Pastel1", ax=ax1)
        ax1.set_title(f"{utype} render_source 분포")
        for i, v in enumerate(src_t["n"].tolist()):
            ax1.text(i, v + 0.03, str(int(v)), ha="center", fontsize=9)
        fig1.tight_layout()
        p1 = out_dir / f"{slug}_01_render_source.png"
        fig1.savefig(p1, dpi=180)
        plt.close(fig1)
        figs.append(str(p1))

        # verifier metric bars
        vals = pd.DataFrame(
            [
                {"metric": "llm_pass_rate", "value": float(g["llm_passed"].mean())},
                {"metric": "final_pass_rate", "value": float(g["final_passed"].mean())},
                {"metric": "reason_alignment", "value": float(g["reason_alignment"].mean())},
                {"metric": "1-hallucination", "value": 1.0 - float(g["hallucination_rate"].mean())},
            ]
        )
        fig2, ax2 = plt.subplots(figsize=(7.0, 4.3))
        sns.barplot(data=vals, x="metric", y="value", color="#457B9D", ax=ax2)
        ax2.set_ylim(0, 1.05)
        ax2.set_title(f"{utype} LLM/Verifier 지표")
        for i, v in enumerate(vals["value"].tolist()):
            ax2.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
        fig2.tight_layout()
        p2 = out_dir / f"{slug}_02_metrics.png"
        fig2.savefig(p2, dpi=180)
        plt.close(fig2)
        figs.append(str(p2))

        # section coverage
        sec_cols = [f"sec_{safe_slug(s)}" for s, _ in SECTION_SPECS]
        sec_names = [s for s, _ in SECTION_SPECS]
        sec_values = [float(g[c].mean()) for c in sec_cols]
        sec_df = pd.DataFrame({"section": sec_names, "coverage": sec_values})
        fig3, ax3 = plt.subplots(figsize=(7.2, 4.5))
        sns.barplot(data=sec_df, x="section", y="coverage", color="#2A9D8F", ax=ax3)
        ax3.set_ylim(0, 1.05)
        ax3.set_title(f"{utype} 설명 섹션 충족률")
        ax3.tick_params(axis="x", rotation=20)
        for i, v in enumerate(sec_df["coverage"].tolist()):
            ax3.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
        fig3.tight_layout()
        p3 = out_dir / f"{slug}_03_section_coverage.png"
        fig3.savefig(p3, dpi=180)
        plt.close(fig3)
        figs.append(str(p3))

    return figs


def write_new_main_report(
    bundle_dir: Path,
    sample_obj: Dict[str, Any],
    metrics: Dict[str, Any],
    fig_paths: List[str],
) -> None:
    lines: List[str] = []
    lines.append("# E2E 결과 리포트 (새 포맷)")
    lines.append("")
    lines.append("## A. 샘플 입력")
    lines.append("```json")
    lines.append(json.dumps(sample_obj.get("sample_input_raw", {}), ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## B. 추천 결과 (Top-K)")
    lines.append("```json")
    lines.append(json.dumps(sample_obj.get("recommendation_output", []), ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## C. LLM 설명 출력")
    recs = sample_obj.get("llm_pipeline_output", {}).get("recommendations", [])
    if recs:
        lines.append("```text")
        lines.append(str(recs[0].get("rendered_explanation", "")))
        lines.append("```")
    else:
        lines.append("- 샘플 설명 결과가 없습니다.")
    lines.append("")
    lines.append("## D. LLM/Verifier 요약")
    ov = metrics.get("overall", {})
    for k in [
        "n",
        "llm_attempt_rate",
        "llm_pass_rate",
        "fallback_rate",
        "final_pass_rate",
        "mean_reason_alignment",
        "mean_hallucination_rate",
        "mean_forbidden_claim_cnt",
    ]:
        if k in ov:
            lines.append(f"- {k}: {ov[k]:.4f}" if isinstance(ov[k], float) else f"- {k}: {ov[k]}")
    lines.append("")
    lines.append("## E. 산출물 경로")
    lines.append(f"- raw/sample: `{bundle_dir / 'raw' / 'sample_e2e_payload.json'}`")
    lines.append(f"- raw/profile: `{bundle_dir / 'raw' / 'profile_type_e2e_payload.json'}`")
    lines.append(f"- llm eval report: `{bundle_dir / 'llm_evaluate_report.md'}`")
    lines.append(f"- llm eval csv: `{bundle_dir / 'llm_eval_metrics_by_type.csv'}`")
    lines.append("- figures:")
    for p in fig_paths:
        lines.append(f"  - `{p}`")
    (bundle_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def write_llm_eval_report(
    bundle_dir: Path,
    df: pd.DataFrame,
    metrics: Dict[str, Any],
    fig_paths: List[str],
) -> None:
    by_type: pd.DataFrame = metrics["by_type"]
    by_type_path = bundle_dir / "llm_eval_metrics_by_type.csv"
    detail_path = bundle_dir / "llm_eval_detail.csv"
    by_type.to_csv(by_type_path, index=False, encoding="utf-8-sig")
    df.to_csv(detail_path, index=False, encoding="utf-8-sig")

    lines: List[str] = []
    lines.append("# LLM Evaluate Report")
    lines.append("")
    lines.append("## 1) Overall")
    for k, v in metrics["overall"].items():
        lines.append(f"- {k}: {v:.4f}" if isinstance(v, float) else f"- {k}: {v}")
    lines.append("")
    lines.append("## 2) By User Type")
    lines.append("```")
    lines.append(by_type.to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("## 3) Figure Index")
    for p in fig_paths:
        if "figures_llm_eval" in p:
            lines.append(f"- `{p}`")
    lines.append("")
    lines.append("## 4) Data Files")
    lines.append(f"- `{by_type_path}`")
    lines.append(f"- `{detail_path}`")
    (bundle_dir / "llm_evaluate_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    bundle_dir: Path = args.bundle_dir
    raw_dir = bundle_dir / "raw"
    sample_path = raw_dir / "sample_e2e_payload.json"
    profile_path = raw_dir / "profile_type_e2e_payload.json"

    if not sample_path.exists() or not profile_path.exists():
        raise FileNotFoundError("profile_bundle raw files are missing. Run generate_e2e_profile_bundle.py first.")

    set_korean_font()
    sample_obj = json.loads(sample_path.read_text(encoding="utf-8"))
    profile_obj = json.loads(profile_path.read_text(encoding="utf-8"))

    df = flatten_records(sample_obj, profile_obj)
    metrics = summarize_metrics(df)

    fig_dir = bundle_dir / "figures_llm_eval"
    fig_paths = save_figures(df, fig_dir)
    write_new_main_report(bundle_dir, sample_obj, metrics, fig_paths)
    write_llm_eval_report(bundle_dir, df, metrics, fig_paths)
    print(f"saved: {bundle_dir / 'report.md'}")
    print(f"saved: {bundle_dir / 'llm_evaluate_report.md'}")
    print(f"saved: {bundle_dir / 'llm_eval_metrics_by_type.csv'}")


if __name__ == "__main__":
    main()
