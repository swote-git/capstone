#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from common.config import RecommenderConfig
from explainer.llm_renderer import OpenAILLMRenderer
from recommender.engine import ThinFilerRecommender
from user_parser.tps import compute_tps_scores, parse_custom_user_frame


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate full E2E report bundle: sample input/result/LLM intermediates + profile E2E + user-type stats")
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument(
        "--profile-csv",
        type=Path,
        default=Path("data/thin_filer/신파일러_군집_최종_피처_통합.csv"),
    )
    p.add_argument("--sample-users-train", type=int, default=500)
    p.add_argument("--max-train-users", type=int, default=400)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--profile-sample-per-type", type=int, default=40)
    p.add_argument(
        "--profile-llm-per-type",
        type=int,
        default=1,
        help="Number of profile users per type to run explanation pipeline on (LLM call unit when llm-scope=all)",
    )
    p.add_argument("--out-dir", type=Path, default=Path("reports/e2e/profile_bundle"))
    p.add_argument("--use-llm", action="store_true")
    p.add_argument("--llm-model", type=str, default="gpt-5-mini")
    p.add_argument(
        "--llm-scope",
        choices=["sample", "all"],
        default="sample",
        help="Where to apply live LLM rendering: sample only or all profile E2E rows",
    )
    p.add_argument("--no-template-fallback", action="store_true")
    return p.parse_args()


def set_korean_font() -> str:
    candidates = ["NanumGothic", "Noto Sans CJK KR", "Noto Sans KR", "AppleGothic", "Malgun Gothic"]
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = next((c for c in candidates if c in available), "DejaVu Sans")
    plt.rcParams["font.family"] = chosen
    plt.rcParams["axes.unicode_minus"] = False
    return chosen


def classify_user_type(df: pd.DataFrame) -> pd.Series:
    q_trust = df["s_trust"].quantile([0.33, 0.66]).to_dict()
    q_activity = df["s_activity"].quantile([0.33, 0.66]).to_dict()
    q_potential = df["s_potential"].quantile([0.33, 0.66]).to_dict()

    trust_lo = float(q_trust[0.33])
    trust_hi = float(q_trust[0.66])
    act_hi = float(q_activity[0.66])
    pot_hi = float(q_potential[0.66])

    cond_recovery = (df["OVERDUE_CNT"] > 0) | (df["s_trust"] <= trust_lo)
    cond_growth = (df["s_potential"] >= pot_hi) & (df["s_activity"] >= act_hi)
    cond_stable = (df["s_trust"] >= trust_hi) & (df["risk_tol"] <= 1.4)
    cond_digital = (df["s_activity"] >= act_hi) & (df["digital_behavior_freq"] >= 0.6)

    return pd.Series(
        np.select(
            [cond_recovery, cond_growth, cond_stable, cond_digital],
            ["회복관리형", "성장잠재형", "안정선호형", "디지털활동형"],
            default="균형형",
        ),
        index=df.index,
    )


def to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def to_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def product_lookup(rec: ThinFilerRecommender) -> pd.DataFrame:
    assert rec.products is not None
    cols = ["product_id", "product_name", "product_family", "risk_level", "liquidity_level", "horizon", "complexity"]
    return rec.products[cols].drop_duplicates("product_id").copy()


def run_recommendation_with_details(
    rec: ThinFilerRecommender,
    user_row: pd.Series,
    k: int,
    llm_renderer: Optional[OpenAILLMRenderer],
    fallback_to_template_on_verify_fail: bool,
) -> Dict[str, Any]:
    recommend = rec.recommend(user_row, k=k)
    explain = rec.explain_recommendation_with(
        user_row,
        k=k,
        llm_renderer=llm_renderer,
        fallback_to_template_on_verify_fail=fallback_to_template_on_verify_fail,
    )
    return {"recommend": recommend, "explain": explain}


def summarize_topk_with_product_info(
    rec_result: Dict[str, Any],
    pmap: pd.DataFrame,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rec_result.get("recommendations", []):
        pid = str(r.get("product_id", ""))
        row = pmap[pmap["product_id"].astype(str).eq(pid)]
        if row.empty:
            out.append({"product_id": pid, "score": to_float(r.get("score", 0.0))})
            continue
        rr = row.iloc[0]
        out.append(
            {
                "product_id": pid,
                "product_name": str(rr.get("product_name", "")),
                "product_family": str(rr.get("product_family", "")),
                "risk_level": to_int(rr.get("risk_level", 0)),
                "liquidity_level": to_int(rr.get("liquidity_level", 0)),
                "horizon": str(rr.get("horizon", "")),
                "complexity": to_int(rr.get("complexity", 0)),
                "score": to_float(r.get("score", 0.0)),
            }
        )
    return out


def build_sample_section(
    rec: ThinFilerRecommender,
    raw_df: pd.DataFrame,
    parsed_df: pd.DataFrame,
    pmap: pd.DataFrame,
    top_k: int,
    llm_renderer: Optional[OpenAILLMRenderer],
    fallback_to_template_on_verify_fail: bool,
) -> Dict[str, Any]:
    sample_raw = raw_df.sort_values("tps_score", ascending=False).iloc[0]
    sample_parsed = parsed_df[parsed_df["CUST_ID"].astype(str).eq(str(sample_raw["CUST_ID"]))].iloc[0]

    bundle = run_recommendation_with_details(
        rec=rec,
        user_row=sample_parsed,
        k=top_k,
        llm_renderer=llm_renderer,
        fallback_to_template_on_verify_fail=fallback_to_template_on_verify_fail,
    )
    topk = summarize_topk_with_product_info(bundle["recommend"], pmap)

    return {
        "sample_input_raw": sample_raw.to_dict(),
        "sample_input_parsed": sample_parsed.to_dict(),
        "recommendation_output": topk,
        "llm_pipeline_output": bundle["explain"],
    }


def build_profile_e2e_section(
    rec: ThinFilerRecommender,
    raw_df: pd.DataFrame,
    parsed_df: pd.DataFrame,
    pmap: pd.DataFrame,
    top_k: int,
    per_type: int,
    llm_renderer: Optional[OpenAILLMRenderer],
    fallback_to_template_on_verify_fail: bool,
) -> List[Dict[str, Any]]:
    sampled_groups: List[pd.DataFrame] = []
    for user_type, g in raw_df.groupby("user_type"):
        n = min(int(per_type), len(g))
        if n <= 0:
            continue
        s = g.sample(n=n, random_state=42).copy()
        s["user_type"] = user_type
        sampled_groups.append(s)

    if not sampled_groups:
        return []

    reps = pd.concat(sampled_groups, ignore_index=True).sort_values(
        ["user_type", "tps_score"], ascending=[True, False]
    )
    rows: List[Dict[str, Any]] = []
    for _, raw in reps.iterrows():
        uid = str(raw["CUST_ID"])
        parsed = parsed_df[parsed_df["CUST_ID"].astype(str).eq(uid)].iloc[0]
        bundle = run_recommendation_with_details(
            rec=rec,
            user_row=parsed,
            k=top_k,
            llm_renderer=llm_renderer,
            fallback_to_template_on_verify_fail=fallback_to_template_on_verify_fail,
        )
        rows.append(
            {
                "user_type": str(raw["user_type"]),
                "user_id": uid,
                "tps_score": to_float(raw.get("tps_score", 0.0)),
                "risk_tol": to_float(parsed.get("risk_tol", 0.0)),
                "liquidity_need": to_float(parsed.get("liquidity_need", 0.0)),
                "topk": summarize_topk_with_product_info(bundle["recommend"], pmap),
                "explain": bundle["explain"],
            }
        )
    return rows


def build_user_type_stats_section(
    rec: ThinFilerRecommender,
    raw_df: pd.DataFrame,
    parsed_df: pd.DataFrame,
    pmap: pd.DataFrame,
    top_k: int,
    sample_per_type: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sampled_uids = (
        raw_df.groupby("user_type", group_keys=False)
        .apply(lambda g: g.sample(n=min(sample_per_type, len(g)), random_state=42))
        ["CUST_ID"]
        .astype(str)
        .tolist()
    )
    eval_df = parsed_df[parsed_df["CUST_ID"].astype(str).isin(sampled_uids)].copy()

    rec_rows: List[Dict[str, Any]] = []
    for _, row in eval_df.iterrows():
        uid = str(row["CUST_ID"])
        utype = str(raw_df.loc[raw_df["CUST_ID"].astype(str).eq(uid), "user_type"].iloc[0])
        res = rec.recommend(row, k=top_k)
        top1 = res["recommendations"][0] if res["recommendations"] else {"product_id": "", "score": 0.0}
        pid = str(top1.get("product_id", ""))
        pinfo = pmap[pmap["product_id"].astype(str).eq(pid)]
        fam = str(pinfo["product_family"].iloc[0]) if not pinfo.empty else "unknown"
        risk = to_int(pinfo["risk_level"].iloc[0], 0) if not pinfo.empty else 0

        rec_rows.append(
            {
                "user_id": uid,
                "user_type": utype,
                "top1_product_id": pid,
                "top1_score": to_float(top1.get("score", 0.0)),
                "top1_family": fam,
                "top1_risk_level": risk,
                "tps_score": to_float(row.get("tps_score", 0.0)),
                "risk_tol": to_float(row.get("risk_tol", 0.0)),
                "liquidity_need": to_float(row.get("liquidity_need", 0.0)),
            }
        )

    detail = pd.DataFrame(rec_rows)
    family_mix = (
        detail.groupby(["user_type", "top1_family"], as_index=False)
        .agg(n=("user_id", "count"))
        .sort_values(["user_type", "top1_family"])
    )
    family_mix["ratio"] = family_mix["n"] / family_mix.groupby("user_type")["n"].transform("sum")

    summary = (
        detail.groupby("user_type", as_index=False)
        .agg(
            n_users=("user_id", "nunique"),
            mean_top1_score=("top1_score", "mean"),
            std_top1_score=("top1_score", "std"),
            mean_top1_risk=("top1_risk_level", "mean"),
            mean_tps_score=("tps_score", "mean"),
            mean_risk_tol=("risk_tol", "mean"),
            mean_liquidity_need=("liquidity_need", "mean"),
        )
        .sort_values("n_users", ascending=False)
    )
    return summary, detail


def plot_stats(detail: pd.DataFrame, summary: pd.DataFrame, out_dir: Path) -> List[str]:
    figs: List[str] = []
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1) user type count
    cnt = detail.groupby("user_type", as_index=False).agg(n=("user_id", "nunique")).sort_values("n", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=cnt, x="user_type", y="n", color="#2A9D8F", ax=ax)
    ax.set_title("사용자 유형 분포")
    ax.set_xlabel("user_type")
    ax.set_ylabel("user_count")
    for i, v in enumerate(cnt["n"].tolist()):
        ax.text(i, v + 0.2, str(int(v)), ha="center", fontsize=9)
    fig.tight_layout()
    p1 = fig_dir / "01_user_type_count.png"
    fig.savefig(p1, dpi=180)
    plt.close(fig)
    figs.append(str(p1))

    # 2) top1 family mix
    mix = (
        detail[["user_id", "user_type", "top1_family"]]
        .drop_duplicates()
        .groupby(["user_type", "top1_family"], as_index=False)
        .agg(n=("user_id", "count"))
    )
    mix["ratio"] = mix["n"] / mix.groupby("user_type")["n"].transform("sum")
    pivot = mix.pivot(index="user_type", columns="top1_family", values="ratio").fillna(0.0)
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="bar", stacked=True, ax=ax, colormap="Set2")
    ax.set_title("사용자 유형별 Top-1 상품군 비중")
    ax.set_xlabel("user_type")
    ax.set_ylabel("ratio")
    ax.legend(title="top1_family", loc="upper right")
    fig.tight_layout()
    p2 = fig_dir / "02_top1_family_mix.png"
    fig.savefig(p2, dpi=180)
    plt.close(fig)
    figs.append(str(p2))

    # 3) mean top1 score
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=summary.sort_values("mean_top1_score", ascending=False), x="user_type", y="mean_top1_score", color="#457B9D", ax=ax)
    ax.set_title("사용자 유형별 평균 Top-1 추천점수")
    ax.set_xlabel("user_type")
    ax.set_ylabel("mean_top1_score")
    fig.tight_layout()
    p3 = fig_dir / "03_mean_top1_score_by_type.png"
    fig.savefig(p3, dpi=180)
    plt.close(fig)
    figs.append(str(p3))

    # 4) risk heatmap
    risk_map = (
        detail.groupby(["user_type", "top1_risk_level"], as_index=False)
        .agg(n=("user_id", "count"))
    )
    risk_map["ratio"] = risk_map["n"] / risk_map.groupby("user_type")["n"].transform("sum")
    hv = risk_map.pivot(index="user_type", columns="top1_risk_level", values="ratio").fillna(0.0)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(hv, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
    ax.set_title("사용자 유형별 Top-1 위험등급 비중")
    ax.set_xlabel("top1_risk_level")
    ax.set_ylabel("user_type")
    fig.tight_layout()
    p4 = fig_dir / "04_top1_risk_heatmap.png"
    fig.savefig(p4, dpi=180)
    plt.close(fig)
    figs.append(str(p4))

    return figs


def to_short_json(data: Any, max_chars: int = 1500) -> str:
    txt = json.dumps(data, ensure_ascii=False, indent=2)
    if len(txt) <= max_chars:
        return txt
    return txt[:max_chars] + "\n... (truncated)"


def write_report(
    out_dir: Path,
    font_name: str,
    sample_section: Dict[str, Any],
    profile_rows: List[Dict[str, Any]],
    summary: pd.DataFrame,
    detail: pd.DataFrame,
    figures: List[str],
) -> None:
    lines: List[str] = []
    lines.append("# E2E 통합 리포트 (샘플 입력/추천 결과/LLM 파이프라인/유형 통계)")
    lines.append("")
    lines.append(f"- font: `{font_name}`")
    lines.append(f"- generated_at_dir: `{out_dir}`")
    lines.append("")
    lines.append("## 1) 샘플 입력과 추천 결과")
    lines.append("- 입력(원본 사용자 프로파일, 일부):")
    lines.append("```json")
    lines.append(to_short_json(sample_section["sample_input_raw"], max_chars=1200))
    lines.append("```")
    lines.append("- 추천 결과(top-k):")
    lines.append("```json")
    lines.append(to_short_json(sample_section["recommendation_output"], max_chars=1500))
    lines.append("```")
    lines.append("")
    lines.append("## 2) LLM 파이프라인 중간 산출물")
    lines.append("- explanation_object + reason_signals + verification + render_source를 포함한 원본은 아래 raw JSON 참고")
    lines.append(f"- raw file: `{out_dir / 'raw' / 'sample_e2e_payload.json'}`")
    lines.append("- LLM 중간 산출물 키:")
    lines.append("  - `llm_intermediate.llm_rendered_explanation`")
    lines.append("  - `llm_intermediate.llm_verification`")
    lines.append("  - `render_source` (`llm` | `template_fallback` | `template`)")
    lines.append("")
    lines.append("## 3) 신파일러 프로파일 유형별 E2E")
    lines.append(f"- profile E2E rows: `{len(profile_rows)}`")
    per_type_counts: Dict[str, int] = {}
    for r in profile_rows:
        per_type_counts[str(r["user_type"])] = per_type_counts.get(str(r["user_type"]), 0) + 1
    lines.append(f"- by user_type: `{per_type_counts}`")
    lines.append("- 상세 예시는 유형별 최대 3건만 표시")
    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for row in profile_rows:
        by_type.setdefault(str(row["user_type"]), []).append(row)
    for utype, rows in sorted(by_type.items(), key=lambda x: x[0]):
        lines.append(f"- [{utype}]")
        for row in rows[:3]:
            lines.append(f"  - user_id=`{row['user_id']}` / tps={row['tps_score']:.2f}")
            lines.append(f"    - risk_tol={row['risk_tol']:.3f}, liquidity_need={row['liquidity_need']:.3f}")
            top1 = row["topk"][0] if row["topk"] else {}
            lines.append(
                f"    - top1: {top1.get('product_id','')} ({top1.get('product_family','')}) score={to_float(top1.get('score',0.0)):.4f}"
            )
            exp = row["explain"]["recommendations"][0] if row["explain"]["recommendations"] else {}
            ver = exp.get("verification", {})
            lines.append(
                f"    - explain: render_source={exp.get('render_source','')}, pass={ver.get('passed')}, hr={to_float(ver.get('hallucination_rate',0.0)):.3f}"
            )
    lines.append("")
    lines.append("## 4) 여러 사용자 유형 통계")
    lines.append("- summary csv:")
    lines.append(f"  - `{out_dir / 'user_type_summary.csv'}`")
    lines.append("  - `{}`".format(out_dir / "user_type_detail.csv"))
    lines.append("- figures:")
    for fp in figures:
        lines.append(f"  - `{fp}`")
    lines.append("")
    lines.append("### 사용자 유형 요약 표")
    lines.append("```")
    lines.append(summary.to_string(index=False))
    lines.append("```")

    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    font_name = set_korean_font()

    # 1) Recommender train
    cfg = RecommenderConfig(data_root=args.data_root, recommender_family="all")
    rec = ThinFilerRecommender(cfg)
    snapshots = rec.build_user_snapshots(sample_users=args.sample_users_train)
    rec.load_products()
    rec.fit(snapshots=snapshots, max_users=args.max_train_users)
    pmap = product_lookup(rec)

    # 2) Load user profiles and parse
    raw_df = pd.read_csv(args.profile_csv)
    if not {"s_trust", "s_activity", "s_potential", "tps_score"}.issubset(raw_df.columns):
        raw_df = compute_tps_scores(raw_df)
    parsed_df = parse_custom_user_frame(raw_df)
    parsed_df["CUST_ID"] = parsed_df["CUST_ID"].astype(str)
    raw_df["CUST_ID"] = raw_df["CUST_ID"].astype(str)
    raw_df["user_type"] = classify_user_type(parsed_df)

    # 3) Optional LLM renderer
    llm_renderer_sample = OpenAILLMRenderer(model=args.llm_model) if args.use_llm else None
    llm_renderer_profile = llm_renderer_sample if (args.use_llm and args.llm_scope == "all") else None

    # 4) Sample E2E section
    sample_section = build_sample_section(
        rec=rec,
        raw_df=raw_df,
        parsed_df=parsed_df,
        pmap=pmap,
        top_k=args.top_k,
        llm_renderer=llm_renderer_sample,
        fallback_to_template_on_verify_fail=not args.no_template_fallback,
    )
    (raw_dir / "sample_e2e_payload.json").write_text(
        json.dumps(sample_section, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 5) Profile-type E2E section
    profile_rows = build_profile_e2e_section(
        rec=rec,
        raw_df=raw_df,
        parsed_df=parsed_df,
        pmap=pmap,
        top_k=args.top_k,
        per_type=args.profile_llm_per_type,
        llm_renderer=llm_renderer_profile,
        fallback_to_template_on_verify_fail=not args.no_template_fallback,
    )
    (raw_dir / "profile_type_e2e_payload.json").write_text(
        json.dumps(profile_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 6) Multi-user statistics
    summary, detail = build_user_type_stats_section(
        rec=rec,
        raw_df=raw_df,
        parsed_df=parsed_df,
        pmap=pmap,
        top_k=args.top_k,
        sample_per_type=args.profile_sample_per_type,
    )
    summary.to_csv(args.out_dir / "user_type_summary.csv", index=False, encoding="utf-8-sig")
    detail.to_csv(args.out_dir / "user_type_detail.csv", index=False, encoding="utf-8-sig")
    figures = plot_stats(detail, summary, args.out_dir)

    write_report(
        out_dir=args.out_dir,
        font_name=font_name,
        sample_section=sample_section,
        profile_rows=profile_rows,
        summary=summary,
        detail=detail,
        figures=figures,
    )

    print(f"saved: {args.out_dir / 'report.md'}")
    print(f"saved: {raw_dir / 'sample_e2e_payload.json'}")
    print(f"saved: {raw_dir / 'profile_type_e2e_payload.json'}")


if __name__ == "__main__":
    main()
