#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from common.config import RecommenderConfig
from common.env import load_dotenv_file
from explainer.llm_renderer import OpenAILLMRenderer
from recommender.engine import ThinFilerRecommender
from user_parser.tps import compute_tps_scores, parse_custom_user_frame

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None


METRIC_SCORE_COLS = [
    "personalization_score20",
    "product_grounding_score20",
    "terminology_clarity_score20",
    "compliance_score20",
    "understanding_gain_score20",
    "misinterpretation_control_score20",
    "overall_score20",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Expanded LLM evaluation by user type with direct 0~20 LLM scoring")
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--profile-csv", type=Path, default=Path("data/thin_filer/신파일러_군집_최종_피처_통합.csv"))
    p.add_argument("--sample-users-train", type=int, default=120)
    p.add_argument("--max-train-users", type=int, default=90)
    p.add_argument("--per-type", type=int, default=5, help="Number of users per user_type for LLM explanation calls")
    p.add_argument("--top-k", type=int, default=1)
    p.add_argument("--llm-model", type=str, default="gpt-5-mini", help="LLM renderer model")
    p.add_argument("--use-explainer-moe", action="store_true")
    p.add_argument(
        "--compliance-rules-path",
        type=Path,
        default=Path("src/explainer/compliance_rules.txt"),
    )
    p.add_argument("--explainer-moe-debug", action="store_true")
    p.add_argument("--llm-score-model", type=str, default="gpt-5-mini", help="LLM evaluator model for 0~20 scoring")
    p.add_argument("--score-timeout", type=float, default=30.0)
    p.add_argument("--score-max-retries", type=int, default=2)
    p.add_argument("--out-dir", type=Path, default=Path("reports/e2e/profile_bundle/llm_eval_expanded"))
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


def _clip20(value: Any) -> float:
    try:
        x = float(value)
    except Exception:
        return float("nan")
    return float(max(0.0, min(20.0, x)))


def _extract_json(raw: str) -> Dict[str, Any]:
    t = (raw or "").strip()
    if not t:
        return {}
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    m = re.search(r"\{[\s\S]*\}", t)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
        if isinstance(obj, dict):
            return obj
    except Exception:
        return {}
    return {}


class LLMMetricScorer:
    def __init__(
        self,
        model: str = "gpt-5-mini",
        api_key: str | None = None,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
    ) -> None:
        if OpenAI is None:
            raise ImportError("openai package is required")
        if api_key is None and not os.getenv("OPENAI_API_KEY"):
            load_dotenv_file()
        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = OpenAI(
            api_key=resolved_api_key,
            timeout=float(timeout_seconds),
            max_retries=int(max_retries),
        )
        self.system_prompt = (
            "당신은 금융 추천 설명 평가자입니다. 반드시 입력 데이터만 사용해 평가하세요.\n"
            "절대 규칙:\n"
            "1) 각 지표를 0~20 실수(소수점 허용)로 채점\n"
            "2) 근거(reasoning)는 입력에 존재하는 사실만 사용\n"
            "3) 입력에 없는 사실 추정 금지\n"
            "4) 최종 출력은 JSON 객체 하나만 반환\n"
            "\n"
            "채점 지표:\n"
            "- personalization_score20\n"
            "- product_grounding_score20\n"
            "- terminology_clarity_score20\n"
            "- compliance_score20\n"
            "- understanding_gain_score20\n"
            "- misinterpretation_control_score20\n"
            "- overall_score20 (위 6개 종합 평균 또는 보수적 종합)\n"
            "\n"
            "JSON 스키마:\n"
            "{\n"
            '  "personalization_score20": 12.34,\n'
            '  "product_grounding_score20": 12.34,\n'
            '  "terminology_clarity_score20": 12.34,\n'
            '  "compliance_score20": 12.34,\n'
            '  "understanding_gain_score20": 12.34,\n'
            '  "misinterpretation_control_score20": 12.34,\n'
            '  "overall_score20": 12.34,\n'
            '  "reasoning_personalization": "...",\n'
            '  "reasoning_product_grounding": "...",\n'
            '  "reasoning_terminology_clarity": "...",\n'
            '  "reasoning_compliance": "...",\n'
            '  "reasoning_understanding_gain": "...",\n'
            '  "reasoning_misinterpretation": "..."\n'
            "}"
        )

    def score(self, payload: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        input_text = json.dumps(payload, ensure_ascii=False)
        raw = ""

        # 1) responses API (retry for parse robustness)
        for _ in range(2):
            try:
                resp = self.client.responses.create(
                    model=self.model,
                    input=[
                        {"role": "system", "content": [{"type": "input_text", "text": self.system_prompt}]},
                        {"role": "user", "content": [{"type": "input_text", "text": input_text}]},
                    ],
                )
                raw = (getattr(resp, "output_text", None) or "").strip()
                obj = _extract_json(raw)
                if obj:
                    return self._normalize(obj), raw
            except Exception as e:
                raw = f"responses_error: {e}"

        # 2) chat completions fallback (API compatibility only)
        for _ in range(2):
            try:
                resp2 = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": input_text},
                    ],
                    temperature=0,
                )
                raw2 = (resp2.choices[0].message.content or "").strip()
                obj2 = _extract_json(raw2)
                if obj2:
                    return self._normalize(obj2), raw2
                raw = raw2 or raw
            except Exception as e2:
                raw = (raw + " | " if raw else "") + f"chat_error: {e2}"

        return {}, raw

    @staticmethod
    def _normalize(obj: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k in METRIC_SCORE_COLS:
            out[k] = _clip20(obj.get(k, float("nan")))

        for k in [
            "reasoning_personalization",
            "reasoning_product_grounding",
            "reasoning_terminology_clarity",
            "reasoning_compliance",
            "reasoning_understanding_gain",
            "reasoning_misinterpretation",
        ]:
            out[k] = str(obj.get(k, "")).strip()
        return out


def summarize_rows(detail: pd.DataFrame) -> pd.DataFrame:
    return (
        detail.groupby("user_type", as_index=False)
        .agg(
            n_calls=("user_id", "count"),
            n_users=("user_id", "nunique"),
            n_scored=("llm_metric_scored", "sum"),
            scoring_success_rate=("llm_metric_scored", "mean"),
            mean_latency_sec=("latency_sec", "mean"),
            llm_pass_rate=("llm_passed", "mean"),
            final_pass_rate=("final_passed", "mean"),
            mean_reason_alignment=("reason_alignment", "mean"),
            mean_hallucination_rate=("hallucination_rate", "mean"),
            mean_forbidden_claim_cnt=("forbidden_claim_cnt", "mean"),
            mean_personalization_score20=("personalization_score20", "mean"),
            mean_product_grounding_score20=("product_grounding_score20", "mean"),
            mean_terminology_clarity_score20=("terminology_clarity_score20", "mean"),
            mean_compliance_score20=("compliance_score20", "mean"),
            mean_understanding_gain_score20=("understanding_gain_score20", "mean"),
            mean_misinterpretation_control_score20=("misinterpretation_control_score20", "mean"),
            mean_overall_score20=("overall_score20", "mean"),
        )
        .sort_values("user_type")
    )


def plot_figures(detail: pd.DataFrame, summary: pd.DataFrame, out_dir: Path) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    figs: List[str] = []

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=summary, x="user_type", y="n_calls", color="#457B9D", ax=ax)
    ax.set_title("유형별 LLM 설명 호출 수")
    ax.set_xlabel("user_type")
    ax.set_ylabel("n_calls")
    for i, v in enumerate(summary["n_calls"].tolist()):
        ax.text(i, v + 0.1, str(int(v)), ha="center", fontsize=9)
    fig.tight_layout()
    p1 = out_dir / "01_calls_by_type.png"
    fig.savefig(p1, dpi=180)
    plt.close(fig)
    figs.append(str(p1))

    src = detail.groupby(["user_type", "render_source"], as_index=False).agg(n=("user_id", "count"))
    src["ratio"] = src["n"] / src.groupby("user_type")["n"].transform("sum")
    pv = src.pivot(index="user_type", columns="render_source", values="ratio").fillna(0.0)
    fig, ax = plt.subplots(figsize=(9, 5))
    pv.plot(kind="bar", stacked=True, ax=ax, colormap="Set2")
    ax.set_title("유형별 render_source 비중")
    ax.set_xlabel("user_type")
    ax.set_ylabel("ratio")
    fig.tight_layout()
    p2 = out_dir / "02_render_source_ratio_by_type.png"
    fig.savefig(p2, dpi=180)
    plt.close(fig)
    figs.append(str(p2))

    mdf = summary.melt(
        id_vars=["user_type"],
        value_vars=["scoring_success_rate", "llm_pass_rate", "final_pass_rate"],
        var_name="metric",
        value_name="value",
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=mdf, x="user_type", y="value", hue="metric", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_title("유형별 LLM 성공률 / pass 비율")
    ax.set_xlabel("user_type")
    ax.set_ylabel("rate")
    fig.tight_layout()
    p3 = out_dir / "03_success_pass_by_type.png"
    fig.savefig(p3, dpi=180)
    plt.close(fig)
    figs.append(str(p3))

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=summary, x="user_type", y="mean_latency_sec", color="#2A9D8F", ax=ax)
    ax.set_title("유형별 평균 호출 지연(초)")
    ax.set_xlabel("user_type")
    ax.set_ylabel("mean_latency_sec")
    fig.tight_layout()
    p4 = out_dir / "04_latency_by_type.png"
    fig.savefig(p4, dpi=180)
    plt.close(fig)
    figs.append(str(p4))

    score_cols = [
        "mean_personalization_score20",
        "mean_product_grounding_score20",
        "mean_terminology_clarity_score20",
        "mean_compliance_score20",
        "mean_understanding_gain_score20",
        "mean_misinterpretation_control_score20",
        "mean_overall_score20",
    ]
    heat = summary.set_index("user_type")[score_cols].copy()
    heat.columns = [
        "Personalization",
        "Product Grounding",
        "Terminology Clarity",
        "Compliance",
        "Understanding Gain",
        "Misinterpretation Control",
        "Overall",
    ]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    sns.heatmap(heat, annot=True, fmt=".1f", cmap="YlGnBu", vmin=0, vmax=20, cbar_kws={"label": "score (0~20)"}, ax=ax)
    ax.set_title("유형별 LLM 직접 채점 지표 (0~20)")
    ax.set_xlabel("metric")
    ax.set_ylabel("user_type")
    fig.tight_layout()
    p5 = out_dir / "05_metric_score20_heatmap.png"
    fig.savefig(p5, dpi=180)
    plt.close(fig)
    figs.append(str(p5))
    return figs


def write_report(
    out_dir: Path,
    font_name: str,
    args: argparse.Namespace,
    summary: pd.DataFrame,
    figs: List[str],
    detail: pd.DataFrame,
) -> None:
    lines: List[str] = []
    lines.append("# Expanded LLM Evaluate Report By User Type")
    lines.append("")
    lines.append(f"- font: `{font_name}`")
    lines.append(f"- per_type: `{args.per_type}`")
    lines.append(f"- top_k: `{args.top_k}`")
    lines.append(f"- expected_call_count: `{args.per_type * 5 + 1}` (5 user types + sample)")
    lines.append("- explanation fallback policy: `disabled` (template_fallback 폐기)")
    lines.append("- scoring policy: `LLM direct 0~20 float scoring`")
    lines.append("")
    lines.append("## Metric Definition (0~20)")
    lines.append("- Personalization: 사용자 특성 반영 정도")
    lines.append("- Product Grounding: 상품 사실 기반 설명 정도")
    lines.append("- Terminology Clarity: 초보자 기준 용어/문장 명료성")
    lines.append("- Compliance: 금지표현/유의사항 준수")
    lines.append("- Understanding Gain: 설명 전후 이해도 향상 체감")
    lines.append("- Misinterpretation Control: 오해 유발 억제 수준")
    lines.append("")
    lines.append("## Summary")
    lines.append("```")
    lines.append(summary.to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("## Metric Reasoning Sample (by user_type)")
    for utype, g in detail.groupby("user_type"):
        row = g.iloc[0]
        lines.append(f"### {utype}")
        lines.append(f"- personalization: {row.get('reasoning_personalization', '')}")
        lines.append(f"- product_grounding: {row.get('reasoning_product_grounding', '')}")
        lines.append(f"- terminology_clarity: {row.get('reasoning_terminology_clarity', '')}")
        lines.append(f"- compliance: {row.get('reasoning_compliance', '')}")
        lines.append(f"- understanding_gain: {row.get('reasoning_understanding_gain', '')}")
        lines.append(f"- misinterpretation_control: {row.get('reasoning_misinterpretation', '')}")
    lines.append("")
    lines.append("## Figures")
    for f in figs:
        lines.append(f"- `{f}`")
    lines.append("")
    lines.append("## Files")
    lines.append(f"- `{out_dir / 'llm_eval_expanded_detail.csv'}`")
    lines.append(f"- `{out_dir / 'llm_eval_expanded_summary.csv'}`")
    lines.append("- detail CSV includes 6 metric scores(0~20) + overall + per-metric reasoning text")
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _score_payload_from_item(user_row: pd.Series, rec_item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "user_profile_detail": (rec_item.get("explanation_object", {}) or {}).get("user_profile_detail", {}),
        "user_summary": (rec_item.get("explanation_object", {}) or {}).get("user_summary", {}),
        "recommended_product_detail": (rec_item.get("explanation_object", {}) or {}).get("recommended_product_detail", {}),
        "recommended_product": (rec_item.get("explanation_object", {}) or {}).get("recommended_product", {}),
        "reason_signals": (rec_item.get("explanation_object", {}) or {}).get("reason_signals", []),
        "comparison": (rec_item.get("explanation_object", {}) or {}).get("comparison", {}),
        "warnings": (rec_item.get("explanation_object", {}) or {}).get("warnings", []),
        "rendered_explanation": str(rec_item.get("rendered_explanation", "") or ""),
        "verification": rec_item.get("verification", {}) or {},
        "render_source": str(rec_item.get("render_source", "unknown")),
        "user_type": str(user_row.get("user_type", "")),
        "user_id": str(user_row.get("CUST_ID", "")),
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    font_name = set_korean_font()

    cfg = RecommenderConfig(data_root=args.data_root, recommender_family="all")
    rec = ThinFilerRecommender(cfg)
    snapshots = rec.build_user_snapshots(sample_users=args.sample_users_train)
    rec.load_products()
    rec.fit(snapshots=snapshots, max_users=args.max_train_users)

    raw_df = pd.read_csv(args.profile_csv)
    if not {"s_trust", "s_activity", "s_potential", "tps_score"}.issubset(raw_df.columns):
        raw_df = compute_tps_scores(raw_df)
    parsed_df = parse_custom_user_frame(raw_df)
    parsed_df["CUST_ID"] = parsed_df["CUST_ID"].astype(str)
    raw_df["CUST_ID"] = raw_df["CUST_ID"].astype(str)
    raw_df["user_type"] = classify_user_type(parsed_df)

    # sample user + per-type users
    sample_uid = str(raw_df.sort_values("tps_score", ascending=False).iloc[0]["CUST_ID"])
    sample_rows: List[pd.DataFrame] = []
    for utype, g in raw_df.groupby("user_type"):
        n = min(int(args.per_type), len(g))
        s = g.sample(n=n, random_state=42).copy()
        s["user_type"] = utype
        sample_rows.append(s)
    eval_users = pd.concat(sample_rows, ignore_index=True)
    if sample_uid not in set(eval_users["CUST_ID"].astype(str)):
        sample_u = raw_df[raw_df["CUST_ID"].astype(str).eq(sample_uid)].head(1).copy()
        sample_u["user_type"] = "샘플"
        eval_users = pd.concat([eval_users, sample_u], ignore_index=True)
    else:
        eval_users = eval_users.copy()
        eval_users.loc[eval_users["CUST_ID"].astype(str).eq(sample_uid), "user_type"] = "샘플"

    renderer = OpenAILLMRenderer(model=args.llm_model, timeout_seconds=args.score_timeout, max_retries=args.score_max_retries)
    scorer = LLMMetricScorer(model=args.llm_score_model, timeout_seconds=args.score_timeout, max_retries=args.score_max_retries)

    detail_rows: List[Dict[str, Any]] = []
    total_users = len(eval_users)
    for i, (_, urow) in enumerate(eval_users.iterrows(), start=1):
        uid = str(urow["CUST_ID"])
        utype = str(urow["user_type"])
        parsed = parsed_df[parsed_df["CUST_ID"].astype(str).eq(uid)].iloc[0]

        t0 = time.time()
        # fallback 폐기: always False
        out = rec.explain_recommendation_with(
            parsed,
            k=args.top_k,
            llm_renderer=renderer,
            fallback_to_template_on_verify_fail=False,
            use_explainer_moe=bool(args.use_explainer_moe),
            compliance_rules_path=args.compliance_rules_path,
            explainer_moe_debug=bool(args.explainer_moe_debug),
        )
        dt = float(time.time() - t0)
        print(f"[{i}/{total_users}] user={uid} type={utype} latency={dt:.2f}s", flush=True)

        for rank, rec_item in enumerate(out.get("recommendations", []), start=1):
            ver = rec_item.get("verification", {}) or {}
            inter = rec_item.get("llm_intermediate")
            llm_ver = (inter or {}).get("llm_verification", {}) if isinstance(inter, dict) else {}
            render_source = str(rec_item.get("render_source", "unknown"))
            llm_passed = int(render_source == "llm")
            if inter is not None:
                llm_passed = int(bool(llm_ver.get("passed", False)))

            score_payload = _score_payload_from_item(urow, rec_item)
            scored, raw_llm = scorer.score(score_payload)
            scored_ok = int(bool(scored))

            row: Dict[str, Any] = {
                "user_id": uid,
                "user_type": utype,
                "rank": int(rank),
                "product_id": str(rec_item.get("product_id", "")),
                "render_source": render_source,
                "llm_attempted": 1,
                "llm_passed": llm_passed,
                "fallback_used": int(render_source == "template_fallback"),
                "final_passed": int(bool(ver.get("passed", False))),
                "reason_alignment": float(ver.get("reason_alignment", 0.0)),
                "fact_consistency": int(bool(ver.get("fact_consistency", False))),
                "hallucination_rate": float(ver.get("hallucination_rate", 0.0)),
                "forbidden_claim_cnt": int(len(ver.get("forbidden_claims", []) or [])),
                "latency_sec": dt,
                "llm_metric_scored": scored_ok,
                "llm_metric_raw_response": str(raw_llm),
            }

            if scored_ok:
                row.update(scored)
            else:
                for c in METRIC_SCORE_COLS:
                    row[c] = float("nan")
                row.update(
                    {
                        "reasoning_personalization": "",
                        "reasoning_product_grounding": "",
                        "reasoning_terminology_clarity": "",
                        "reasoning_compliance": "",
                        "reasoning_understanding_gain": "",
                        "reasoning_misinterpretation": "",
                    }
                )

            detail_rows.append(row)

    detail = pd.DataFrame(detail_rows)
    summary = summarize_rows(detail)
    detail.to_csv(args.out_dir / "llm_eval_expanded_detail.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.out_dir / "llm_eval_expanded_summary.csv", index=False, encoding="utf-8-sig")
    figs = plot_figures(detail, summary, args.out_dir / "figures")
    write_report(args.out_dir, font_name, args, summary, figs, detail)

    print(f"saved: {args.out_dir / 'report.md'}")
    print(f"saved: {args.out_dir / 'llm_eval_expanded_detail.csv'}")
    print(f"saved: {args.out_dir / 'llm_eval_expanded_summary.csv'}")


if __name__ == "__main__":
    main()
