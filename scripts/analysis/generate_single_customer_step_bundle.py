#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from common.config import RecommenderConfig
from common.helpers import _ndcg_at_k
from evaluate.explainer_understanding_eval import ExplainerUnderstandingEvaluator
from explainer.llm_renderer import OpenAILLMRenderer
from recommender.engine import ThinFilerRecommender
from user_parser.tps import parse_custom_user_frame


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a single-sample end-to-end bundle with step-by-step outputs"
    )
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--family", choices=["all", "deposit", "fund"], default="all")
    p.add_argument("--sample-users-train", type=int, default=600)
    p.add_argument("--max-train-users", type=int, default=450)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--user-id", type=str, default=None)
    p.add_argument("--as-of-date", type=str, default=None)
    p.add_argument("--custom-user-xlsx", type=Path, default=None, help="Custom user xlsx path (e.g., data/thin_filer/cluster0_sample_user.xlsx)")
    p.add_argument("--custom-user-row", type=int, default=0, help="Row index in custom user xlsx")
    p.add_argument("--use-llm", action="store_true")
    p.add_argument("--llm-model", type=str, default="gpt-5-mini")
    p.add_argument("--llm-prompt-path", type=Path, default=Path("src/explainer/explain.txt"))
    p.add_argument("--llm-timeout-seconds", type=float, default=30.0)
    p.add_argument("--llm-max-retries", type=int, default=1)
    p.add_argument("--no-template-fallback", action="store_true")
    p.add_argument("--use-explainer-moe", action="store_true")
    p.add_argument("--explainer-moe-debug", action="store_true")
    p.add_argument(
        "--compliance-rules-path",
        type=Path,
        default=Path("src/explainer/compliance_rules.txt"),
        help="Text file path for external compliance rules (금융소비자보호법 문항 등)",
    )
    p.add_argument(
        "--tuned-utility-json",
        type=Path,
        default=None,
        help="Optional tuned utility json (e.g., reports/raw/utility_tuning_deposit.json)",
    )
    p.add_argument("--out-dir", type=Path, default=Path("reports/e2e/single_customer_bundle"))
    return p.parse_args()


def _jsonable(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if pd.isna(v) if not isinstance(v, (dict, list, str, bool)) else False:
        return None
    return v


def _safe_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(values))


def _to_dict_series(s: pd.Series) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in s.to_dict().items():
        out[str(k)] = _jsonable(v)
    return out


def _clean_text(v: Any) -> str:
    if v is None:
        return ""
    text = str(v).replace("\n", " ").replace("\r", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _nonempty(*vals: Any) -> str:
    for v in vals:
        t = _clean_text(v)
        if t and t.lower() not in {"none", "nan", "null"}:
            return t
    return ""


def _family_ko(family: str) -> str:
    return {"deposit": "예금·적금", "fund": "공모펀드"}.get(family, family)


def _is_true_like(v: Any) -> bool:
    s = _clean_text(v).upper()
    return s in {"Y", "YES", "TRUE", "1"}


def _strip_channel_codes(s: str) -> str:
    if not s:
        return ""
    return _clean_text(re.sub(r"(^|,\s*)\d+:", r"\1", s))


def _build_fit_case(family: str, raw: Dict[str, Any], reasons: List[str]) -> str:
    reasons = [_clean_text(r) for r in reasons if _clean_text(r)]
    raw_way = _clean_text(raw.get("예금입출금방식"))
    if reasons:
        return reasons[0]
    if family == "deposit":
        if "적립식" in raw_way:
            return "매월 꾸준히 저축해서 목돈을 만들고 싶은 경우"
        if "거치식" in raw_way:
            return "목돈을 한 번에 예치해 만기까지 운용하고 싶은 경우"
        return "중기 만기형 상품으로 자금을 비교적 안정적으로 운용하고 싶은 경우"
    return "원금 변동 가능성을 감수하고 수익 기회를 함께 보려는 경우"


def _build_key_feature_text(family: str, raw: Dict[str, Any], norm: Dict[str, Any]) -> str:
    parts: List[str] = []
    if family == "deposit":
        way = _clean_text(raw.get("예금입출금방식"))
        term_min = _clean_text(raw.get("계약기간개월수_최소구간"))
        term_max = _clean_text(raw.get("계약기간개월수_최대구간"))
        base_rate = _clean_text(raw.get("기본금리"))
        bonus_rate = _clean_text(raw.get("최대우대금리"))
        if way:
            parts.append(way)
        if term_min or term_max:
            parts.append(f"기간 {term_min}~{term_max}".rstrip("~"))
        if base_rate:
            parts.append(f"기본금리 {base_rate}%")
        if bonus_rate and bonus_rate not in {"0", "0.0"}:
            parts.append(f"최대 우대금리 {bonus_rate}%")
        if _is_true_like(raw.get("예금자보호대상여부")):
            parts.append("예금자보호 대상")
    else:
        big_type = _clean_text(raw.get("펀드유형_대유형"))
        risk = _clean_text(raw.get("펀드성과정보_투자위험등급"))
        ret_1y = _clean_text(raw.get("펀드성과정보_1년"))
        fee = _clean_text(raw.get("펀드비용정보_총보수"))
        if big_type:
            parts.append(f"대유형 {big_type}")
        if risk:
            parts.append(f"위험등급 {risk}")
        if ret_1y:
            parts.append(f"최근 1년 수익률 {ret_1y}")
        if fee:
            parts.append(f"총보수 {fee}")
        if _is_true_like(norm.get("principal_variation")):
            parts.append("원금 변동 가능")
    return ", ".join(parts[:5])


def _build_check_point_text(
    family: str,
    raw: Dict[str, Any],
    warnings: List[str],
) -> str:
    out: List[str] = []
    if family == "deposit":
        target = _nonempty(raw.get("가입대상고객_조건"), raw.get("가입대상"))
        limit = _nonempty(raw.get("가입제한_조건"), raw.get("기타_상품가입_고려사항"))
        min_amt = _clean_text(raw.get("가입금액_최소구간"))
        if target:
            out.append(f"대상 조건({target}) 확인이 필요합니다.")
        if limit:
            out.append(limit)
        if min_amt and min_amt != "제한없음":
            out.append(f"최소 가입금액({min_amt})을 확인하세요.")
        if not out:
            out.extend([_clean_text(w) for w in warnings if _clean_text(w)])
    else:
        for w in warnings:
            t = _clean_text(w)
            if t:
                out.append(t)
        if not any("과거 수익률" in x for x in out):
            out.append("과거 수익률은 미래 수익률을 보장하지 않습니다.")
        if not any("원금 변동" in x for x in out):
            out.append("원금 변동 가능성이 있습니다.")
    return " ".join(out[:2]) if out else "가입 전 조건과 위험 정보를 다시 확인하세요."


def _build_choice_guide(candidate_rows: List[Dict[str, Any]], family: str) -> List[str]:
    guides: List[str] = []
    if not candidate_rows:
        return guides
    top = candidate_rows[0]
    top_name = top.get("name", top.get("product_id", ""))
    guides.append(f"가장 무난한 후보는 \"{top_name}\"입니다.")
    if top.get("key_features"):
        guides.append(f"핵심 이유: {top['key_features']}.")

    if family == "deposit":
        install = [x for x in candidate_rows if "적립식" in x.get("key_features", "")]
        lump = [x for x in candidate_rows if "거치식" in x.get("key_features", "")]
        if install:
            guides.append(f"매월 저축형을 원하면 \"{install[0]['name']}\" 쪽이 더 맞을 수 있습니다.")
        if lump:
            guides.append(f"목돈 예치형을 원하면 \"{lump[0]['name']}\"도 함께 비교해 보세요.")
    else:
        high_risk = [x for x in candidate_rows if "위험등급" in x.get("key_features", "")]
        if high_risk:
            guides.append("펀드는 변동성이 있으므로 투자기간과 손실 감내 수준을 먼저 점검하는 것이 좋습니다.")

    return guides[:4]


def _build_common_cautions(family: str) -> List[str]:
    if family == "fund":
        return [
            "펀드는 시장 상황에 따라 원금이 변동될 수 있습니다.",
            "과거 수익률은 미래 수익률을 보장하지 않습니다.",
            "수수료(운용보수/판매보수)와 환매 가능 시점을 함께 확인해야 합니다.",
            "추천 순위보다 투자기간·위험수용도 적합성이 더 중요합니다.",
        ]
    return [
        "추천된 상품은 예금·적금 계열이라 일반적으로 펀드 대비 원금 변동 부담이 낮은 편입니다.",
        "만기 전 해지 시 기대한 이자나 우대 혜택을 받지 못할 수 있습니다.",
        "우대금리는 조건 충족 시에만 적용될 수 있어 최대 금리가 항상 적용되지는 않습니다.",
        "가입 대상·기간·최소 가입금액 같은 조건은 실제 가입 전에 반드시 확인해야 합니다.",
    ]


def _build_glossary(family: str) -> List[str]:
    base = [
        "적립식: 매월 또는 정해진 방식으로 돈을 넣어 목돈을 만드는 방식입니다.",
        "거치식: 처음에 목돈을 한 번에 넣고 만기까지 맡기는 방식입니다.",
        "만기: 약속한 저축 또는 투자 기간이 끝나는 시점입니다.",
        "단리: 이자가 원금에만 붙는 방식입니다.",
        "복리: 이자에 다시 이자가 붙는 방식입니다.",
    ]
    if family == "fund":
        base.extend(
            [
                "원금 변동: 시장 상황에 따라 원금이 오르거나 내릴 수 있다는 뜻입니다.",
                "최대낙폭: 일정 기간 중 고점 대비 가장 크게 하락한 폭입니다.",
                "운용보수/판매보수: 펀드 운용·판매 과정에서 발생하는 비용입니다.",
            ]
        )
    else:
        base.append("예금자보호: 제도상 정해진 한도 내에서 예금을 보호받을 수 있다는 뜻입니다.")
    return base[:6]


def build_sampleout_markdown(
    family: str,
    customer_info: Dict[str, Any],
    top5_raw: List[Dict[str, Any]],
    explain_output: Dict[str, Any],
) -> str:
    recommendations = explain_output.get("recommendations", []) if isinstance(explain_output, dict) else []
    exp_by_pid = {str(x.get("product_id", "")): x for x in recommendations}
    user_summary = {}
    if recommendations:
        user_summary = (recommendations[0].get("explanation_object", {}) or {}).get("user_summary", {}) or {}

    cand_rows: List[Dict[str, Any]] = []
    scores: List[float] = []
    for item in top5_raw:
        pid = str(item.get("product_id", ""))
        rank = int(item.get("rank", 0))
        score = float(item.get("score", 0.0))
        scores.append(score)
        raw = item.get("raw_product_source", {}) or {}
        norm = item.get("normalized_product", {}) or {}
        exp = exp_by_pid.get(pid, {})
        exp_obj = exp.get("explanation_object", {}) if isinstance(exp, dict) else {}
        reasons = [str(x) for x in (exp_obj.get("model_reasons", []) or [])]
        warnings = [str(x) for x in (exp_obj.get("warnings", []) or [])]

        name = _nonempty(raw.get("상품명"), raw.get("펀드명"), norm.get("product_name"), pid)
        fit_case = _build_fit_case(family, raw, reasons)
        key_features = _build_key_feature_text(family, raw, norm)
        check_point = _build_check_point_text(family, raw, warnings)

        cand_rows.append(
            {
                "rank": rank,
                "product_id": pid,
                "name": name,
                "fit_case": fit_case,
                "key_features": key_features,
                "check_point": check_point,
            }
        )

    score_spread = (max(scores) - min(scores)) if scores else 0.0
    same_score_note = "점수 차이가 매우 작아 상위권 후보로 함께 분류된 상품들입니다." if score_spread < 1e-6 else "점수 차이가 크지 않아 순위보다는 조건 적합성을 함께 비교하는 것이 좋습니다."

    family_text = _family_ko(family)
    liquidity_need = _nonempty(user_summary.get("liquidity_need"), "보통")
    risk_pref = _nonempty(user_summary.get("risk_preference"), "보통")

    lines: List[str] = []
    lines.append("[추천 전체 요약]")
    lines.append("")
    lines.append(f"고객님께는 \"{family_text} 중심의 중기 운용 후보\"가 우선 추천되었습니다.")
    lines.append("")
    lines.append(f"고객님의 유동성 필요는 {liquidity_need} 수준이고, 위험 선호는 {risk_pref} 수준으로 반영되었습니다.")
    lines.append(same_score_note)
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("[추천 후보 비교]")
    lines.append("")
    for row in cand_rows:
        lines.append(f"{row['rank']}. {row['name']}")
        lines.append(f"- 적합한 경우: {row['fit_case']}")
        if row["key_features"]:
            lines.append(f"- 핵심 특징: {row['key_features']}")
        lines.append(f"- 확인할 점: {row['check_point']}")
        lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("[고객님 기준 선택 가이드]")
    lines.append("")
    for g in _build_choice_guide(cand_rows, family):
        lines.append(g)
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("[꼭 알아둘 점]")
    lines.append("")
    for c in _build_common_cautions(family):
        lines.append(f"- {c}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("[쉬운 용어 풀이]")
    lines.append("")
    for g in _build_glossary(family):
        lines.append(f"- {g}")
    lines.append("")
    lines.append("---")
    lines.append("")
    if family == "fund":
        lines.append("[예금·적금 대안과의 차이]")
        lines.append("")
        lines.append("예금·적금은 일반적으로 원금 변동 부담이 낮지만 수익 기회는 상대적으로 제한될 수 있습니다.")
        lines.append("펀드는 수익 기회가 더 클 수 있지만 변동성과 손실 가능성을 함께 감수해야 합니다.")
    else:
        lines.append("[펀드 대안과의 차이]")
        lines.append("")
        lines.append("펀드는 예금·적금보다 수익 기회가 클 수 있지만 시장 상황에 따라 원금이 변동될 수 있습니다.")
        lines.append("원금 안정성을 더 우선하면 현재 후보군이 더 적합하고, 수익 기회를 더 원하면 펀드형 상품을 별도로 비교하는 것이 좋습니다.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("[한줄 정리]")
    lines.append("")
    if cand_rows:
        lines.append(f"고객님께는 {cand_rows[0]['name']}을 포함한 상위 후보를 기간·가입조건·채널 편의성 기준으로 비교해 선택하는 접근이 가장 현실적입니다.")
    else:
        lines.append("고객님 프로파일에 맞는 후보를 조건 중심으로 비교해 선택하는 것이 좋습니다.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = RecommenderConfig(data_root=args.data_root, recommender_family=args.family)
    rec = ThinFilerRecommender(cfg)

    as_of_dates = [args.as_of_date] if args.as_of_date else None
    snapshots = rec.build_user_snapshots(as_of_dates=as_of_dates, sample_users=args.sample_users_train)
    rec.load_products()
    tuned_mode = args.tuned_utility_json is not None

    if tuned_mode:
        from cli.improve_recommender_with_utility import (
            UtilityParams,
            apply_utility_features_and_labels,
            build_base_dataset,
            load_item_priors,
        )

        try:
            from lightgbm import LGBMRanker
        except Exception as exc:
            raise ImportError("lightgbm is required for tuned utility mode") from exc

        tuned_obj = json.loads(args.tuned_utility_json.read_text(encoding="utf-8"))
        utility_params = UtilityParams(**tuned_obj["utility_params"])
        priors = load_item_priors(
            Path("data/processed/product12_deposit_utility_index.csv"),
            Path("data/processed/product12_fund_utility_index.csv"),
            family=args.family,
        )

        train_base, train_group = build_base_dataset(
            rec,
            snapshots,
            priors,
            rec.config.candidate_max,
            args.max_train_users,
        )
        train_data = apply_utility_features_and_labels(train_base.copy(), rec, utility_params)
        feature_cols = [
            "risk_match", "liquidity_match", "horizon_match", "complexity_match", "amount_feasibility",
            "family_match", "digital_match", "risk_level", "liquidity_level", "complexity", "min_amount_bin",
            "principal_variation", "max_rate", "risk_tol", "liquidity_need", "complexity_tol", "amount_bin",
            "investment_possible", "credit_depth", "credit_recency", "telecom_payment_consistency",
            "card_usage_stability", "spending_vs_balance_ratio", "digital_behavior_freq",
            "item_utility_prior", "realizability", "rate_factor", "hybrid_utility_score",
        ]
        feature_cols = [c for c in feature_cols if c in train_data.columns]

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
        rec.model = model
        rec.feature_columns = feature_cols

        prior_cols = [
            "product_id",
            "item_utility_prior",
            "U_rate",
            "U_bonus",
            "U_feasibility",
            "U_liquidity",
            "U_return",
            "U_risk_eff",
            "U_cost_eff",
            "U_simplicity",
        ]
        prior_cols = [c for c in prior_cols if c in priors.columns]
        prior_view = priors[prior_cols].copy()
        original_add_pair_features = rec._add_pair_features

        def _add_pair_features_tuned(users: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
            pair = original_add_pair_features(users, items)
            pair["product_id"] = pair["product_id"].astype(str)
            pair = pair.merge(prior_view, on="product_id", how="left")
            pair = apply_utility_features_and_labels(pair, rec, utility_params)
            return pair

        rec._add_pair_features = _add_pair_features_tuned  # type: ignore[assignment]
    else:
        rec.fit(snapshots=snapshots, max_users=args.max_train_users)

    if args.custom_user_xlsx is not None:
        cdf = pd.read_excel(args.custom_user_xlsx)
        if cdf.empty:
            raise ValueError(f"custom user file has no rows: {args.custom_user_xlsx}")
        ridx = int(args.custom_user_row)
        if ridx < 0 or ridx >= len(cdf):
            raise ValueError(f"custom-user-row out of range: {ridx} (rows={len(cdf)})")
        parsed = parse_custom_user_frame(cdf.iloc[[ridx]].copy())
        user_row = parsed.iloc[0]
    elif args.user_id:
        chosen = snapshots[snapshots[cfg.user_key_11].astype(str).eq(str(args.user_id))]
        if chosen.empty:
            raise ValueError(f"user_id not found in sampled snapshots: {args.user_id}")
        user_row = chosen.iloc[0]
    else:
        order_col = "tps_score" if "tps_score" in snapshots.columns else cfg.user_key_11
        user_row = snapshots.sort_values(order_col, ascending=False).iloc[0]

    llm_renderer = None
    if args.use_llm:
        llm_renderer = OpenAILLMRenderer(
            model=args.llm_model,
            prompt_path=args.llm_prompt_path,
            timeout_seconds=float(args.llm_timeout_seconds),
            max_retries=int(args.llm_max_retries),
        )

    recommend_output = rec.recommend(user_row, k=args.top_k)
    explain_output = rec.explain_recommendation_with(
        user_row,
        k=args.top_k,
        llm_renderer=llm_renderer,
        fallback_to_template_on_verify_fail=not args.no_template_fallback,
        use_explainer_moe=bool(args.use_explainer_moe),
        compliance_rules_path=args.compliance_rules_path,
        explainer_moe_debug=bool(args.explainer_moe_debug),
    )

    # Step 1: customer info
    first_exp_obj = {}
    if explain_output.get("recommendations"):
        first_exp_obj = (explain_output["recommendations"][0] or {}).get("explanation_object", {}) or {}

    customer_info = {
        "user_id": str(user_row.get(cfg.user_key_11, "")),
        "as_of_date": str(user_row.get("as_of_date", "")),
        "anchor_ym": _jsonable(user_row.get("anchor_ym")),
        "cb_join_found": _jsonable(user_row.get("cb_join_found")),
        "user_profile_detail": first_exp_obj.get("user_profile_detail", {}),
        "user_source_data": first_exp_obj.get("user_source_data", {}),
        "raw_snapshot_row": _to_dict_series(user_row),
    }

    # Step 2: top5 raw product data
    top5_raw: List[Dict[str, Any]] = []
    product_df = rec.products.copy() if rec.products is not None else pd.DataFrame()
    for rank, rec_item in enumerate(recommend_output.get("recommendations", []), start=1):
        pid = str(rec_item.get("product_id", ""))
        src = rec.product_source_lookup.get(pid, {})
        norm = {}
        if not product_df.empty:
            row = product_df[product_df["product_id"].astype(str).eq(pid)]
            if not row.empty:
                norm = _to_dict_series(row.iloc[0])
        top5_raw.append(
            {
                "rank": rank,
                "product_id": pid,
                "score": float(rec_item.get("score", 0.0)),
                "normalized_product": norm,
                "raw_product_source": src,
            }
        )

    # Step 3: recommender object
    recommender_object = {
        "recommend_output": recommend_output,
        "explain_output": explain_output,
    }

    # Step 4: llm outputs only
    llm_outputs: List[Dict[str, Any]] = []
    for rank, item in enumerate(explain_output.get("recommendations", []), start=1):
        llm_outputs.append(
            {
                "rank": rank,
                "product_id": str(item.get("product_id", "")),
                "render_source": str(item.get("render_source", "")),
                "rendered_explanation": str(item.get("rendered_explanation", "")),
                "llm_intermediate": item.get("llm_intermediate"),
            }
        )

    # Step 5: evaluation
    candidates = rec.generate_candidates(user_row)
    pair = rec._add_pair_features(pd.DataFrame([user_row]), candidates)
    labels = rec._build_labels(pair).to_numpy(dtype=float)
    baseline_scores = pair["baseline_score"].to_numpy(dtype=float)
    model_scores = rec.model.predict(pair[rec.feature_columns].fillna(0.0)) if (rec.model is not None and rec.feature_columns) else baseline_scores

    ranking_eval = {
        "candidate_count": int(len(pair)),
        "baseline_ndcg@5": float(_ndcg_at_k(labels, baseline_scores, 5)),
        "model_ndcg@5": float(_ndcg_at_k(labels, model_scores, 5)),
        "baseline_ndcg@10": float(_ndcg_at_k(labels, baseline_scores, 10)),
        "model_ndcg@10": float(_ndcg_at_k(labels, model_scores, 10)),
    }

    verifier_rows = [r.get("verification", {}) for r in explain_output.get("recommendations", [])]
    verifier_eval = {
        "topk_count": int(len(verifier_rows)),
        "pass_rate": _safe_mean([1.0 if bool(v.get("passed", False)) else 0.0 for v in verifier_rows]),
        "mean_reason_alignment": _safe_mean([float(v.get("reason_alignment", 0.0)) for v in verifier_rows]),
        "mean_hallucination_rate": _safe_mean([float(v.get("hallucination_rate", 0.0)) for v in verifier_rows]),
        "fact_consistency_rate": _safe_mean([1.0 if bool(v.get("fact_consistency", False)) else 0.0 for v in verifier_rows]),
        "forbidden_claim_avg_count": _safe_mean([float(len(v.get("forbidden_claims", []) or [])) for v in verifier_rows]),
    }

    ue = ExplainerUnderstandingEvaluator()
    understanding_records = [
        ue.evaluate_recommendation(str(customer_info["user_id"]), item)
        for item in explain_output.get("recommendations", [])
    ]
    understanding_summary = ue.summarize(understanding_records)

    evaluation_result = {
        "ranking_evaluation": ranking_eval,
        "verifier_evaluation": verifier_eval,
        "understanding_evaluation_summary": understanding_summary,
        "understanding_evaluation_records": understanding_records,
    }

    step_files = {
        "01_customer_info": out_dir / "01_customer_info.json",
        "02_top5_product_raw": out_dir / "02_top5_product_raw.json",
        "03_recommender_object": out_dir / "03_recommender_object.json",
        "04_llm_output": out_dir / "04_llm_output.json",
        "05_evaluation_result": out_dir / "05_evaluation_result.json",
    }

    step_files["01_customer_info"].write_text(json.dumps(customer_info, ensure_ascii=False, indent=2), encoding="utf-8")
    step_files["02_top5_product_raw"].write_text(json.dumps(top5_raw, ensure_ascii=False, indent=2), encoding="utf-8")
    step_files["03_recommender_object"].write_text(json.dumps(recommender_object, ensure_ascii=False, indent=2), encoding="utf-8")
    step_files["04_llm_output"].write_text(json.dumps(llm_outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    step_files["05_evaluation_result"].write_text(json.dumps(evaluation_result, ensure_ascii=False, indent=2), encoding="utf-8")

    sampleout_md = build_sampleout_markdown(
        family=args.family if args.family in {"deposit", "fund"} else str(top5_raw[0].get("normalized_product", {}).get("product_family", "deposit") if top5_raw else "deposit"),
        customer_info=customer_info,
        top5_raw=top5_raw,
        explain_output=explain_output,
    )
    (out_dir / "sampleout.md").write_text(sampleout_md, encoding="utf-8")

    report_lines = [
        "# Single Sample Customer E2E Bundle",
        "",
        f"- generated_at: `{ts}`",
        f"- out_dir: `{out_dir}`",
        f"- user_id: `{customer_info['user_id']}`",
        f"- as_of_date: `{customer_info['as_of_date']}`",
        f"- family_mode: `{args.family}`",
        f"- tuned_utility_mode: `{bool(tuned_mode)}`",
        f"- tuned_utility_json: `{args.tuned_utility_json}`",
        f"- use_llm: `{bool(args.use_llm)}`",
        f"- llm_timeout_seconds: `{float(args.llm_timeout_seconds)}`",
        f"- llm_max_retries: `{int(args.llm_max_retries)}`",
        f"- custom_user_xlsx: `{args.custom_user_xlsx}`",
        f"- custom_user_row: `{int(args.custom_user_row)}`",
        "",
        "## Step Files",
        f"- 1) customer info: `{step_files['01_customer_info']}`",
        f"- 2) top5 raw products: `{step_files['02_top5_product_raw']}`",
        f"- 3) recommender object: `{step_files['03_recommender_object']}`",
        f"- 4) llm output: `{step_files['04_llm_output']}`",
        f"- 5) evaluation result: `{step_files['05_evaluation_result']}`",
        f"- 6) chat-style sample out: `{out_dir / 'sampleout.md'}`",
        "",
        "## Quick Evaluation Snapshot",
        f"- candidate_count: {ranking_eval['candidate_count']}",
        f"- baseline_ndcg@5: {ranking_eval['baseline_ndcg@5']:.4f}",
        f"- model_ndcg@5: {ranking_eval['model_ndcg@5']:.4f}",
        f"- verifier_pass_rate(topk): {verifier_eval['pass_rate']:.4f}",
        f"- understanding_gain(mean): {understanding_summary.get('understanding_gain', 0.0):.4f}",
        f"- misinterpretation_rate(mean): {understanding_summary.get('misinterpretation_rate', 0.0):.4f}",
    ]
    (out_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"saved: {out_dir / 'report.md'}")
    for k, v in step_files.items():
        print(f"saved: {k} -> {v}")


if __name__ == "__main__":
    main()
