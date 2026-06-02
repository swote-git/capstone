#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .runtime_config import parse_args_with_config
from common.config import RecommenderConfig
from common.env import load_dotenv_file
from evaluate.explainer_eval import evaluate_explainer_batch, sample_users
from evaluate.explainer_understanding_eval import ExplainerUnderstandingEvaluator
from explainer.llm_renderer import OpenAILLMRenderer
from explainer.moe_orchestrator import ExplainerMoEOrchestrator
from explainer.render_verify import verify
from explainer.service import GroundedExplainer
from recommender.engine import ThinFilerRecommender

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None


def _one_snapshot_per_user(df: pd.DataFrame, user_col: str) -> pd.DataFrame:
    if df.empty or user_col not in df.columns:
        return df
    work = df.copy()
    if "anchor_ym" in work.columns:
        work["__ord_anchor"] = pd.to_numeric(work["anchor_ym"], errors="coerce").fillna(-1)
    else:
        work["__ord_anchor"] = -1
    if "as_of_date" in work.columns:
        work["__ord_asof"] = work["as_of_date"].astype(str)
    else:
        work["__ord_asof"] = ""
    work = work.sort_values([user_col, "__ord_anchor", "__ord_asof"], ascending=[True, False, False])
    out = work.drop_duplicates(subset=[user_col], keep="first").drop(columns=["__ord_anchor", "__ord_asof"])
    return out.reset_index(drop=True)


def _synthetic_explanation_cases() -> List[Dict[str, Any]]:
    """Privacy-safe synthetic cases (no workspace customer/product data)."""
    return [
        {
            "product_id": "SYN-DEP-001",
            "explanation_object": {
                "user_summary": {"risk_preference": "낮음", "liquidity_need": "보통", "financial_knowledge": "낮음"},
                "recommended_product": {"product_id": "SYN-DEP-001", "product_name": "안정형 정기예금", "family": "deposit", "risk": "낮음", "liquidity": "보통"},
                "recommended_product_detail": {"horizon": "중기", "complexity": "낮음", "principal_variation": False, "product_meta": {"max_rate": 3.6}},
                "model_reasons": [
                    "이 상품은 고객님의 위험 선호(낮음)와 비교적 잘 맞습니다.",
                    "유동성 필요가 보통 수준인 점을 고려하면 만기형 구조를 수용 가능한 편입니다.",
                    "금융지식이 낮은 사용자에게도 구조가 단순한 편입니다.",
                ],
                "comparison": {"alternative": "fund", "difference": "수익 잠재력은 높지만 위험과 원금 변동 가능성이 커질 수 있습니다"},
                "warnings": ["만기 전 해지 시 기대한 이자를 받지 못할 수 있습니다."],
                "reason_signals": [{"feature": "risk_match"}, {"feature": "complexity_match"}, {"feature": "liquidity_match"}],
                "ranking_context": {"same_score_topk": False, "same_score_note": ""},
                "explanation_policy": {"caution_first": False, "hide_internal_feature_names": True, "hide_raw_long_decimals": True},
            },
        },
        {
            "product_id": "SYN-DEP-002",
            "explanation_object": {
                "user_summary": {"risk_preference": "보통", "liquidity_need": "낮음", "financial_knowledge": "보통"},
                "recommended_product": {"product_id": "SYN-DEP-002", "product_name": "적립식 정기적금", "family": "deposit", "risk": "낮음", "liquidity": "보통"},
                "recommended_product_detail": {"horizon": "중기", "complexity": "보통", "principal_variation": False, "product_meta": {"max_rate": 4.1}},
                "model_reasons": [
                    "중기 자금 운용을 선호하는 성향과 상품 기간이 비교적 잘 맞습니다.",
                    "매월 저축형 구조라 자금관리 습관 형성에 유리할 수 있습니다.",
                    "위험 수준이 낮아 변동성 부담이 큰 편은 아닙니다.",
                ],
                "comparison": {"alternative": "fund", "difference": "기대수익은 낮지만 원금 변동 부담이 상대적으로 작습니다"},
                "warnings": ["우대금리는 조건 충족 시에만 적용될 수 있습니다."],
                "reason_signals": [{"feature": "horizon_match"}, {"feature": "risk_match"}, {"feature": "amount_feasibility"}],
                "ranking_context": {"same_score_topk": True, "same_score_note": "상위권 후보로 함께 분류된 상품입니다."},
                "explanation_policy": {"caution_first": False, "hide_internal_feature_names": True, "hide_raw_long_decimals": True},
            },
        },
        {
            "product_id": "SYN-FUND-001",
            "explanation_object": {
                "user_summary": {"risk_preference": "보통", "liquidity_need": "보통", "financial_knowledge": "보통"},
                "recommended_product": {"product_id": "SYN-FUND-001", "product_name": "국내혼합형 펀드", "family": "fund", "risk": "높음", "liquidity": "보통"},
                "recommended_product_detail": {"horizon": "중기", "complexity": "보통", "principal_variation": True, "product_meta": {"fund_fee": 0.9}},
                "model_reasons": [
                    "수익 기회를 일부 확보하려는 성향과 펀드형 구조가 부분적으로 맞습니다.",
                    "유동성 필요가 보통 수준이라 중기 투자기간을 고려할 수 있습니다.",
                    "다만 위험 측면에서는 완전한 일치는 아니므로 주의가 필요합니다.",
                ],
                "comparison": {"alternative": "deposit", "difference": "원금 안정성은 높지만 수익 잠재력은 낮아질 수 있습니다"},
                "warnings": [
                    "시장 상황에 따라 원금 변동 가능성이 있습니다.",
                    "과거 수익률은 미래 수익률을 보장하지 않습니다.",
                ],
                "reason_signals": [{"feature": "risk_match"}, {"feature": "horizon_match"}, {"feature": "family_match"}],
                "ranking_context": {"same_score_topk": False, "same_score_note": ""},
                "explanation_policy": {"caution_first": True, "hide_internal_feature_names": True, "hide_raw_long_decimals": True},
            },
        },
        {
            "product_id": "SYN-FUND-002",
            "explanation_object": {
                "user_summary": {"risk_preference": "높음", "liquidity_need": "낮음", "financial_knowledge": "높음"},
                "recommended_product": {"product_id": "SYN-FUND-002", "product_name": "해외주식형 펀드", "family": "fund", "risk": "높음", "liquidity": "보통"},
                "recommended_product_detail": {"horizon": "장기", "complexity": "높음", "principal_variation": True, "product_meta": {"fund_fee": 1.2}},
                "model_reasons": [
                    "고위험 수용 성향과 고변동 상품 특성이 비교적 잘 맞습니다.",
                    "장기 운용 관점에서 수익 기회 탐색에 적합할 수 있습니다.",
                    "복잡도가 높은 상품이므로 비용 구조를 함께 확인할 필요가 있습니다.",
                ],
                "comparison": {"alternative": "deposit", "difference": "원금 안정성은 낮지만 기대수익 기회는 높을 수 있습니다"},
                "warnings": [
                    "원금 손실이 발생할 수 있습니다.",
                    "운용보수/판매보수 등 비용을 확인해야 합니다.",
                    "과거 수익률은 미래 수익률을 보장하지 않습니다.",
                ],
                "reason_signals": [{"feature": "risk_match"}, {"feature": "complexity_match"}, {"feature": "family_match"}],
                "ranking_context": {"same_score_topk": False, "same_score_note": ""},
                "explanation_policy": {"caution_first": True, "hide_internal_feature_names": True, "hide_raw_long_decimals": True},
            },
        },
    ]


def _product_facts_from_object(explanation_object: Dict[str, Any]) -> Dict[str, Any]:
    p = explanation_object.get("recommended_product", {})
    d = explanation_object.get("recommended_product_detail", {})
    return {
        "product_id": str(p.get("product_id", "")),
        "product_name": str(p.get("product_name", "")),
        "family": str(p.get("family", "")),
        "risk": str(p.get("risk", "")),
        "liquidity": str(p.get("liquidity", "")),
        "horizon": str(d.get("horizon", "")),
        "complexity": str(d.get("complexity", "")),
        "principal_variation": bool(d.get("principal_variation", False)),
        "product_meta": d.get("product_meta", {}) or {},
    }


def _mean_or_zero(xs: List[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0


def _evaluate_synthetic_batch(
    llm_renderer: OpenAILLMRenderer,
    understanding_evaluator: Optional[ExplainerUnderstandingEvaluator],
    max_understanding_samples: int,
    synthetic_repeat: int = 5,
    use_explainer_moe: bool = False,
    compliance_rules_path: Optional[Path] = None,
) -> Dict[str, Any]:
    alignments: List[float] = []
    hallucinations: List[float] = []
    fact_consistency: List[float] = []
    pass_flags: List[float] = []
    forbidden_flags: List[float] = []
    reasons_per_item: List[int] = []
    reason_feature_counts: Dict[str, int] = {}
    reason_patterns: Dict[str, int] = {}
    failed_examples: List[Dict[str, object]] = []
    understanding_records: List[Dict[str, object]] = []
    render_sources: List[str] = []
    render_texts: List[str] = []

    base_cases = _synthetic_explanation_cases()
    orchestrator = (
        ExplainerMoEOrchestrator(
            llm_renderer=llm_renderer,
            compliance_rules_path=compliance_rules_path,
            include_template_expert=True,
        )
        if bool(use_explainer_moe)
        else None
    )
    extra_forbidden_patterns = list(orchestrator.extra_forbidden_patterns) if orchestrator is not None else []
    repeat = max(1, int(synthetic_repeat))
    cases: List[Dict[str, Any]] = []
    for r in range(repeat):
        for c in base_cases:
            cc = copy.deepcopy(c)
            product_id = f"{str(c.get('product_id', 'SYN'))}-R{r + 1:02d}"
            cc["product_id"] = product_id
            eo = cc.get("explanation_object", {})
            rp = eo.get("recommended_product", {})
            if isinstance(rp, dict):
                rp["product_id"] = product_id
            cc["_synthetic_user_id"] = f"SYN_USER_{(r % 5) + 1}"
            cases.append(cc)

    user_ids_seen: set[str] = set()
    for c in cases:
        explanation_object = c["explanation_object"]
        facts = _product_facts_from_object(explanation_object)
        render_source = "llm"
        rendered = ""
        verification: Dict[str, Any] = {}

        if orchestrator is None:
            rendered = llm_renderer.render(explanation_object)
            verification = verify(
                rendered,
                explanation_object,
                facts,
                extra_forbidden_patterns=extra_forbidden_patterns,
            )
        else:
            bundle = orchestrator.build_candidates(explanation_object)
            best_score = -1e9
            for cand in bundle.get("candidates", []):
                cand_text = str(cand.get("text", "") or "")
                cand_source = str(cand.get("source", "llm"))
                cand_ver = verify(
                    cand_text,
                    explanation_object,
                    facts,
                    extra_forbidden_patterns=extra_forbidden_patterns,
                )
                s = (
                    (2.0 if bool(cand_ver.get("passed", False)) else 0.0)
                    + float(cand_ver.get("reason_alignment", 0.0))
                    + (1.0 if bool(cand_ver.get("fact_consistency", False)) else 0.0)
                    + (1.0 - float(cand_ver.get("hallucination_rate", 1.0)))
                    - (0.2 * len(cand_ver.get("forbidden_claims", []) or []))
                )
                if s > best_score:
                    best_score = float(s)
                    rendered = cand_text
                    verification = cand_ver
                    render_source = cand_source
        user_id = str(c.get("_synthetic_user_id", "SYN_USER"))
        user_ids_seen.add(user_id)

        rec_item = {
            "product_id": c["product_id"],
            "score": 0.0,
            "reason_signals": explanation_object.get("reason_signals", []),
            "product_facts": facts,
            "explanation_object": explanation_object,
            "rendered_explanation": rendered,
            "verification": verification,
            "render_source": render_source,
            "llm_intermediate": None,
        }
        render_sources.append(render_source)
        render_texts.append(str(rendered or ""))

        alignments.append(float(verification.get("reason_alignment", 0.0)))
        hallucinations.append(float(verification.get("hallucination_rate", 0.0)))
        fact_consistency.append(1.0 if bool(verification.get("fact_consistency", False)) else 0.0)
        pass_flags.append(1.0 if bool(verification.get("passed", False)) else 0.0)
        forbidden_flags.append(1.0 if len(verification.get("forbidden_claims", []) or []) > 0 else 0.0)

        rs = rec_item.get("reason_signals", []) or []
        reasons_per_item.append(len(rs))
        for s in rs:
            f = str((s or {}).get("feature", "unknown"))
            reason_feature_counts[f] = reason_feature_counts.get(f, 0) + 1
        pat = "|".join(sorted(str((s or {}).get("feature", "unknown")) for s in rs))
        reason_patterns[pat] = reason_patterns.get(pat, 0) + 1

        if not bool(verification.get("passed", False)) and len(failed_examples) < 5:
            failed_examples.append(
                {
                    "user_id": user_id,
                    "product_id": rec_item["product_id"],
                    "verification": verification,
                    "rendered_explanation": rec_item["rendered_explanation"],
                }
            )

        if understanding_evaluator is not None and len(understanding_records) < int(max_understanding_samples):
            understanding_records.append(
                understanding_evaluator.evaluate_recommendation(
                    user_id=user_id,
                    recommendation_item=rec_item,
                )
            )

    top_reason_features = sorted(reason_feature_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    understanding_summary: Dict[str, object] = {
        "enabled": bool(understanding_evaluator is not None),
        "evaluated_count": int(len(understanding_records)),
        "metrics": {},
        "sample_records": understanding_records[:5],
        "records": understanding_records,
    }
    if understanding_evaluator is not None:
        understanding_summary["metrics"] = understanding_evaluator.summarize(understanding_records)

    total_render = len(render_sources)
    llm_render = sum(1 for s in render_sources if s == "llm")
    nonempty_render = sum(1 for t in render_texts if len(str(t).strip()) > 0)

    return {
        "coverage": {
            "evaluated_users": int(len(user_ids_seen) if user_ids_seen else 1),
            "evaluated_explanations": int(len(pass_flags)),
        },
        "metrics": {
            "reason_coverage_rc": _mean_or_zero(alignments),
            "hallucination_rate_hr": _mean_or_zero(hallucinations),
            "fact_consistency_rate": _mean_or_zero(fact_consistency),
            "verification_pass_rate": _mean_or_zero(pass_flags),
            "forbidden_claim_rate": _mean_or_zero(forbidden_flags),
            "avg_reasons_per_item": _mean_or_zero([float(x) for x in reasons_per_item]),
            "reason_pattern_diversity": float(len(reason_patterns) / max(1, len(pass_flags))),
        },
        "reason_feature_distribution_top10": [
            {"feature": feat, "count": int(cnt)} for feat, cnt in top_reason_features
        ],
        "reason_pattern_distribution_top10": [
            {"pattern": p, "count": int(c)}
            for p, c in sorted(reason_patterns.items(), key=lambda x: x[1], reverse=True)[:10]
        ],
        "failed_examples": failed_examples,
        "understanding_eval": understanding_summary,
        "render_stats": {
            "total": int(total_render),
            "llm": int(llm_render),
            "template_fallback": 0,
            "template": 0,
            "nonempty": int(nonempty_render),
        },
    }


def _normalize_model_key(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def resolve_model_id(name: str) -> Tuple[str, str]:
    """Resolve user-friendly model names into API model IDs.

    Returns:
    - resolved_model_id
    - note (resolution note)
    """
    raw = str(name).strip()
    k = _normalize_model_key(raw)

    alias_map = {
        # Requested aliases
        "chatgpt54mini": "gpt-5.4-mini",
        "gpt54mini": "gpt-5.4-mini",
        "chatgpt55": "gpt-5.5",
        "gpt55": "gpt-5.5",
        "chatgpt55pro": "gpt-5.5-pro",
        "gpt55pro": "gpt-5.5-pro",
        "chatgpt54pro": "gpt-5.4-pro",
        "gpt54pro": "gpt-5.4-pro",
    }

    if k in alias_map:
        return alias_map[k], f"alias:{raw}->{alias_map[k]}"

    # If user already passed a likely API ID, keep it.
    if raw.startswith("gpt-"):
        return raw, "as_is"

    return raw, "as_is_unverified"


def _extract_json_object(raw: str) -> Dict[str, Any]:
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


def _clip20(value: Any) -> float:
    try:
        x = float(value)
    except Exception:
        return float("nan")
    if np.isnan(x) or np.isinf(x):
        return float("nan")
    return float(max(0.0, min(20.0, x)))


class _LLMQualityScorer:
    """Direct LLM grader for explanation quality dimensions (0~20 scale)."""

    def __init__(
        self,
        model: str,
        timeout_seconds: float = 30.0,
        max_retries: int = 1,
        api_key: Optional[str] = None,
    ) -> None:
        if OpenAI is None:
            raise ImportError("openai package is not installed.")
        if api_key is None and not os.getenv("OPENAI_API_KEY"):
            load_dotenv_file()
        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = OpenAI(
            api_key=resolved_api_key,
            timeout=float(timeout_seconds),
            max_retries=int(max_retries),
        )
        self.prompt = (
            "당신은 금융 추천 설명 평가자입니다.\n"
            "입력 데이터만 사용해 아래 4개 지표를 0~20 실수 점수로 채점하세요.\n"
            "- personalization_score20\n"
            "- product_grounding_score20\n"
            "- terminology_clarity_score20\n"
            "- compliance_score20\n"
            "절대 규칙: 입력에 없는 사실 추정 금지, JSON 객체 하나만 출력.\n"
            "출력 JSON 키: personalization_score20, product_grounding_score20, "
            "terminology_clarity_score20, compliance_score20"
        )

    def score(self, payload: Dict[str, Any]) -> Dict[str, float]:
        text_payload = json.dumps(payload, ensure_ascii=False)
        raw = ""
        try:
            resp = self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": [{"type": "input_text", "text": self.prompt}]},
                    {"role": "user", "content": [{"type": "input_text", "text": text_payload}]},
                ],
            )
            raw = (getattr(resp, "output_text", None) or "").strip()
            obj = _extract_json_object(raw)
            if obj:
                return {
                    "personalization_score20": _clip20(obj.get("personalization_score20")),
                    "product_grounding_score20": _clip20(obj.get("product_grounding_score20")),
                    "terminology_clarity_score20": _clip20(obj.get("terminology_clarity_score20")),
                    "compliance_score20": _clip20(obj.get("compliance_score20")),
                }
        except Exception:
            pass

        try:
            resp2 = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.prompt},
                    {"role": "user", "content": text_payload},
                ],
                temperature=0,
            )
            raw2 = (resp2.choices[0].message.content or "").strip()
            obj2 = _extract_json_object(raw2)
            if obj2:
                return {
                    "personalization_score20": _clip20(obj2.get("personalization_score20")),
                    "product_grounding_score20": _clip20(obj2.get("product_grounding_score20")),
                    "terminology_clarity_score20": _clip20(obj2.get("terminology_clarity_score20")),
                    "compliance_score20": _clip20(obj2.get("compliance_score20")),
                }
        except Exception:
            pass

        return {}


def _score_quality_by_llm(
    records: List[Dict[str, Any]],
    scorer: _LLMQualityScorer,
    max_items: int,
) -> Dict[str, float]:
    cols = [
        "personalization_score20",
        "product_grounding_score20",
        "terminology_clarity_score20",
        "compliance_score20",
    ]
    acc: Dict[str, List[float]] = {c: [] for c in cols}
    n_total = min(max(0, int(max_items)), len(records))
    n_success = 0

    for r in records[:n_total]:
        payload = {
            "recommendation_payload": r.get("recommendation_payload", {}),
            "explanation_object": r.get("explanation_object", {}),
            "rendered_explanation": r.get("rendered_explanation", ""),
            "verification": r.get("verification", {}),
        }
        out = scorer.score(payload)
        if not out:
            continue
        has_any = False
        for c in cols:
            v = out.get(c, float("nan"))
            if isinstance(v, float) and np.isfinite(v):
                acc[c].append(float(v))
                has_any = True
        if has_any:
            n_success += 1

    result: Dict[str, float] = {
        "llm_quality_eval_count": float(n_total),
        "llm_quality_eval_success_count": float(n_success),
    }
    for c in cols:
        vals = acc[c]
        result[f"llm_{c}"] = float(np.mean(vals)) if vals else float("nan")
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark multiple LLM models on grounded explainer quality/effect metrics")
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--sample-users", type=int, default=300)
    p.add_argument("--max-eval-users", type=int, default=80)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--fit", action="store_true")
    p.add_argument("--family", choices=["all", "deposit", "fund"], default="all")
    p.add_argument("--max-train-users", type=int, default=200)
    p.add_argument("--as-of-dates", nargs="*", default=None)

    p.add_argument(
        "--models",
        nargs="+",
        default=["gpt-5-mini", "gpt-5.4-mini", "gpt-5.4", "gpt-5.5"],
        help="Model list to benchmark (user-friendly names or API model IDs)",
    )
    p.add_argument("--llm-prompt-path", type=Path, default=Path("src/explainer/explain.txt"))
    p.add_argument("--no-template-fallback", action="store_true")
    p.add_argument("--use-explainer-moe", action="store_true")
    p.add_argument("--explainer-moe-debug", action="store_true")
    p.add_argument(
        "--compliance-rules-path",
        type=Path,
        default=Path("src/explainer/compliance_rules.txt"),
        help="Text file path for external compliance rules (금융소비자보호법 문항 등)",
    )
    p.add_argument("--llm-timeout-seconds", type=float, default=45.0)
    p.add_argument("--llm-max-retries", type=int, default=1)
    p.add_argument("--continue-on-error", action="store_true", help="Continue benchmarking even when a model fails")

    p.add_argument("--enable-understanding-eval", action="store_true")
    p.add_argument("--max-understanding-samples", type=int, default=120)
    p.add_argument("--use-llm-user-simulator", action="store_true")
    p.add_argument("--use-llm-evaluator", action="store_true")
    p.add_argument("--simulator-model", type=str, default="gpt-5-mini")
    p.add_argument("--evaluator-model", type=str, default="gpt-5-mini")
    p.add_argument("--simulator-prompt-path", type=Path, default=Path("src/explainer/simulator.txt"))
    p.add_argument("--evaluator-prompt-path", type=Path, default=Path("src/explainer/evaluator.txt"))
    p.add_argument("--disable-llm-quality-scoring", action="store_true")
    p.add_argument(
        "--llm-quality-model",
        type=str,
        default="auto",
        help="Model for direct LLM scoring of personalization/product_grounding/terminology_clarity/compliance. auto=same as benchmarked model",
    )
    p.add_argument("--llm-quality-timeout-seconds", type=float, default=30.0)
    p.add_argument("--llm-quality-max-retries", type=int, default=1)
    p.add_argument("--llm-quality-max-items", type=int, default=40)

    p.add_argument("--use-moe-harness", action="store_true")
    p.add_argument("--moe-debug", action="store_true")
    p.add_argument("--moe-ranker-weight", type=float, default=0.60)
    p.add_argument("--moe-baseline-weight", type=float, default=0.25)
    p.add_argument("--moe-utility-weight", type=float, default=0.15)
    p.add_argument("--moe-deposit-baseline-boost", type=float, default=0.05)
    p.add_argument("--moe-fund-utility-boost", type=float, default=0.10)
    p.add_argument("--moe-low-risk-fund-penalty", type=float, default=0.15)

    p.add_argument("--out-dir", type=Path, default=Path("reports/e2e/llm_model_benchmark"))
    p.add_argument(
        "--synthetic-safe-mode",
        action="store_true",
        help="Use only built-in synthetic explanation objects (no workspace customer/product rows).",
    )
    p.add_argument(
        "--synthetic-repeat",
        type=int,
        default=5,
        help="Repeat synthetic case bundle this many times (base 4 cases, so repeat=5 -> 20 calls/model).",
    )
    return parse_args_with_config(p, section="benchmark_llm_models")


def _metric_get(d: Dict[str, Any], key: str) -> float:
    try:
        return float(d.get(key, 0.0) or 0.0)
    except Exception:
        return 0.0


def _plot_summary(df: pd.DataFrame, out_png: Path) -> None:
    if df.empty:
        return
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        return

    x = np.arange(len(ok))
    width = 0.22

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width, ok["verification_pass_rate"], width, label="PassRate")
    ax.bar(x, ok["understanding_gain"], width, label="UG")
    ax.bar(x + width, 1.0 - ok["misinterpretation_rate"], width, label="1-MR")
    ax.set_xticks(x)
    ax.set_xticklabels(ok["model_input"], rotation=15, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("score (0~1)")
    ax.set_title("LLM Model Benchmark: Verifier/UG/MR")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def _build_model_score20_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()

    out = pd.DataFrame(index=ok.index)
    out["model"] = ok["model_input"].astype(str)

    def _score20(direct_col: str, fallback01_col: str) -> pd.Series:
        if direct_col in ok.columns and ok[direct_col].notna().any():
            return pd.to_numeric(ok[direct_col], errors="coerce").clip(0.0, 20.0)
        if fallback01_col in ok.columns:
            return (pd.to_numeric(ok[fallback01_col], errors="coerce") * 20.0).clip(0.0, 20.0)
        return pd.Series(np.nan, index=ok.index)

    out["Personalization"] = _score20("llm_personalization_score20", "personalization")
    out["Product Grounding"] = _score20("llm_product_grounding_score20", "product_grounding")
    out["Terminology Clarity"] = _score20("llm_terminology_clarity_score20", "terminology_clarity")
    # Compliance is reported by default as internal evaluator score converted to 0~20.
    # (Direct LLM compliance score remains available in CSV as llm_compliance_score20.)
    if "compliance" in ok.columns:
        out["Compliance"] = (pd.to_numeric(ok["compliance"], errors="coerce") * 20.0).clip(0.0, 20.0)
    else:
        out["Compliance"] = _score20("llm_compliance_score20", "compliance")

    if "mean_delta_total_100" in ok.columns and ok["mean_delta_total_100"].notna().any():
        ug20 = 10.0 + pd.to_numeric(ok["mean_delta_total_100"], errors="coerce") / 10.0
    else:
        ug20 = 10.0 + pd.to_numeric(ok.get("understanding_gain", 0.0), errors="coerce") * 10.0
    out["Understanding Gain"] = ug20.clip(0.0, 20.0)

    mr = pd.to_numeric(ok.get("misinterpretation_rate", np.nan), errors="coerce")
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


def _plot_model_score20_heatmap(score_df: pd.DataFrame, out_png: Path) -> None:
    if score_df.empty:
        return

    vals = score_df.to_numpy(dtype=float)
    h = max(4.5, 1.2 + 0.65 * vals.shape[0])
    w = max(10.0, 1.8 + 1.7 * vals.shape[1])
    fig, ax = plt.subplots(figsize=(w, h))
    im = ax.imshow(vals, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=20.0)

    ax.set_title("모델별 LLM 직접/환산 채점 지표 (0~20)", fontsize=16, pad=10)
    ax.set_xlabel("metric")
    ax.set_ylabel("model")
    ax.set_xticks(np.arange(vals.shape[1]))
    ax.set_xticklabels(score_df.columns.tolist(), rotation=90)
    ax.set_yticks(np.arange(vals.shape[0]))
    ax.set_yticklabels(score_df.index.tolist())

    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            txt = "-" if not np.isfinite(v) else f"{v:.1f}"
            color = "white" if np.isfinite(v) and v >= 10.0 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=10, color=color)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("score (0~20)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.out_dir / ts
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"[benchmark] run_dir={run_dir}", flush=True)
    print(
        f"[benchmark] mode={'synthetic-safe' if args.synthetic_safe_mode else 'full-data'} "
        f"models={len(args.models)} top_k={args.top_k}",
        flush=True,
    )

    rec: Optional[ThinFilerRecommender] = None
    eval_snapshots: Optional[pd.DataFrame] = None
    if not args.synthetic_safe_mode:
        print("[benchmark] preparing recommender/snapshots...", flush=True)
        cfg = RecommenderConfig(
            data_root=args.data_root,
            top_k=args.top_k,
            recommender_family=args.family,
            use_moe_harness=bool(args.use_moe_harness),
            moe_debug=bool(args.moe_debug),
            moe_default_weights={
                "ranker": float(args.moe_ranker_weight),
                "baseline": float(args.moe_baseline_weight),
                "utility": float(args.moe_utility_weight),
            },
            moe_deposit_baseline_boost=float(args.moe_deposit_baseline_boost),
            moe_fund_utility_boost=float(args.moe_fund_utility_boost),
            moe_low_risk_fund_penalty=float(args.moe_low_risk_fund_penalty),
        )
        rec = ThinFilerRecommender(cfg)

        snapshots = rec.build_user_snapshots(as_of_dates=args.as_of_dates, sample_users=args.sample_users)
        print(f"[benchmark] snapshots loaded rows={len(snapshots)}", flush=True)
        rec.load_products()
        print("[benchmark] products loaded", flush=True)
        if args.fit:
            print("[benchmark] fitting ranker...", flush=True)
            rec.fit(snapshots=snapshots, max_users=args.max_train_users)
            print("[benchmark] fit complete", flush=True)

        eval_snapshots = sample_users(
            snapshots,
            user_col=cfg.user_key_11,
            max_users=args.max_eval_users,
            random_state=cfg.random_state,
        )
        eval_snapshots = _one_snapshot_per_user(eval_snapshots, cfg.user_key_11)
        print(f"[benchmark] eval users={eval_snapshots[cfg.user_key_11].nunique()}", flush=True)

    rows: List[Dict[str, Any]] = []
    full_reports: Dict[str, Any] = {}
    total_models = len(args.models)

    for i, model_name in enumerate(args.models, start=1):
        model_id, resolve_note = resolve_model_id(model_name)
        started = time.time()
        rec_key = f"{i:02d}_{model_id.replace('/', '_')}"
        print(f"[benchmark] [{i}/{total_models}] model={model_name} resolved={model_id} start", flush=True)
        try:
            llm_renderer = OpenAILLMRenderer(
                model=model_id,
                prompt_path=args.llm_prompt_path,
                timeout_seconds=float(args.llm_timeout_seconds),
                max_retries=int(args.llm_max_retries),
            )
            understanding_evaluator = None
            if args.enable_understanding_eval:
                understanding_evaluator = ExplainerUnderstandingEvaluator(
                    use_llm_user_simulator=bool(args.use_llm_user_simulator),
                    use_llm_evaluator=bool(args.use_llm_evaluator),
                    simulator_model=str(args.simulator_model),
                    evaluator_model=str(args.evaluator_model),
                    simulator_prompt_path=args.simulator_prompt_path,
                    evaluator_prompt_path=args.evaluator_prompt_path,
                )

            if args.synthetic_safe_mode:
                batch = _evaluate_synthetic_batch(
                    llm_renderer=llm_renderer,
                    understanding_evaluator=understanding_evaluator,
                    max_understanding_samples=int(args.max_understanding_samples),
                    synthetic_repeat=int(args.synthetic_repeat),
                    use_explainer_moe=bool(args.use_explainer_moe),
                    compliance_rules_path=args.compliance_rules_path,
                )
            else:
                assert rec is not None
                assert eval_snapshots is not None
                explainer = GroundedExplainer(
                    rec,
                    llm_renderer=llm_renderer,
                    fallback_to_template_on_verify_fail=not args.no_template_fallback,
                    use_explainer_moe=bool(args.use_explainer_moe),
                    compliance_rules_path=args.compliance_rules_path,
                    explainer_moe_debug=bool(args.explainer_moe_debug),
                )
                batch = evaluate_explainer_batch(
                    explainer=explainer,
                    eval_snapshots=eval_snapshots,
                    top_k=args.top_k,
                    understanding_evaluator=understanding_evaluator,
                    max_understanding_samples=int(args.max_understanding_samples),
                )

            um = (batch.get("understanding_eval", {}) or {}).get("metrics", {}) or {}
            records = (batch.get("understanding_eval", {}) or {}).get("records", []) or []
            render_stats = (batch.get("render_stats", {}) or {})
            if render_stats:
                src_total = max(1, int(render_stats.get("total", 0)))
                llm_ratio = float(render_stats.get("llm", 0)) / src_total
                fallback_ratio = float(render_stats.get("template_fallback", 0)) / src_total
                template_ratio = float(render_stats.get("template", 0)) / src_total
                nonempty_ratio = float(render_stats.get("nonempty", 0)) / src_total
            else:
                src_vals = [str(r.get("render_source", "")) for r in records]
                txt_vals = [str(r.get("rendered_explanation", "") or "") for r in records]
                src_total = max(1, len(src_vals))
                llm_ratio = float(sum(1 for s in src_vals if s == "llm")) / src_total if src_vals else 0.0
                fallback_ratio = (
                    float(sum(1 for s in src_vals if s == "template_fallback")) / src_total if src_vals else 0.0
                )
                template_ratio = float(sum(1 for s in src_vals if s == "template")) / src_total if src_vals else 0.0
                nonempty_ratio = (
                    float(sum(1 for t in txt_vals if len(t.strip()) > 0)) / max(1, len(txt_vals))
                    if txt_vals
                    else 0.0
                )
            status = "ok" if nonempty_ratio > 0.0 else "degraded_empty_output"
            err_msg = "" if status == "ok" else "LLM returned empty explanation text for all sampled items."
            llm_debug = getattr(llm_renderer, "last_debug", {}) or {}
            llm_quality_result: Dict[str, float] = {}
            if not bool(args.disable_llm_quality_scoring):
                quality_model = str(args.llm_quality_model)
                if quality_model.strip().lower() == "auto":
                    quality_model = str(model_id)
                try:
                    quality_scorer = _LLMQualityScorer(
                        model=quality_model,
                        timeout_seconds=float(args.llm_quality_timeout_seconds),
                        max_retries=int(args.llm_quality_max_retries),
                    )
                    llm_quality_result = _score_quality_by_llm(
                        records=records,
                        scorer=quality_scorer,
                        max_items=int(args.llm_quality_max_items),
                    )
                except Exception:
                    llm_quality_result = {}
            row = {
                "model_input": str(model_name),
                "model_id": str(model_id),
                "resolve_note": resolve_note,
                "status": status,
                "error": err_msg,
                "elapsed_sec": round(time.time() - started, 3),
                "verification_pass_rate": _metric_get(batch.get("metrics", {}), "verification_pass_rate"),
                "reason_coverage_rc": _metric_get(batch.get("metrics", {}), "reason_coverage_rc"),
                "hallucination_rate_hr": _metric_get(batch.get("metrics", {}), "hallucination_rate_hr"),
                "fact_consistency_rate": _metric_get(batch.get("metrics", {}), "fact_consistency_rate"),
                "forbidden_claim_rate": _metric_get(batch.get("metrics", {}), "forbidden_claim_rate"),
                "understanding_gain": _metric_get(um, "understanding_gain"),
                "misinterpretation_rate": _metric_get(um, "misinterpretation_rate"),
                "misinterpretation_rate_weighted": _metric_get(um, "misinterpretation_rate_weighted"),
                "misinterpretation_rate_raw": _metric_get(um, "misinterpretation_rate_raw"),
                "mean_misinterpretation_major_count": _metric_get(um, "mean_misinterpretation_major_count"),
                "mean_misinterpretation_minor_count": _metric_get(um, "mean_misinterpretation_minor_count"),
                "mean_total_before_100": _metric_get(um, "mean_total_before_100"),
                "mean_total_after_100": _metric_get(um, "mean_total_after_100"),
                "mean_delta_total_100": _metric_get(um, "mean_delta_total_100"),
                "personalization": _metric_get(um, "personalization"),
                "product_grounding": _metric_get(um, "product_grounding"),
                "terminology_clarity": _metric_get(um, "terminology_clarity"),
                "compliance": _metric_get(um, "compliance"),
                "compliance_score20_internal": _metric_get(um, "compliance") * 20.0,
                "evaluated_explanations": int((batch.get("coverage", {}) or {}).get("evaluated_explanations", 0)),
                "evaluated_users": int((batch.get("coverage", {}) or {}).get("evaluated_users", 0)),
                "llm_render_ratio": llm_ratio,
                "template_fallback_ratio": fallback_ratio,
                "template_render_ratio": template_ratio,
                "llm_nonempty_ratio": nonempty_ratio,
                "llm_last_stage": str(llm_debug.get("stage", "")),
                "llm_last_error": str(llm_debug.get("responses_error", "") or llm_debug.get("chat_error", "")),
                "llm_personalization_score20": llm_quality_result.get("llm_personalization_score20", float("nan")),
                "llm_product_grounding_score20": llm_quality_result.get("llm_product_grounding_score20", float("nan")),
                "llm_terminology_clarity_score20": llm_quality_result.get("llm_terminology_clarity_score20", float("nan")),
                "llm_compliance_score20": llm_quality_result.get("llm_compliance_score20", float("nan")),
                "llm_quality_eval_count": llm_quality_result.get("llm_quality_eval_count", 0.0),
                "llm_quality_eval_success_count": llm_quality_result.get("llm_quality_eval_success_count", 0.0),
            }
            rows.append(row)
            full_reports[rec_key] = {
                "model_input": model_name,
                "model_id": model_id,
                "resolve_note": resolve_note,
                "use_explainer_moe": bool(args.use_explainer_moe),
                "compliance_rules_path": str(args.compliance_rules_path),
                "batch": batch,
                "llm_quality_result": llm_quality_result,
            }
            print(
                f"[benchmark] [{i}/{total_models}] model={model_name} done "
                f"status={status} elapsed={row['elapsed_sec']}s ug={row['understanding_gain']:.4f} mr={row['misinterpretation_rate']:.4f}",
                flush=True,
            )
        except Exception as e:
            row = {
                "model_input": str(model_name),
                "model_id": str(model_id),
                "resolve_note": resolve_note,
                "status": "error",
                "error": f"{type(e).__name__}: {e}",
                "elapsed_sec": round(time.time() - started, 3),
                "verification_pass_rate": 0.0,
                "reason_coverage_rc": 0.0,
                "hallucination_rate_hr": 1.0,
                "fact_consistency_rate": 0.0,
                "forbidden_claim_rate": 0.0,
                "understanding_gain": 0.0,
                "misinterpretation_rate": 1.0,
                "misinterpretation_rate_weighted": 1.0,
                "misinterpretation_rate_raw": 1.0,
                "mean_misinterpretation_major_count": 0.0,
                "mean_misinterpretation_minor_count": 0.0,
                "mean_total_before_100": 0.0,
                "mean_total_after_100": 0.0,
                "mean_delta_total_100": 0.0,
                "personalization": 0.0,
                "product_grounding": 0.0,
                "terminology_clarity": 0.0,
                "compliance": 0.0,
                "compliance_score20_internal": 0.0,
                "evaluated_explanations": 0,
                "evaluated_users": 0,
                "llm_personalization_score20": float("nan"),
                "llm_product_grounding_score20": float("nan"),
                "llm_terminology_clarity_score20": float("nan"),
                "llm_compliance_score20": float("nan"),
                "llm_quality_eval_count": 0.0,
                "llm_quality_eval_success_count": 0.0,
            }
            rows.append(row)
            full_reports[rec_key] = {
                "model_input": model_name,
                "model_id": model_id,
                "resolve_note": resolve_note,
                "error": row["error"],
            }
            print(
                f"[benchmark] [{i}/{total_models}] model={model_name} error={row['error']}",
                flush=True,
            )
            if not args.continue_on_error:
                break

    df = pd.DataFrame(rows)
    summary_csv = run_dir / "model_benchmark_summary.csv"
    summary_json = run_dir / "model_benchmark_summary.json"
    detail_json = raw_dir / "model_benchmark_detail.json"
    report_md = run_dir / "report.md"
    fig_png = run_dir / "benchmark_verifier_ug_mr.png"
    heatmap_png = run_dir / "benchmark_model_score20_heatmap.png"
    heatmap_csv = run_dir / "benchmark_model_score20_heatmap_table.csv"

    df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    summary_payload = {
        "generated_at": ts,
        "config": {
            "family": args.family,
            "sample_users": int(args.sample_users),
            "max_eval_users": int(args.max_eval_users),
            "top_k": int(args.top_k),
            "fit": bool(args.fit),
            "synthetic_safe_mode": bool(args.synthetic_safe_mode),
            "synthetic_repeat": int(args.synthetic_repeat),
            "use_explainer_moe": bool(args.use_explainer_moe),
            "explainer_moe_debug": bool(args.explainer_moe_debug),
            "compliance_rules_path": str(args.compliance_rules_path),
            "disable_llm_quality_scoring": bool(args.disable_llm_quality_scoring),
            "llm_quality_model": str(args.llm_quality_model),
            "llm_quality_max_items": int(args.llm_quality_max_items),
            "models": [str(x) for x in args.models],
        },
        "rows": rows,
    }
    summary_json.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    detail_json.write_text(json.dumps(full_reports, ensure_ascii=False, indent=2), encoding="utf-8")

    _plot_summary(df, fig_png)
    score20 = _build_model_score20_table(df)
    if not score20.empty:
        score20.to_csv(heatmap_csv, encoding="utf-8-sig")
        _plot_model_score20_heatmap(score20, heatmap_png)

    lines = [
        "# LLM Model Benchmark Report",
        "",
        f"- generated_at: `{ts}`",
        f"- family: `{args.family}`",
        f"- top_k: `{args.top_k}`",
        f"- sample_users: `{args.sample_users}`",
        f"- max_eval_users: `{args.max_eval_users}`",
        f"- fit: `{bool(args.fit)}`",
        "",
        "## Summary",
        f"- csv: `{summary_csv}`",
        f"- json: `{summary_json}`",
        f"- detail: `{detail_json}`",
        f"- figure: `{fig_png}`",
        f"- figure: `{heatmap_png}`",
        f"- table: `{heatmap_csv}`",
        "",
        "## Notes",
        "- `model_input`은 사용자가 입력한 이름입니다.",
        "- `model_id`는 실제 API 호출에 사용한 모델 ID입니다.",
        "- `resolve_note=alias:*`는 별칭을 API ID로 변환한 경우입니다.",
        "- `status=error`인 모델은 API 모델명/권한/네트워크 이슈일 수 있습니다.",
        "- `status=degraded_empty_output`은 호출은 되었지만 설명 텍스트가 비어 있어 비교 신뢰도가 낮은 상태입니다.",
        "- `llm_render_ratio`가 낮고 `template_fallback_ratio`가 높으면 모델 비교가 왜곡될 수 있습니다.",
        "- 순수 LLM 비교가 목적이면 `--no-template-fallback` 사용을 권장합니다.",
        "- `--use-explainer-moe`를 켜면 설명 단계에서 reason/compliance/template expert 라우팅을 적용합니다.",
        "- `--compliance-rules-path`의 txt 규칙은 외부 컴플라이언스 금지/필수 문구 검증에 반영됩니다.",
        "- 이해도 채점은 문항별 0~20 실수 점수이며, UG는 (after-before)/100 으로 정규화됩니다.",
        "- 설명 없음(before) 조건에서는 detail/comparison/warnings를 숨겨 baseline ceiling effect를 완화합니다.",
        "- CSV 컬럼 `mean_total_before_100`, `mean_total_after_100`, `mean_delta_total_100`를 함께 확인하세요.",
        "- `misinterpretation_rate`는 major/minor 가중치(MR=(major+0.5*minor)/5) 기반입니다.",
        "- raw 기준이 필요하면 `misinterpretation_rate_raw`를 확인하세요.",
        "- 컴플라이언스 내부 환산 점수(0~20)는 `compliance_score20_internal` 컬럼입니다.",
        "- LLM 직접 품질 채점 컬럼: `llm_personalization_score20`, `llm_product_grounding_score20`, `llm_terminology_clarity_score20`, `llm_compliance_score20`",
    ]
    report_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[benchmark] summary_csv={summary_csv}", flush=True)
    print(f"[benchmark] summary_json={summary_json}", flush=True)
    print(f"[benchmark] report_md={report_md}", flush=True)

    print(json.dumps({"run_dir": str(run_dir), "summary_csv": str(summary_csv), "report_md": str(report_md)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
