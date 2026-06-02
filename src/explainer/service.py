from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from .common import (
    FEATURE_LABELS,
    FORBIDDEN_PATTERNS,
    ReasonSignal,
    complexity_label,
    expected_summary_line,
    horizon_label,
    liquidity_label,
    reason_sentence,
    risk_label,
    warnings_from_facts,
)
from .reasoning import (
    build_explanation_object,
    extract_reasons,
    local_contributions,
    retrieve_product_facts,
)
from .render_verify import (
    check_fact_consistency,
    contains_forbidden_claims,
    hallucination_rate,
    reason_alignment,
    render_explanation as render_explanation_template,
    verify,
)
from .moe_orchestrator import ExplainerMoEOrchestrator
from .compliance_rules import (
    evaluate_compliance_rules,
    load_compliance_rules,
    scoped_forbidden_patterns,
)


class GroundedExplainer:
    """Strict grounded explanation pipeline.

    Layers:
    1) Reason extractor
    2) Product fact retriever
    3) Explanation object builder
    4) Renderer
    5) Verifier
    """

    def __init__(
        self,
        recommender: Any,
        top_reason_k: int = 3,
        llm_renderer: Optional[Any] = None,
        fallback_to_template_on_verify_fail: bool = True,
        use_explainer_moe: bool = False,
        compliance_rules_path: Optional[Path] = None,
        explainer_moe_debug: bool = False,
    ) -> None:
        self.rec = recommender
        self.top_reason_k = top_reason_k
        self.llm_renderer = llm_renderer
        self.fallback_to_template_on_verify_fail = fallback_to_template_on_verify_fail
        self.use_explainer_moe = bool(use_explainer_moe)
        self.explainer_moe_debug = bool(explainer_moe_debug)
        self.compliance_rules = load_compliance_rules(compliance_rules_path)
        self.explainer_moe = (
            ExplainerMoEOrchestrator(
                llm_renderer=llm_renderer,
                compliance_rules_path=compliance_rules_path,
                include_template_expert=True,
            )
            if (self.llm_renderer is not None and self.use_explainer_moe)
            else None
        )

    def explain_top_k(self, user_snapshot: pd.Series, k: int = 5) -> Dict[str, Any]:
        ranked = self._rank_with_context(user_snapshot, k=k)
        score_values = ranked["score"].to_numpy(dtype=float) if "score" in ranked.columns else []
        same_score_topk = False
        if len(score_values) > 1:
            same_score_topk = (float(score_values.max()) - float(score_values.min())) <= 1e-9
        outputs: List[Dict[str, Any]] = []
        for _, row in ranked.iterrows():
            reason_signals = self.extract_reasons(row)
            product_facts = self.retrieve_product_facts(row)
            explanation_object = self.build_explanation_object(
                user_snapshot,
                product_facts,
                reason_signals,
                ranking_context={
                    "same_score_topk": bool(same_score_topk),
                    "same_score_note": "상위권 후보로 함께 분류된 상품입니다." if same_score_topk else "",
                },
            )
            rendered = ""
            verification: Dict[str, Any] = {}
            render_source = "template"
            llm_intermediate: Optional[Dict[str, Any]] = None

            if self.explainer_moe is not None:
                route_bundle = self.explainer_moe.build_candidates(explanation_object)
                best_payload = self._select_best_candidate(
                    route_bundle.get("candidates", []),
                    explanation_object,
                    product_facts,
                )
                rendered = best_payload["rendered_explanation"]
                verification = best_payload["verification"]
                render_source = str(best_payload.get("render_source", "llm"))
                llm_intermediate = {
                    "orchestrator": {
                        "route": route_bundle.get("route"),
                        "rules": route_bundle.get("rules"),
                        "candidates_evaluated": best_payload.get("candidate_evaluations", []),
                    },
                    "llm_debug": getattr(self.llm_renderer, "last_debug", None),
                }
            else:
                rendered = self.render_explanation(explanation_object)
                rendered = self._sanitize_customer_text(rendered)
                verification = self.verify(rendered, explanation_object, product_facts)
                render_source = "llm" if self.llm_renderer is not None else "template"

            if (
                self.llm_renderer is not None
                and self.fallback_to_template_on_verify_fail
                and not bool(verification.get("passed", False))
                and render_source != "template"
            ):
                prev_debug = {
                    "llm_rendered_explanation": rendered,
                    "llm_verification": verification,
                    "llm_debug": getattr(self.llm_renderer, "last_debug", None),
                }
                rendered = self.render_explanation_template(explanation_object)
                rendered = self._sanitize_customer_text(rendered)
                verification = self.verify(rendered, explanation_object, product_facts)
                render_source = "template_fallback"
                llm_intermediate = {**(llm_intermediate or {}), **prev_debug}

            outputs.append(
                {
                    "product_id": str(row["product_id"]),
                    "score": float(row["score"]),
                    "reason_signals": [r.__dict__ for r in reason_signals],
                    "product_facts": product_facts,
                    "explanation_object": explanation_object,
                    "rendered_explanation": rendered,
                    "verification": verification,
                    "render_source": render_source,
                    "llm_intermediate": llm_intermediate,
                }
            )

        return {
            "user_id": str(user_snapshot[self.rec.config.user_key_11]),
            "recommendations": outputs,
        }

    def _rank_with_context(self, user_snapshot: pd.Series, k: int) -> pd.DataFrame:
        candidates = self.rec.generate_candidates(user_snapshot)
        pair = self.rec._add_pair_features(pd.DataFrame([user_snapshot]), candidates)
        if hasattr(self.rec, "score_pairs"):
            scores, _ = self.rec.score_pairs(user_snapshot, pair)
        elif self.rec.model is not None and self.rec.feature_columns:
            scores = self.rec.model.predict(pair[self.rec.feature_columns].fillna(0.0))
        else:
            scores = pair["baseline_score"].to_numpy()
        pair = pair.copy()
        pair["score"] = scores
        return pair.sort_values("score", ascending=False).head(k)

    def extract_reasons(self, pair_row: pd.Series) -> List[ReasonSignal]:
        return extract_reasons(self.rec, pair_row, self.top_reason_k)

    def _local_contributions(self, feature_cols: Sequence[str], values):
        return local_contributions(self.rec, feature_cols, values)

    def retrieve_product_facts(self, pair_row: pd.Series) -> Dict[str, Any]:
        return retrieve_product_facts(self.rec, pair_row)

    def build_explanation_object(
        self,
        user_snapshot: pd.Series,
        product_facts: Dict[str, Any],
        reason_signals: Sequence[ReasonSignal],
        ranking_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return build_explanation_object(
            user_snapshot,
            product_facts,
            reason_signals,
            ranking_context=ranking_context,
        )

    def render_explanation(self, explanation_object: Dict[str, Any]) -> str:
        if self.llm_renderer is not None:
            return self.llm_renderer.render(explanation_object)
        return self.render_explanation_template(explanation_object)

    def render_explanation_template(self, explanation_object: Dict[str, Any]) -> str:
        return render_explanation_template(explanation_object)

    def verify(
        self,
        rendered_text: str,
        explanation_object: Dict[str, Any],
        product_facts: Dict[str, Any],
    ) -> Dict[str, Any]:
        family = str(
            (explanation_object.get("recommended_product", {}) or {}).get(
                "family", product_facts.get("family", "")
            )
        )
        extra_patterns = scoped_forbidden_patterns(self.compliance_rules, family=family)
        result = verify(
            rendered_text,
            explanation_object,
            product_facts,
            extra_forbidden_patterns=extra_patterns,
        )
        rule_eval = evaluate_compliance_rules(
            rendered_text,
            self.compliance_rules,
            family=family,
        )
        result["external_compliance"] = {
            "rule_source": self.compliance_rules.source_path,
            "forbidden_hits": rule_eval["forbidden_hits"],
            "missing_required": rule_eval["missing_required"],
            "rule_passed": (len(rule_eval["forbidden_hits"]) == 0 and len(rule_eval["missing_required"]) == 0),
        }
        result["passed"] = bool(result.get("passed", False) and result["external_compliance"]["rule_passed"])
        return result

    def reason_alignment(self, rendered_text: str, model_reasons: Sequence[str]) -> float:
        return reason_alignment(rendered_text, model_reasons)

    def check_fact_consistency(
        self,
        rendered_text: str,
        product_facts: Dict[str, Any],
        explanation_object: Dict[str, Any],
    ) -> bool:
        return check_fact_consistency(rendered_text, product_facts, explanation_object)

    def hallucination_rate(self, rendered_text: str, explanation_object: Dict[str, Any]) -> float:
        return hallucination_rate(rendered_text, explanation_object)

    def contains_forbidden_claims(self, rendered_text: str):
        return contains_forbidden_claims(rendered_text)

    def _expected_summary_line(self, explanation_object: Dict[str, Any]) -> str:
        return expected_summary_line(explanation_object)

    def _reason_sentence(
        self,
        signal: ReasonSignal,
        user_summary: Dict[str, Any],
        product_facts: Dict[str, Any],
    ) -> str:
        return reason_sentence(signal, user_summary, product_facts)

    def _warnings_from_facts(self, facts: Dict[str, Any]):
        return warnings_from_facts(facts)

    @staticmethod
    def _risk_label(value: float) -> str:
        return risk_label(value)

    @staticmethod
    def _liquidity_label(value: float) -> str:
        return liquidity_label(value)

    @staticmethod
    def _horizon_label(value: float) -> str:
        return horizon_label(value)

    @staticmethod
    def _complexity_label(value: float) -> str:
        return complexity_label(value)

    @staticmethod
    def _sanitize_customer_text(text: str) -> str:
        """Remove internal feature tokens and noisy long decimals from customer-facing text."""
        if not text:
            return text
        out = str(text)

        # Hide internal feature/schema names from customer view.
        internal_tokens = [
            "liquidity_match",
            "risk_match",
            "tps_score",
            "principal_variation",
            "product_source_data",
            "reason_signals",
            "match_detail",
        ]
        for token in internal_tokens:
            out = re.sub(re.escape(token), "내부 평가 신호", out, flags=re.IGNORECASE)

        # Limit very long decimals that hurt readability (keep up to 2 decimal places).
        def _round_long_decimal(m: re.Match) -> str:
            try:
                val = float(m.group(0))
                return f"{val:.2f}"
            except Exception:
                return m.group(0)

        out = re.sub(r"\d+\.\d{4,}", _round_long_decimal, out)
        return out

    def _select_best_candidate(
        self,
        candidates: Sequence[Dict[str, Any]],
        explanation_object: Dict[str, Any],
        product_facts: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not candidates:
            txt = self._sanitize_customer_text(self.render_explanation_template(explanation_object))
            ver = self.verify(txt, explanation_object, product_facts)
            return {
                "render_source": "template",
                "rendered_explanation": txt,
                "verification": ver,
                "candidate_evaluations": [],
            }

        scored: List[Dict[str, Any]] = []
        for cand in candidates:
            raw_txt = str(cand.get("text", "") or "")
            txt = self._sanitize_customer_text(raw_txt)
            ver = self.verify(txt, explanation_object, product_facts)
            score = (
                (2.0 if bool(ver.get("passed", False)) else 0.0)
                + float(ver.get("reason_alignment", 0.0))
                + (1.0 if bool(ver.get("fact_consistency", False)) else 0.0)
                + (1.0 - float(ver.get("hallucination_rate", 1.0)))
                - (0.2 * len(ver.get("forbidden_claims", []) or []))
            )
            scored.append(
                {
                    "render_source": str(cand.get("source", "llm")),
                    "note": str(cand.get("note", "")),
                    "score": float(score),
                    "rendered_explanation": txt,
                    "verification": ver,
                }
            )

        best = sorted(scored, key=lambda x: x["score"], reverse=True)[0]
        return {
            "render_source": best["render_source"],
            "rendered_explanation": best["rendered_explanation"],
            "verification": best["verification"],
            "candidate_evaluations": scored if self.explainer_moe_debug else [],
        }


__all__ = [
    "GroundedExplainer",
    "ReasonSignal",
    "FEATURE_LABELS",
    "FORBIDDEN_PATTERNS",
]
