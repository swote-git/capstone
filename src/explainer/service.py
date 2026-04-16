from __future__ import annotations

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
    ) -> None:
        self.rec = recommender
        self.top_reason_k = top_reason_k
        self.llm_renderer = llm_renderer
        self.fallback_to_template_on_verify_fail = fallback_to_template_on_verify_fail

    def explain_top_k(self, user_snapshot: pd.Series, k: int = 5) -> Dict[str, Any]:
        ranked = self._rank_with_context(user_snapshot, k=k)
        outputs: List[Dict[str, Any]] = []
        for _, row in ranked.iterrows():
            reason_signals = self.extract_reasons(row)
            product_facts = self.retrieve_product_facts(row)
            explanation_object = self.build_explanation_object(user_snapshot, product_facts, reason_signals)
            rendered = self.render_explanation(explanation_object)
            verification = self.verify(rendered, explanation_object, product_facts)
            render_source = "llm" if self.llm_renderer is not None else "template"

            if (
                self.llm_renderer is not None
                and self.fallback_to_template_on_verify_fail
                and not bool(verification.get("passed", False))
            ):
                rendered = self.render_explanation_template(explanation_object)
                verification = self.verify(rendered, explanation_object, product_facts)
                render_source = "template_fallback"

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
                }
            )

        return {
            "user_id": str(user_snapshot[self.rec.config.user_key_11]),
            "recommendations": outputs,
        }

    def _rank_with_context(self, user_snapshot: pd.Series, k: int) -> pd.DataFrame:
        candidates = self.rec.generate_candidates(user_snapshot)
        pair = self.rec._add_pair_features(pd.DataFrame([user_snapshot]), candidates)
        if self.rec.model is not None and self.rec.feature_columns:
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
        return retrieve_product_facts(pair_row)

    def build_explanation_object(
        self,
        user_snapshot: pd.Series,
        product_facts: Dict[str, Any],
        reason_signals: Sequence[ReasonSignal],
    ) -> Dict[str, Any]:
        return build_explanation_object(user_snapshot, product_facts, reason_signals)

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
        return verify(rendered_text, explanation_object, product_facts)

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


__all__ = [
    "GroundedExplainer",
    "ReasonSignal",
    "FEATURE_LABELS",
    "FORBIDDEN_PATTERNS",
]
