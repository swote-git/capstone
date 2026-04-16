from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

FEATURE_LABELS = {
    "risk_match": "risk fit",
    "liquidity_match": "liquidity fit",
    "horizon_match": "time-horizon fit",
    "complexity_match": "complexity fit",
    "amount_feasibility": "amount feasibility",
    "family_match": "product-family fit",
    "digital_match": "digital-behavior fit",
}

FORBIDDEN_PATTERNS = [
    r"guaranteed return",
    r"no risk",
    r"must choose",
    r"approval likelihood",
]


@dataclass
class ReasonSignal:
    feature: str
    value: float
    impact: str
    contribution: float


def risk_label(value: float) -> str:
    if value < 1.0:
        return "low"
    if value < 2.0:
        return "medium"
    if value < 2.7:
        return "high"
    return "very_high"


def liquidity_label(value: float) -> str:
    if value < 1.0:
        return "low"
    if value < 2.0:
        return "medium"
    return "high"


def horizon_label(value: float) -> str:
    code = int(round(value))
    return {0: "short", 1: "mid", 2: "long"}.get(code, "mid")


def complexity_label(value: float) -> str:
    if value < 0.8:
        return "low"
    if value < 1.6:
        return "medium"
    return "high"


def reason_sentence(
    signal: ReasonSignal,
    user_summary: Dict[str, Any],
    product_facts: Dict[str, Any],
) -> str:
    f = signal.feature
    if f == "risk_match":
        return f"This product matches your {user_summary['risk_preference']} risk preference."
    if f == "liquidity_match":
        return f"This product matches your {user_summary['liquidity_need']} liquidity need."
    if f == "complexity_match":
        return f"This product complexity fits your {user_summary['financial_knowledge']} financial knowledge level."
    if f == "horizon_match":
        return "This product horizon is aligned with your investment time preference."
    if f == "amount_feasibility":
        return "Your available amount is compatible with this product's minimum requirement."
    if f == "family_match":
        return f"The {product_facts['family']} family is a suitable type for your current profile."
    if f == "digital_match":
        return "Your digital usage pattern is compatible with this product profile."
    label = FEATURE_LABELS.get(f, f)
    return f"Model indicates positive contribution from {label}."


def warnings_from_facts(facts: Dict[str, Any]) -> List[str]:
    warnings: List[str] = []
    if facts["family"] == "deposit":
        warnings.append("Lower returns compared to investment products.")
    if facts["family"] == "fund":
        warnings.append("Principal value can fluctuate with market conditions.")
    if facts["risk"] in {"high", "very_high"}:
        warnings.append("Short-term losses are possible due to higher risk level.")
    if not warnings:
        warnings.append("Review fees, term conditions, and liquidity constraints before decision.")
    return warnings


def expected_summary_line(explanation_object: Dict[str, Any]) -> str:
    u = explanation_object["user_summary"]
    p = explanation_object["recommended_product"]
    return (
        f"Recommended {p['family']} with {p['risk']} risk and {p['liquidity']} liquidity "
        f"for a user with {u['risk_preference']} risk preference and {u['liquidity_need']} liquidity need."
    )


def top_feature_cols(rec: Any) -> Sequence[str]:
    if rec.feature_columns:
        return list(rec.feature_columns)
    return list(FEATURE_LABELS.keys())
