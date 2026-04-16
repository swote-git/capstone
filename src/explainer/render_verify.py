from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence

from .common import FORBIDDEN_PATTERNS, expected_summary_line


def render_explanation(explanation_object: Dict[str, Any]) -> str:
    reasons = explanation_object["model_reasons"]
    warnings = explanation_object["warnings"]
    comparison = explanation_object["comparison"]
    user_summary = explanation_object["user_summary"]
    product = explanation_object["recommended_product"]

    reason_lines = "\n".join(f"- {r}" for r in reasons)
    warning_lines = "\n".join(f"- {w}" for w in warnings)

    summary = (
        f"- Recommended {product['family']} with {product['risk']} risk and {product['liquidity']} liquidity "
        f"for a user with {user_summary['risk_preference']} risk preference and {user_summary['liquidity_need']} liquidity need."
    )

    return (
        "[Reason]\n"
        f"{reason_lines}\n\n"
        "[Warning]\n"
        f"{warning_lines}\n\n"
        "[Comparison]\n"
        f"- Compared with {comparison['alternative']}: {comparison['difference']}\n\n"
        "[Simple Summary]\n"
        f"{summary}"
    )


def reason_alignment(rendered_text: str, model_reasons: Sequence[str]) -> float:
    if not model_reasons:
        return 1.0
    found = sum(1 for r in model_reasons if r in rendered_text)
    return found / len(model_reasons)


def check_fact_consistency(
    rendered_text: str,
    product_facts: Dict[str, Any],
    explanation_object: Dict[str, Any],
) -> bool:
    required_tokens = [
        str(product_facts["family"]),
        str(product_facts["risk"]),
        str(product_facts["liquidity"]),
        str(explanation_object["comparison"]["alternative"]),
        str(explanation_object["comparison"]["difference"]),
    ]
    return all(tok in rendered_text for tok in required_tokens)


def hallucination_rate(rendered_text: str, explanation_object: Dict[str, Any]) -> float:
    lines = [ln.strip()[2:] for ln in rendered_text.splitlines() if ln.strip().startswith("-")]
    if not lines:
        return 0.0

    allowed = set(explanation_object["model_reasons"])
    allowed.update(explanation_object["warnings"])
    allowed.add(
        f"Compared with {explanation_object['comparison']['alternative']}: {explanation_object['comparison']['difference']}"
    )
    allowed.add(expected_summary_line(explanation_object))

    unknown = sum(1 for claim in lines if claim not in allowed)
    return unknown / len(lines)


def contains_forbidden_claims(rendered_text: str) -> List[str]:
    hits: List[str] = []
    text_lower = rendered_text.lower()
    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, text_lower):
            hits.append(pattern)
    return hits


def verify(
    rendered_text: str,
    explanation_object: Dict[str, Any],
    product_facts: Dict[str, Any],
) -> Dict[str, Any]:
    model_reasons = explanation_object["model_reasons"]
    reason_alignment_score = reason_alignment(rendered_text, model_reasons)
    fact_ok = check_fact_consistency(rendered_text, product_facts, explanation_object)
    hallucination = hallucination_rate(rendered_text, explanation_object)
    forbidden = contains_forbidden_claims(rendered_text)
    return {
        "reason_alignment": reason_alignment_score,
        "fact_consistency": fact_ok,
        "hallucination_rate": hallucination,
        "forbidden_claims": forbidden,
        "passed": bool(reason_alignment_score >= 1.0 and fact_ok and hallucination == 0.0 and not forbidden),
    }
