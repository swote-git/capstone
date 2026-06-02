from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from .common import FORBIDDEN_PATTERNS, expected_product_info_lines, expected_summary_line


def render_explanation(explanation_object: Dict[str, Any]) -> str:
    reasons = explanation_object.get("model_reasons", [])
    warnings = explanation_object.get("warnings", [])
    comparison = explanation_object.get("comparison", {})
    user_summary = explanation_object.get("user_summary", {})
    product = explanation_object.get("recommended_product", {})
    detail = explanation_object.get("recommended_product_detail", {})
    ranking_ctx = explanation_object.get("ranking_context", {})

    risk = str(product.get("risk", "보통"))
    user_risk = str(user_summary.get("risk_preference", "보통"))
    principal_var = bool(detail.get("principal_variation", False))
    high_risk_for_user = (
        risk in {"높음", "매우 높음"} and user_risk in {"낮음", "보통"}
    )

    reason_lines: List[str] = []
    if ranking_ctx.get("same_score_topk"):
        reason_lines.append("상위권 후보로 함께 분류된 상품입니다.")
    if high_risk_for_user or principal_var:
        reason_lines.append("위험 측면에서 완전한 일치는 아니므로 손실 가능성을 먼저 확인해 주세요.")
    reason_lines.extend([str(r) for r in reasons])

    warning_lines = list(str(w) for w in warnings)
    if product.get("family") == "fund":
        if "과거 수익률이 미래 수익률을 보장하지 않습니다." not in warning_lines:
            warning_lines.append("과거 수익률이 미래 수익률을 보장하지 않습니다.")
        if "원금 변동 가능성이 있습니다." not in warning_lines:
            warning_lines.append("원금 변동 가능성이 있습니다.")

    glossary = [
        "단리: 원금에 대해서만 이자가 붙는 방식입니다.",
        "복리: 이자에도 다시 이자가 붙는 방식입니다.",
        "만기: 약정한 기간이 끝나 자금을 찾을 수 있는 시점입니다.",
        "원금 변동: 투자 결과에 따라 원금이 늘거나 줄 수 있다는 뜻입니다.",
        "최대낙폭: 일정 기간 중 고점 대비 가장 크게 하락한 폭입니다.",
        "운용보수/판매보수: 펀드 운용·판매 과정에서 발생하는 비용입니다.",
    ]

    return (
        "[왜 이 상품인가]\n"
        + "\n".join(f"- {x}" for x in reason_lines[:5])
        + "\n\n[꼭 알아둘 점]\n"
        + "\n".join(f"- {x}" for x in warning_lines[:5])
        + "\n\n[쉬운 용어 풀이]\n"
        + "\n".join(f"- {x}" for x in glossary[:4])
        + "\n\n[대안과의 차이]\n"
        + f"- 대안 {comparison.get('alternative', '')} 대비: {comparison.get('difference', '')}\n"
        + "\n[한줄 정리]\n"
        + f"- {product.get('family', '')} 상품으로, 위험은 {risk}이며 고객님의 유동성 필요({user_summary.get('liquidity_need','보통')})를 함께 고려한 후보입니다."
    )


def reason_alignment(rendered_text: str, model_reasons: Sequence[str]) -> float:
    if not model_reasons:
        return 1.0
    found_exact = sum(1 for r in model_reasons if r in rendered_text)
    exact_score = found_exact / len(model_reasons)

    # Soft fallback for paraphrased customer-friendly responses.
    soft_hits = 0
    for r in model_reasons:
        keys: List[str] = []
        if "위험" in r:
            keys.extend(["위험", "손실", "변동"])
        if "유동성" in r:
            keys.extend(["유동성", "자금", "해지", "만기"])
        if "복잡" in r or "이해" in r:
            keys.extend(["복잡", "이해", "어려"])
        if "기간" in r:
            keys.extend(["기간", "만기", "중기", "단기", "장기"])
        if "금액" in r or "가입 요건" in r:
            keys.extend(["가입금액", "최소", "요건"])
        if "상품군" in r:
            keys.extend(["상품군", "예금", "펀드"])
        if "디지털" in r or "채널" in r:
            keys.extend(["채널", "모바일", "인터넷"])
        if "잠재력" in r or "tps" in r.lower():
            keys.extend(["거래", "활동", "잠재", "신뢰"])
        if keys and any(k in rendered_text for k in keys):
            soft_hits += 1
    soft_score = soft_hits / len(model_reasons)
    return max(exact_score, soft_score)


def check_fact_consistency(
    rendered_text: str,
    product_facts: Dict[str, Any],
    explanation_object: Dict[str, Any],
) -> bool:
    family = str(product_facts.get("family", ""))
    risk = str(product_facts.get("risk", ""))
    liquidity = str(product_facts.get("liquidity", ""))
    alternative = str(explanation_object.get("comparison", {}).get("alternative", ""))
    difference = str(explanation_object.get("comparison", {}).get("difference", ""))

    family_alias = {
        "deposit": ["deposit", "예금", "적금", "수신상품"],
        "fund": ["fund", "펀드", "공모펀드"],
    }
    alt_alias = family_alias.get(alternative, [alternative])
    fam_alias = family_alias.get(family, [family])

    fact_flags = [
        any(x and x in rendered_text for x in fam_alias),
        bool(risk) and (risk in rendered_text or "위험" in rendered_text),
        bool(liquidity) and (liquidity in rendered_text or "유동성" in rendered_text or "만기" in rendered_text),
        any(x and x in rendered_text for x in alt_alias),
        (difference in rendered_text) or ("차이" in rendered_text and ("위험" in rendered_text or "수익" in rendered_text)),
    ]
    return sum(1 for f in fact_flags if f) >= 4


def hallucination_rate(rendered_text: str, explanation_object: Dict[str, Any]) -> float:
    lines = [ln.strip()[2:] for ln in rendered_text.splitlines() if ln.strip().startswith("-")]
    if not lines:
        return 0.0

    allowed_texts = []
    allowed_texts.extend(explanation_object.get("model_reasons", []))
    allowed_texts.extend(explanation_object.get("warnings", []))
    cmp = explanation_object.get("comparison", {})
    allowed_texts.append(f"대안 {cmp.get('alternative', '')} 대비: {cmp.get('difference', '')}")
    allowed_texts.append(expected_summary_line(explanation_object))
    allowed_texts.extend(expected_product_info_lines(explanation_object))

    keyword_pool: List[str] = []
    for txt in allowed_texts:
        for tok in re.findall(r"[가-힣A-Za-z]{2,}", str(txt)):
            if tok not in keyword_pool:
                keyword_pool.append(tok)
    keyword_pool.extend(
        [
            "단리",
            "복리",
            "만기",
            "원금",
            "최대낙폭",
            "운용보수",
            "판매보수",
            "상위권 후보",
            "주의",
            "손실",
            "대안",
            "차이",
        ]
    )

    unknown = 0
    for claim in lines:
        if any(k and k in claim for k in keyword_pool):
            continue
        unknown += 1
    return unknown / len(lines)


def contains_forbidden_claims(
    rendered_text: str,
    extra_forbidden_patterns: Optional[Sequence[str]] = None,
) -> List[str]:
    hits: List[str] = []
    text_lower = rendered_text.lower()
    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, text_lower):
            hits.append(pattern)
    for pattern in list(extra_forbidden_patterns or []):
        if not pattern:
            continue
        if pattern.lower() in text_lower:
            hits.append(f"external:{pattern}")

    internal_tokens = [
        "liquidity_match",
        "risk_match",
        "tps_score",
        "principal_variation",
        "product_source_data",
    ]
    for token in internal_tokens:
        if token.lower() in text_lower:
            hits.append(f"internal_token:{token}")
    return hits


def verify(
    rendered_text: str,
    explanation_object: Dict[str, Any],
    product_facts: Dict[str, Any],
    extra_forbidden_patterns: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    model_reasons = explanation_object["model_reasons"]
    reason_alignment_score = reason_alignment(rendered_text, model_reasons)
    fact_ok = check_fact_consistency(rendered_text, product_facts, explanation_object)
    hallucination = hallucination_rate(rendered_text, explanation_object)
    forbidden = contains_forbidden_claims(
        rendered_text,
        extra_forbidden_patterns=extra_forbidden_patterns,
    )
    return {
        "reason_alignment": reason_alignment_score,
        "fact_consistency": fact_ok,
        "hallucination_rate": hallucination,
        "forbidden_claims": forbidden,
        "passed": bool(
            reason_alignment_score >= 0.67
            and fact_ok
            and hallucination <= 0.20
            and not forbidden
        ),
    }
