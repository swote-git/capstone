from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

FEATURE_LABELS = {
    "risk_match": "위험 성향 적합도",
    "liquidity_match": "유동성 필요 적합도",
    "horizon_match": "투자 기간 적합도",
    "complexity_match": "복잡도 수용 적합도",
    "amount_feasibility": "가입 금액 충족 가능성",
    "family_match": "상품군 적합도",
    "digital_match": "디지털 행동 적합도",
}

FORBIDDEN_PATTERNS = [
    r"원금\s*보장",
    r"수익\s*보장",
    r"무위험",
    r"반드시\s*가입",
    r"무조건\s*가입",
    r"승인\s*가능성",
    r"승인률",
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
        return "낮음"
    if value < 2.0:
        return "보통"
    if value < 2.7:
        return "높음"
    return "매우 높음"


def liquidity_label(value: float) -> str:
    if value < 1.0:
        return "낮음"
    if value < 2.0:
        return "보통"
    return "높음"


def horizon_label(value: float) -> str:
    code = int(round(value))
    return {0: "단기", 1: "중기", 2: "장기"}.get(code, "중기")


def complexity_label(value: float) -> str:
    if value < 0.8:
        return "낮음"
    if value < 1.6:
        return "보통"
    return "높음"


def reason_sentence(
    signal: ReasonSignal,
    user_summary: Dict[str, Any],
    product_facts: Dict[str, Any],
) -> str:
    f = signal.feature
    if f == "risk_match":
        return f"이 상품은 고객님의 위험 선호({user_summary['risk_preference']})와 잘 맞습니다."
    if f == "liquidity_match":
        return f"이 상품은 고객님의 유동성 필요 수준({user_summary['liquidity_need']})에 적합합니다."
    if f == "complexity_match":
        return f"이 상품의 복잡도는 고객님의 금융 이해 수준({user_summary['financial_knowledge']})에 맞습니다."
    if f == "horizon_match":
        return "이 상품의 운용 기간 특성은 고객님의 투자 기간 선호와 일치합니다."
    if f == "amount_feasibility":
        return "고객님의 자금 여건이 이 상품의 최소 가입 요건을 충족합니다."
    if f == "family_match":
        return f"고객님의 현재 프로파일에는 {product_facts['family']} 상품군이 적합합니다."
    if f == "digital_match":
        return "고객님의 디지털 이용 패턴이 이 상품 특성과 잘 부합합니다."
    label = FEATURE_LABELS.get(f, f)
    return f"모델 기준으로 {label} 항목이 긍정적으로 기여했습니다."


def warnings_from_facts(facts: Dict[str, Any]) -> List[str]:
    warnings: List[str] = []
    if facts["family"] == "deposit":
        warnings.append("투자형 상품 대비 기대수익이 낮을 수 있습니다.")
    if facts["family"] == "fund":
        warnings.append("시장 상황에 따라 원금 변동이 발생할 수 있습니다.")
    if facts["risk"] in {"높음", "매우 높음"}:
        warnings.append("위험 수준이 높아 단기 손실 가능성이 있습니다.")
    if not warnings:
        warnings.append("가입 전 보수·수수료, 만기 조건, 중도해지 제약을 확인하세요.")
    return warnings


def expected_summary_line(explanation_object: Dict[str, Any]) -> str:
    u = explanation_object["user_summary"]
    p = explanation_object["recommended_product"]
    return (
        f"{p['family']} 상품은 위험 수준 {p['risk']}, 유동성 {p['liquidity']} 특성을 가지며, "
        f"위험 선호가 {u['risk_preference']}이고 유동성 필요가 {u['liquidity_need']}인 사용자에게 적합합니다."
    )


def top_feature_cols(rec: Any) -> Sequence[str]:
    if rec.feature_columns:
        return list(rec.feature_columns)
    return list(FEATURE_LABELS.keys())
