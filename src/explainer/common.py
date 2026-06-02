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
    weak_or_negative = (signal.impact == "negative") or (float(signal.value) < 0.55)
    if f == "risk_match":
        if weak_or_negative:
            return "위험 측면에서는 고객님 성향과 완전한 일치는 아니므로 주의가 필요합니다."
        return f"이 상품은 고객님의 위험 성향({user_summary['risk_preference']})과 비교적 잘 맞는 편입니다."
    if f == "liquidity_match":
        if weak_or_negative:
            return "유동성 측면은 대체로 맞지만, 자금 사용 시점은 한 번 더 점검하는 것이 좋습니다."
        return f"고객님의 유동성 필요 수준({user_summary['liquidity_need']})과 비교적 잘 맞는 편입니다."
    if f == "complexity_match":
        if weak_or_negative:
            return "상품 구조가 다소 복잡하게 느껴질 수 있어 가입 전 핵심 조건 확인이 필요합니다."
        return f"상품 이해 난이도는 고객님의 금융 이해 수준({user_summary['financial_knowledge']})에서 받아들이기 쉬운 편입니다."
    if f == "horizon_match":
        if weak_or_negative:
            return "자금 운용 기간 측면에서 완전한 일치는 아니므로 기간 계획을 먼저 확인해 주세요."
        return "자금 운용 기간 측면에서 고객님의 선호와 비교적 잘 맞습니다."
    if f == "amount_feasibility":
        if weak_or_negative:
            return "가입 가능 금액 조건은 충족 여부를 가입 직전에 다시 확인하는 것이 안전합니다."
        return "현재 자금 여건에서 가입 요건을 충족할 가능성이 높습니다."
    if f == "family_match":
        return f"고객님의 현재 상황에서는 {product_facts['family']} 상품군이 우선 검토 대상이 될 수 있습니다."
    if f == "digital_match":
        if weak_or_negative:
            return "이용 채널 측면에서 일부 불편할 수 있으니 가입/해지 채널을 확인해 주세요."
        return "가입/이용 채널 특성이 고객님의 이용 패턴과 비교적 잘 맞습니다."
    if f == "tps_score":
        return "고객님의 거래 안정성과 활동 정보가 추천 판단에 참고되었습니다."
    if f == "tps_trust":
        return "고객님의 신뢰 관련 정보가 추천 판단에 참고되었습니다."
    if f == "tps_activity":
        return "고객님의 거래·이용 활동 정보가 추천 판단에 참고되었습니다."
    if f == "tps_potential":
        return "고객님의 잠재 여력 정보가 현재 상품군 판단에 참고되었습니다."
    label = FEATURE_LABELS.get(f, f)
    if weak_or_negative:
        return f"{label} 측면은 완전한 일치가 아니므로 확인이 필요합니다."
    return f"{label} 측면은 비교적 잘 맞는 편입니다."


def warnings_from_facts(facts: Dict[str, Any]) -> List[str]:
    warnings: List[str] = []
    if facts["family"] == "deposit":
        warnings.append("투자형 상품 대비 기대수익이 낮을 수 있습니다.")
    if facts["family"] == "fund":
        warnings.append("시장 상황에 따라 원금 변동이 발생할 수 있습니다.")
        warnings.append("과거 수익률이 미래 수익률을 보장하지 않습니다.")
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


def expected_product_info_lines(explanation_object: Dict[str, Any]) -> List[str]:
    p = explanation_object.get("recommended_product", {})
    d = explanation_object.get("recommended_product_detail", {})
    meta = d.get("product_meta", {}) if isinstance(d, dict) else {}
    principal_var = bool(d.get("principal_variation", False)) if isinstance(d, dict) else False
    principal_text = "있음" if principal_var else "없음"

    lines = []
    product_name = str(p.get("product_name", "")).strip()
    if product_name:
        lines.append(f"상품명: {product_name}")

    lines.extend([
        f"상품군: {p.get('family', '')}",
        f"위험수준: {p.get('risk', '')}",
        f"유동성: {p.get('liquidity', '')}",
        f"투자기간: {d.get('horizon', '')}",
        f"복잡도: {d.get('complexity', '')}",
        f"원금 변동 가능성: {principal_text}",
    ])
    if "max_rate" in meta:
        try:
            lines.append(f"수익지표(참고): {float(meta.get('max_rate', 0.0)):.4f}")
        except Exception:
            pass
    return lines


def top_feature_cols(rec: Any) -> Sequence[str]:
    if rec.feature_columns:
        return list(rec.feature_columns)
    return list(FEATURE_LABELS.keys())
