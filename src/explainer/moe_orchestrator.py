from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .compliance_rules import ComplianceRuleSet, load_compliance_rules
from .llm_renderer import OpenAILLMRenderer, build_customer_payload
from .render_verify import render_explanation as render_explanation_template


COMPLIANCE_EDITOR_PROMPT = (
    "당신은 금융추천 설명의 컴플라이언스 에디터입니다.\n"
    "역할: 제공된 설명 초안을, 제공된 사실과 규칙을 유지한 채 고객 친화적으로 교정합니다.\n"
    "[절대 규칙]\n"
    "1) explanation_object에 없는 사실을 추가하지 마세요.\n"
    "2) 과장/단정/권유성 표현을 제거하세요.\n"
    "3) 금지표현과 내부 피처명(risk_match 등)을 노출하지 마세요.\n"
    "4) 숫자는 고객 이해에 필요한 핵심만 남기고 과도한 소수점은 제거하세요.\n"
    "5) 펀드이며 고객 위험성향이 낮음/보통인데 상품위험이 높으면 주의 문장을 먼저 배치하세요.\n"
    "6) 과거 수익률은 미래 수익률을 보장하지 않음을 명시하세요.\n"
    "7) 펀드의 경우 원금 변동 가능성을 명시하세요.\n"
    "[출력 형식]\n"
    "[왜 이 상품인가]\n- ...\n\n"
    "[꼭 알아둘 점]\n- ...\n\n"
    "[쉬운 용어 풀이]\n- ...\n\n"
    "[대안과의 차이]\n- ...\n\n"
    "[한줄 정리]\n- ...\n"
)


@dataclass
class OrchestratorCandidate:
    source: str
    text: str
    route: str
    note: str = ""


class ExplainerMoEOrchestrator:
    """Explanation-layer MoE orchestrator.

    Expert 1: LLM reason renderer
    Expert 2: LLM compliance editor (optional, routed)
    Expert 3: deterministic template expert (always available)
    """

    def __init__(
        self,
        llm_renderer: OpenAILLMRenderer,
        compliance_rules_path: Optional[Path] = None,
        include_template_expert: bool = True,
    ) -> None:
        self.renderer = llm_renderer
        self.rules: ComplianceRuleSet = load_compliance_rules(compliance_rules_path)
        self.include_template_expert = bool(include_template_expert)

    @property
    def extra_forbidden_patterns(self) -> Sequence[str]:
        return list(self.rules.forbidden_patterns)

    @property
    def rules_debug(self) -> Dict[str, Any]:
        return {
            "source_path": self.rules.source_path,
            "has_rules": bool(self.rules.has_rules),
            "rule_count": int(self.rules.rule_count),
            "forbidden_count": int(len(self.rules.forbidden_patterns)),
            "required_count": int(len(self.rules.required_phrases)),
        }

    def _need_compliance_first(self, explanation_object: Dict[str, Any]) -> bool:
        product = explanation_object.get("recommended_product", {})
        user = explanation_object.get("user_summary", {})
        policy = explanation_object.get("explanation_policy", {})
        family = str(product.get("family", "")).strip().lower()
        risk = str(product.get("risk", "보통"))
        user_risk = str(user.get("risk_preference", "보통"))
        principal_var = bool(
            (explanation_object.get("recommended_product_detail", {}) or {}).get("principal_variation", False)
        )
        if bool(policy.get("caution_first", False)):
            return True
        if family == "fund" and (risk in {"높음", "매우 높음"} or principal_var) and user_risk in {"낮음", "보통"}:
            return True
        return False

    def _compliance_edit(self, explanation_object: Dict[str, Any], draft_text: str) -> str:
        payload = {
            "explanation_object": build_customer_payload(explanation_object),
            "draft_explanation": str(draft_text or ""),
            "compliance_rules_text": self.rules.raw_text,
            "rules_hint": {
                "forbidden_patterns": self.rules.forbidden_patterns,
                "required_phrases": self.rules.required_phrases,
            },
        }
        return self.renderer.render_with_payload(payload=payload, system_prompt=COMPLIANCE_EDITOR_PROMPT)

    def build_candidates(self, explanation_object: Dict[str, Any]) -> Dict[str, Any]:
        route = "reason_first"
        if self._need_compliance_first(explanation_object):
            route = "compliance_first"

        candidates: List[OrchestratorCandidate] = []
        reason_text = self.renderer.render(explanation_object)
        candidates.append(
            OrchestratorCandidate(
                source="llm_reason",
                text=reason_text,
                route=route,
                note="primary_renderer",
            )
        )

        if route == "compliance_first" or self.rules.has_rules:
            edited = self._compliance_edit(explanation_object, reason_text)
            candidates.append(
                OrchestratorCandidate(
                    source="llm_compliance",
                    text=edited,
                    route=route,
                    note="compliance_editor",
                )
            )

        if self.include_template_expert:
            template_text = render_explanation_template(explanation_object)
            candidates.append(
                OrchestratorCandidate(
                    source="template",
                    text=template_text,
                    route=route,
                    note="deterministic_template",
                )
            )

        return {
            "route": route,
            "rules": self.rules_debug,
            "candidates": [c.__dict__ for c in candidates],
        }

