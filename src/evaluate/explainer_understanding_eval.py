from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from common.env import load_dotenv_file
from explainer.common import FORBIDDEN_PATTERNS

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None


QUESTIONS: List[str] = [
    "Q1. 이 상품은 위험한가요, 안정적인가요?",
    "Q2. 왜 이 상품이 추천되었나요?",
    "Q3. 이 상품의 가장 큰 장점은 무엇인가요?",
    "Q4. 이 상품에서 주의해야 할 점은 무엇인가요?",
    "Q5. 다른 상품과 비교하면 무엇이 다른가요?",
]

DEFAULT_SIMULATOR_PROMPT = (
    "당신은 금융상품 추천을 받은 사용자입니다.\n"
    "[사용자 특성]\n"
    "- 금융 지식이 낮습니다.\n"
    "- 모르면 반드시 '모르겠습니다'라고 답합니다.\n"
    "[규칙]\n"
    "1) 제공된 정보만 사용\n"
    "2) 외부 지식 금지\n"
    "3) 추측 금지\n"
    "4) 질문별로 짧게 답변\n"
    "5) JSON으로만 응답: {\"Q1\":\"...\",\"Q2\":\"...\",\"Q3\":\"...\",\"Q4\":\"...\",\"Q5\":\"...\"}"
)

DEFAULT_EVALUATOR_PROMPT = (
    "당신은 금융 설명 이해도 평가자입니다.\n"
    "사용자 답변과 정답을 비교해 질문별 0/1 점수를 부여하세요.\n"
    "[채점 기준]\n"
    "- 1점: 의미상 정답\n"
    "- 0점: 오답/불명확/근거 없는 추론\n"
    "[출력]\n"
    "반드시 JSON으로만 응답: "
    "{\"Q1\":0,\"Q2\":0,\"Q3\":0,\"Q4\":0,\"Q5\":0,\"total\":0,\"misinterpretations\":[]}"
)


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _load_prompt(path: Optional[Path], default: str) -> str:
    if path is None:
        return default
    try:
        txt = path.read_text(encoding="utf-8").strip()
        return txt or default
    except Exception:
        return default


def _normalize(text: str) -> str:
    t = (text or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    return t


def _extract_json(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        return {}
    try:
        parsed = json.loads(m.group(0))
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return {}
    return {}


def _contains_any(text: str, keywords: Sequence[str]) -> bool:
    t = _normalize(text)
    return any(k.lower() in t for k in keywords if k)


def _first_bullet_keywords(lines: Sequence[str]) -> List[str]:
    joined = " ".join([str(x) for x in lines if x])
    keywords = []
    for k in ["위험", "유동성", "복잡도", "기간", "금액", "디지털", "상품군", "원금", "수익"]:
        if k in joined:
            keywords.append(k)
    return keywords


def build_ground_truth(explanation_object: Dict[str, Any]) -> Dict[str, str]:
    product = explanation_object.get("recommended_product", {})
    detail = explanation_object.get("recommended_product_detail", {})
    warnings = explanation_object.get("warnings", [])
    reasons = explanation_object.get("model_reasons", [])
    comparison = explanation_object.get("comparison", {})

    risk_label = str(product.get("risk", "보통"))
    principal_var = bool(detail.get("principal_variation", False))
    high_risk = risk_label in {"높음", "매우 높음"} or principal_var

    q1 = "변동 위험이 있는 편입니다." if high_risk else "상대적으로 안정적인 편입니다."
    q2 = " ".join(reasons[:2]).strip() or "추천 근거 정보가 충분하지 않습니다."
    q3 = (reasons[0] if reasons else "사용자 특성과 상품 특성의 적합도가 장점입니다.").strip()
    q4 = (warnings[0] if warnings else "주의사항 정보가 없습니다.").strip()
    q5 = (
        f"대안 {comparison.get('alternative', '')} 대비 차이: {comparison.get('difference', '')}".strip()
    )

    return {
        "Q1": q1,
        "Q2": q2,
        "Q3": q3,
        "Q4": q4,
        "Q5": q5,
    }


def _ground_truth_keywords(explanation_object: Dict[str, Any]) -> Dict[str, List[str]]:
    product = explanation_object.get("recommended_product", {})
    detail = explanation_object.get("recommended_product_detail", {})
    comparison = explanation_object.get("comparison", {})
    reasons = explanation_object.get("model_reasons", [])
    warnings = explanation_object.get("warnings", [])

    risk_label = str(product.get("risk", "보통"))
    principal_var = bool(detail.get("principal_variation", False))
    high_risk = risk_label in {"높음", "매우 높음"} or principal_var
    q1 = ["위험", "변동", "손실"] if high_risk else ["안정", "낮은 위험", "변동 적음"]

    q2 = _first_bullet_keywords(reasons[:2]) or ["적합", "추천 이유"]
    q3 = _first_bullet_keywords(reasons[:1]) or ["장점", "적합"]
    q4 = _first_bullet_keywords(warnings[:1]) or ["주의", "확인"]

    q5 = [str(comparison.get("alternative", "")), "차이", "위험", "수익", "원금", "유동성"]

    return {"Q1": q1, "Q2": q2, "Q3": q3, "Q4": q4, "Q5": q5}


def _rule_evaluate_answers(
    explanation_object: Dict[str, Any],
    answers: Dict[str, str],
) -> Tuple[Dict[str, int], List[str]]:
    keys = _ground_truth_keywords(explanation_object)
    scores: Dict[str, int] = {}
    mis: List[str] = []

    q1 = str(answers.get("Q1", ""))
    risk = explanation_object.get("recommended_product", {}).get("risk", "보통")
    principal_var = bool(
        explanation_object.get("recommended_product_detail", {}).get("principal_variation", False)
    )
    high_risk = risk in {"높음", "매우 높음"} or principal_var
    if high_risk:
        scores["Q1"] = int(_contains_any(q1, ["위험", "변동", "손실"]))
        if _contains_any(q1, ["안정", "무위험"]):
            mis.append("Q1: 위험 상품을 안정적으로 오해")
    else:
        scores["Q1"] = int(_contains_any(q1, ["안정", "낮은 위험", "변동 적"]))
        if _contains_any(q1, ["위험", "손실 큼"]):
            mis.append("Q1: 안정형 상품을 고위험으로 오해")

    for q in ["Q2", "Q3", "Q4", "Q5"]:
        ans = str(answers.get(q, ""))
        if "모르겠습니다" in ans:
            scores[q] = 0
            continue
        scores[q] = int(_contains_any(ans, keys[q]))

    if _contains_any(str(answers.get("Q4", "")), ["없", "주의할 점 없음", "리스크 없음"]):
        mis.append("Q4: 유의사항 누락/무시")

    alt = str(explanation_object.get("comparison", {}).get("alternative", "")).strip().lower()
    q5 = _normalize(str(answers.get("Q5", "")))
    if alt and alt not in q5:
        mis.append("Q5: 대안 상품군 비교 누락")

    return scores, mis


def _fallback_user_answers(
    recommendation: Dict[str, Any],
    explanation: str,
    explanation_object: Dict[str, Any],
) -> Dict[str, str]:
    product = recommendation.get("recommended_product", {})
    risk = str(product.get("risk", "보통"))
    has_expl = bool((explanation or "").strip())

    if not has_expl:
        q1 = "안정적인 편 같습니다." if risk in {"낮음", "보통"} else "위험할 수 있습니다."
        return {
            "Q1": q1,
            "Q2": "모르겠습니다.",
            "Q3": "모르겠습니다.",
            "Q4": "모르겠습니다.",
            "Q5": "모르겠습니다.",
        }

    # Parse rendered explanation text instead of directly copying ground truth
    # to avoid circularity in fallback mode.
    reason_bullets = _extract_section_bullets(explanation, "추천 이유")
    warning_bullets = _extract_section_bullets(explanation, "유의사항")
    compare_bullets = _extract_section_bullets(explanation, "대안 비교")
    summary_bullets = _extract_section_bullets(explanation, "한줄 요약")

    q1 = "안정적인 편 같습니다." if risk in {"낮음", "보통"} else "변동 위험이 있는 편 같습니다."
    q2 = reason_bullets[0] if reason_bullets else "모르겠습니다."
    q3 = reason_bullets[1] if len(reason_bullets) > 1 else (reason_bullets[0] if reason_bullets else "모르겠습니다.")
    q4 = warning_bullets[0] if warning_bullets else "모르겠습니다."
    q5 = compare_bullets[0] if compare_bullets else (summary_bullets[0] if summary_bullets else "모르겠습니다.")

    return {"Q1": q1, "Q2": q2, "Q3": q3, "Q4": q4, "Q5": q5}


def _extract_section_bullets(text: str, section_title: str) -> List[str]:
    if not text:
        return []
    pattern = rf"\[{re.escape(section_title)}\]([\s\S]*?)(?:\n\[[^\]]+\]|\Z)"
    m = re.search(pattern, text)
    if not m:
        return []
    body = m.group(1)
    lines = []
    for ln in body.splitlines():
        s = ln.strip()
        if s.startswith("-"):
            lines.append(s[1:].strip())
    return lines


@dataclass
class _LLMWorker:
    model: str
    prompt: str
    api_key: Optional[str] = None

    def __post_init__(self) -> None:
        if self.api_key is None and not os.getenv("OPENAI_API_KEY"):
            load_dotenv_file()
        resolved_api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=resolved_api_key) if OpenAI is not None else None

    def run_json(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.client is None:
            return {}
        text_payload = json.dumps(payload, ensure_ascii=False)
        try:
            resp = self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": [{"type": "input_text", "text": self.prompt}]},
                    {"role": "user", "content": [{"type": "input_text", "text": text_payload}]},
                ],
            )
            raw = (getattr(resp, "output_text", None) or "").strip()
            parsed = _extract_json(raw)
            if parsed:
                return parsed
        except Exception:
            pass
        return {}


class ExplainerUnderstandingEvaluator:
    def __init__(
        self,
        use_llm_user_simulator: bool = False,
        use_llm_evaluator: bool = False,
        simulator_model: str = "gpt-5-mini",
        evaluator_model: str = "gpt-5-mini",
        simulator_prompt_path: Optional[Path] = None,
        evaluator_prompt_path: Optional[Path] = None,
        api_key: Optional[str] = None,
    ) -> None:
        sim_prompt = _load_prompt(simulator_prompt_path, DEFAULT_SIMULATOR_PROMPT)
        eval_prompt = _load_prompt(evaluator_prompt_path, DEFAULT_EVALUATOR_PROMPT)

        self.user_simulator = (
            _LLMWorker(model=simulator_model, prompt=sim_prompt, api_key=api_key)
            if use_llm_user_simulator
            else None
        )
        self.answer_evaluator = (
            _LLMWorker(model=evaluator_model, prompt=eval_prompt, api_key=api_key)
            if use_llm_evaluator
            else None
        )
        self.simulator_prompt = sim_prompt
        self.evaluator_prompt = eval_prompt

    def _simulate_answers(
        self,
        recommendation_payload: Dict[str, Any],
        explanation_text: str,
        explanation_object: Dict[str, Any],
    ) -> Dict[str, str]:
        if self.user_simulator is not None:
            payload = {
                "recommendation": recommendation_payload,
                "explanation": explanation_text,
                "questions": QUESTIONS,
            }
            out = self.user_simulator.run_json(payload)
            if out:
                return {f"Q{i}": str(out.get(f"Q{i}", "모르겠습니다.")) for i in range(1, 6)}
        return _fallback_user_answers(recommendation_payload, explanation_text, explanation_object)

    def _score_answers(
        self,
        ground_truth: Dict[str, str],
        answers: Dict[str, str],
        explanation_object: Dict[str, Any],
    ) -> Tuple[Dict[str, int], List[str]]:
        if self.answer_evaluator is not None:
            payload = {"ground_truth": ground_truth, "answers": answers}
            out = self.answer_evaluator.run_json(payload)
            if out:
                scores = {f"Q{i}": int(out.get(f"Q{i}", 0)) for i in range(1, 6)}
                mis = [str(x) for x in out.get("misinterpretations", []) if str(x).strip()]
                return scores, mis
        return _rule_evaluate_answers(explanation_object, answers)

    def _quality_scores(
        self,
        explanation_text: str,
        explanation_object: Dict[str, Any],
        verification: Dict[str, Any],
    ) -> Dict[str, float]:
        user = explanation_object.get("user_summary", {})
        product = explanation_object.get("recommended_product", {})
        detail = explanation_object.get("recommended_product_detail", {})

        risk_hit = 1.0 if str(user.get("risk_preference", "")) in explanation_text else 0.0
        liq_hit = 1.0 if str(user.get("liquidity_need", "")) in explanation_text else 0.0
        persona_hit = 1.0 if (risk_hit + liq_hit) >= 1.0 else 0.0
        reason_hits = sum(1 for r in explanation_object.get("model_reasons", []) if str(r) in explanation_text)
        reason_cov = reason_hits / max(1, len(explanation_object.get("model_reasons", [])))
        personalization = _clip01(0.35 * risk_hit + 0.25 * liq_hit + 0.15 * persona_hit + 0.25 * reason_cov)

        prod_tokens = [
            str(product.get("family", "")),
            str(product.get("risk", "")),
            str(product.get("liquidity", "")),
            str(detail.get("horizon", "")),
            str(detail.get("complexity", "")),
        ]
        prod_token_hit = sum(1 for t in prod_tokens if t and t in explanation_text) / max(1, len(prod_tokens))
        fact_ok = 1.0 if bool(verification.get("fact_consistency", False)) else 0.0
        product_grounding = _clip01(0.7 * prod_token_hit + 0.3 * fact_ok)

        jargon_terms = ["샤프", "mdd", "std", "알파", "베타", "duration", "tracking error", "ncf"]
        jargon_count = sum(1 for j in jargon_terms if j in _normalize(explanation_text))
        bullet_lines = [ln.strip()[2:] for ln in explanation_text.splitlines() if ln.strip().startswith("-")]
        avg_len = float(np.mean([len(x) for x in bullet_lines])) if bullet_lines else 0.0
        clarity_penalty = min(0.6, jargon_count * 0.2) + min(0.4, max(0.0, (avg_len - 45.0) / 80.0))
        terminology_clarity = _clip01(1.0 - clarity_penalty)

        forbidden = len(verification.get("forbidden_claims", []))
        has_warning_section = "[유의사항]" in explanation_text
        compliance = 1.0 - min(1.0, 0.5 * forbidden)
        if not has_warning_section:
            compliance -= 0.2
        if float(verification.get("hallucination_rate", 0.0)) > 0:
            compliance -= 0.1
        compliance = _clip01(compliance)

        return {
            "personalization": personalization,
            "product_grounding": product_grounding,
            "terminology_clarity": terminology_clarity,
            "compliance": compliance,
        }

    def evaluate_recommendation(
        self,
        user_id: str,
        recommendation_item: Dict[str, Any],
    ) -> Dict[str, Any]:
        explanation_object = recommendation_item.get("explanation_object", {})
        explanation_text = str(recommendation_item.get("rendered_explanation", ""))
        verification = recommendation_item.get("verification", {})

        rec_payload = {
            "product_id": recommendation_item.get("product_id", ""),
            "score": recommendation_item.get("score", 0.0),
            "recommended_product": explanation_object.get("recommended_product", {}),
            "recommended_product_detail": explanation_object.get("recommended_product_detail", {}),
            "comparison": explanation_object.get("comparison", {}),
            "warnings": explanation_object.get("warnings", []),
        }
        ground_truth = build_ground_truth(explanation_object)

        answers_before = self._simulate_answers(rec_payload, "", explanation_object)
        answers_after = self._simulate_answers(rec_payload, explanation_text, explanation_object)

        score_before, mis_before = self._score_answers(ground_truth, answers_before, explanation_object)
        score_after, mis_after = self._score_answers(ground_truth, answers_after, explanation_object)
        total_before = int(sum(score_before.values()))
        total_after = int(sum(score_after.values()))

        quality = self._quality_scores(explanation_text, explanation_object, verification)
        ug = float(total_after - total_before) / 5.0
        mr = float(len(mis_after)) / 5.0

        return {
            "user_id": str(user_id),
            "product_id": str(recommendation_item.get("product_id", "")),
            "render_source": str(recommendation_item.get("render_source", "unknown")),
            "recommendation_payload": rec_payload,
            "evaluation_prompts": {
                "simulator_prompt": self.simulator_prompt,
                "evaluator_prompt": self.evaluator_prompt,
            },
            "ground_truth": ground_truth,
            "questions": QUESTIONS,
            "answers_before": answers_before,
            "answers_after": answers_after,
            "scores_before": score_before,
            "scores_after": score_after,
            "total_before": total_before,
            "total_after": total_after,
            "misinterpretations_before": mis_before,
            "misinterpretations_after": mis_after,
            "quality_scores": quality,
            "effect_scores": {
                "understanding_gain": ug,
                "misinterpretation_rate": mr,
            },
            "explanation_object": explanation_object,
            "rendered_explanation": explanation_text,
            "verification": verification,
        }

    @staticmethod
    def summarize(records: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        if not records:
            return {
                "personalization": 0.0,
                "product_grounding": 0.0,
                "terminology_clarity": 0.0,
                "compliance": 0.0,
                "understanding_gain": 0.0,
                "misinterpretation_rate": 0.0,
            }

        def mean(path: str) -> float:
            vals = [float(_dig(r, path, 0.0)) for r in records]
            return float(np.mean(vals)) if vals else 0.0

        return {
            "personalization": mean("quality_scores.personalization"),
            "product_grounding": mean("quality_scores.product_grounding"),
            "terminology_clarity": mean("quality_scores.terminology_clarity"),
            "compliance": mean("quality_scores.compliance"),
            "understanding_gain": mean("effect_scores.understanding_gain"),
            "misinterpretation_rate": mean("effect_scores.misinterpretation_rate"),
        }


def _dig(dct: Dict[str, Any], key_path: str, default: Any) -> Any:
    cur: Any = dct
    for k in key_path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def export_jsonl(records: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def compliance_forbidden_hits(text: str) -> List[str]:
    lower = _normalize(text)
    hits = []
    for p in FORBIDDEN_PATTERNS:
        if re.search(p, lower):
            hits.append(p)
    return hits
