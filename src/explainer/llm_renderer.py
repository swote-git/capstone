from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from common.env import load_dotenv_file

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None


DEFAULT_SYSTEM_PROMPT = (
    "당신은 금융 추천 결과를 설명하는 렌더러입니다. "
    "반드시 제공된 explanation object의 정보만 사용하세요. "
    "외부 지식, 추정, 일반론을 추가하지 마세요. "
    "근거 없는 문장을 만들지 마세요. "
    "주어진 사실을 넘어 과장하거나 단정하지 마세요. "
    "모든 문장은 explanation object의 항목과 직접 대응되어야 합니다. "
    "특히 user_profile_detail, recommended_product_detail, user_source_data, product_source_data, reason_signals에 있는 수치/사실을 우선 활용해 "
    "추천 이유를 구체적으로 작성하세요. "
    "출력 형식은 아래와 정확히 동일하게 작성하세요:\n"
    "[상품 정보 요약]\n- ...\n\n"
    "[추천 이유]\n- ...\n\n"
    "[유의사항]\n- ...\n\n"
    "[대안 비교]\n- ...\n\n"
    "[한줄 요약]\n- ..."
)


def _load_system_prompt(prompt_path: Optional[Path] = None) -> str:
    if prompt_path is not None:
        candidate_paths = [prompt_path]
    else:
        # Primary: explain.txt / Backward compatibility: prompt.txt
        candidate_paths = [
            Path(__file__).with_name("explain.txt"),
            Path(__file__).with_name("prompt.txt"),
        ]

    for path in candidate_paths:
        try:
            text = path.read_text(encoding="utf-8").strip()
            if text:
                return text
        except Exception:
            continue
    return DEFAULT_SYSTEM_PROMPT


def _to_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float):
        # Keep readability while preserving enough precision for user-facing explanation.
        return round(value, 2)
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    return value


def _build_customer_payload(explanation_object: Dict[str, Any]) -> Dict[str, Any]:
    user_summary = explanation_object.get("user_summary", {})
    product = explanation_object.get("recommended_product", {})
    detail = explanation_object.get("recommended_product_detail", {})
    comparison = explanation_object.get("comparison", {})
    warnings = explanation_object.get("warnings", [])
    ranking_context = explanation_object.get("ranking_context", {})

    # Remove raw internal score maps for customer-facing rendering.
    product_detail = {
        "horizon": detail.get("horizon"),
        "complexity": detail.get("complexity"),
        "principal_variation": detail.get("principal_variation"),
        "product_meta": detail.get("product_meta", {}),
        "source_highlights": detail.get("product_source_data", {}),
    }

    payload = {
        "user_summary": user_summary,
        "recommended_product": product,
        "recommended_product_detail": product_detail,
        "model_reasons": explanation_object.get("model_reasons", []),
        "comparison": comparison,
        "warnings": warnings,
        "ranking_context": ranking_context,
        "explanation_policy": explanation_object.get("explanation_policy", {}),
    }
    return _to_jsonable(payload)


class OpenAILLMRenderer:
    """Render grounded explanation objects via OpenAI API.

    This renderer is constrained to verbalization only; reasoning facts must already
    be contained in the explanation object.
    """

    def __init__(
        self,
        model: str = "gpt-5-mini",
        api_key: Optional[str] = None,
        prompt_path: Optional[Path] = None,
        timeout_seconds: float = 30.0,
        max_retries: int = 1,
    ) -> None:
        if OpenAI is None:
            raise ImportError("openai package is not installed. Install with: pip install openai")
        if api_key is None and not os.getenv("OPENAI_API_KEY"):
            load_dotenv_file()
        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = OpenAI(
            api_key=resolved_api_key,
            timeout=float(timeout_seconds),
            max_retries=int(max_retries),
        )
        self.system_prompt = _load_system_prompt(prompt_path)
        self.last_debug: Dict[str, Any] = {}

    def _request_text(self, system_prompt: str, payload_text: str) -> str:
        self.last_debug = {"stage": "start", "responses_error": None, "chat_error": None, "output_len": 0}
        try:
            resp = self.client.responses.create(
                model=self.model,
                input=[
                    {
                        "role": "system", "content": [{"type": "input_text", "text": system_prompt}],
                    },
                    {
                        "role": "user", "content": [{"type": "input_text", "text": payload_text}],
                    },
                ],
            )
            text = (getattr(resp, "output_text", None) or "").strip()
            if text:
                self.last_debug = {"stage": "responses_ok", "responses_error": None, "chat_error": None, "output_len": len(text)}
                return text
            self.last_debug["stage"] = "responses_empty"
        except Exception as e:
            self.last_debug["responses_error"] = f"responses.create_failed: {type(e).__name__}: {str(e)[:300]}"
            self.last_debug["stage"] = "responses_error"

        # Backward-compatible fallback for SDK/API variants.
        try:
            resp2 = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": payload_text},
                ],
            )
            text2 = (resp2.choices[0].message.content or "").strip()
            if text2:
                self.last_debug = {"stage": "chat_ok", "responses_error": self.last_debug.get("responses_error"), "chat_error": None, "output_len": len(text2)}
                return text2
            self.last_debug["chat_error"] = "chat_empty"
            self.last_debug["stage"] = "chat_empty"
            return ""
        except Exception as e:
            self.last_debug["chat_error"] = f"chat.create_failed: {type(e).__name__}: {str(e)[:300]}"
            self.last_debug["stage"] = "chat_error"
            return ""

    def render_with_payload(
        self,
        payload: Dict[str, Any],
        system_prompt: Optional[str] = None,
    ) -> str:
        payload_text = json.dumps(_to_jsonable(payload), ensure_ascii=False)
        prompt_text = system_prompt or self.system_prompt
        return self._request_text(prompt_text, payload_text)

    def render(self, explanation_object: Dict[str, Any]) -> str:
        customer_payload = _build_customer_payload(explanation_object)
        return self.render_with_payload(customer_payload, system_prompt=self.system_prompt)


def build_customer_payload(explanation_object: Dict[str, Any]) -> Dict[str, Any]:
    return _build_customer_payload(explanation_object)
