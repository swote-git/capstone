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
    "특히 user_profile_detail, recommended_product_detail, reason_signals에 있는 수치/사실을 우선 활용해 "
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

    def render(self, explanation_object: Dict[str, Any]) -> str:
        payload = json.dumps(explanation_object, ensure_ascii=False)

        try:
            resp = self.client.responses.create(
                model=self.model,
                input=[
                    {
                        "role": "system",
                        "content": [{"type": "input_text", "text": self.system_prompt}],
                    },
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": payload}],
                    },
                ],
            )
            text = (getattr(resp, "output_text", None) or "").strip()
            if text:
                return text
        except Exception:
            pass

        # Backward-compatible fallback for SDK/API variants.
        try:
            resp2 = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": payload},
                ],
                temperature=0,
            )
            return (resp2.choices[0].message.content or "").strip()
        except Exception:
            return ""
