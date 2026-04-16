from __future__ import annotations

import json
from typing import Any, Dict, Optional

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None


SYSTEM_PROMPT = (
    "You are a financial recommendation explanation renderer. "
    "You MUST ONLY use the provided explanation object. "
    "Do NOT add external knowledge or unstated assumptions. "
    "Do NOT hallucinate. "
    "Do NOT generalize beyond provided facts. "
    "Output format exactly:\n"
    "[Reason]\n- ...\n\n"
    "[Warning]\n- ...\n\n"
    "[Comparison]\n- ...\n\n"
    "[Simple Summary]\n- ..."
)


class OpenAILLMRenderer:
    """Render grounded explanation objects via OpenAI API.

    This renderer is constrained to verbalization only; reasoning facts must already
    be contained in the explanation object.
    """

    def __init__(
        self,
        model: str = "gpt-5-mini",
        api_key: Optional[str] = None,
    ) -> None:
        if OpenAI is None:
            raise ImportError("openai package is not installed. Install with: pip install openai")
        self.model = model
        self.client = OpenAI(api_key=api_key)

    def render(self, explanation_object: Dict[str, Any]) -> str:
        payload = json.dumps(explanation_object, ensure_ascii=False)

        try:
            resp = self.client.responses.create(
                model=self.model,
                input=[
                    {
                        "role": "system",
                        "content": [{"type": "input_text", "text": SYSTEM_PROMPT}],
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
        resp2 = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": payload},
            ],
            temperature=0,
        )
        return (resp2.choices[0].message.content or "").strip()
