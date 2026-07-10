"""Backend for hosted Gemini models with native audio input.

Uses the ``google-genai`` SDK. The two clips are sent as inline audio parts, ordered
between the prompt's text segments; the model is asked for a JSON verdict.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from vocencebench.judge.base import extract_json, salvage_verdict
from vocencebench.prompts import PromptParts, parse_verdict


class GeminiBackend:
    # Gemini 3.1 Pro standard pricing (USD per token).
    PRICE_IN = 2.0 / 1_000_000
    PRICE_OUT = 12.0 / 1_000_000

    def __init__(self, model: str = "gemini-3.1-pro-preview", api_key: Optional[str] = None):
        self.model = model
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self._client = None
        # Exact token usage accumulated across calls (input, output incl. thinking).
        self.usage = {"calls": 0, "input": 0, "output": 0, "total": 0}

    def cost_usd(self) -> float:
        return round(self.usage["input"] * self.PRICE_IN + self.usage["output"] * self.PRICE_OUT, 6)

    def _track(self, resp) -> None:
        um = getattr(resp, "usage_metadata", None)
        if not um:
            return
        inp = int(getattr(um, "prompt_token_count", 0) or 0)
        out = int(getattr(um, "candidates_token_count", 0) or 0)
        think = int(getattr(um, "thoughts_token_count", 0) or 0)
        self.usage["calls"] += 1
        self.usage["input"] += inp
        self.usage["output"] += out + think
        self.usage["total"] += int(getattr(um, "total_token_count", 0) or (inp + out + think))

    def _get_client(self):
        if self._client is None:
            try:
                from google import genai
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "The Gemini judge needs the 'google-genai' package: "
                    "pip install 'vocencebench[gemini]'"
                ) from exc
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def raw(self, parts: PromptParts, audio_a: bytes, audio_b: bytes,
            *, temperature: float) -> str:
        from google.genai import types
        client = self._get_client()
        contents = [
            parts.intro,
            types.Part.from_bytes(data=audio_a, mime_type="audio/wav"),
            parts.mid,
            types.Part.from_bytes(data=audio_b, mime_type="audio/wav"),
            parts.outro,
        ]
        config = types.GenerateContentConfig(
            system_instruction=parts.system, temperature=temperature,
            response_mime_type="application/json",
        )
        resp = client.models.generate_content(model=self.model, contents=contents, config=config)
        self._track(resp)
        return resp.text or ""

    def compare(self, parts: PromptParts, audio_a: bytes, audio_b: bytes,
                *, temperature: float) -> Dict[str, Any]:
        text = self.raw(parts, audio_a, audio_b, temperature=temperature)
        obj = extract_json(text) or salvage_verdict(text)
        if obj is None:
            raise ValueError(f"Gemini judge returned no parseable JSON: {text[:200]!r}")
        out = parse_verdict(obj)
        out["raw"] = text
        return out
