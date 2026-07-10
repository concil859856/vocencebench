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
    def __init__(self, model: str = "gemini-3.1-pro", api_key: Optional[str] = None):
        self.model = model
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self._client = None

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

    def compare(self, parts: PromptParts, audio_a: bytes, audio_b: bytes,
                *, temperature: float) -> Dict[str, Any]:
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
            system_instruction=parts.system,
            temperature=temperature,
            response_mime_type="application/json",
        )
        resp = client.models.generate_content(model=self.model, contents=contents, config=config)
        text = resp.text or ""
        obj = extract_json(text) or salvage_verdict(text)
        if obj is None:
            raise ValueError(f"Gemini judge returned no parseable JSON: {text[:200]!r}")
        out = parse_verdict(obj)
        out["raw"] = text
        return out
