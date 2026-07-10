"""Backend for any OpenAI-compatible chat endpoint that accepts audio input.

Works with a local server (e.g. an audio-capable model served behind an
OpenAI-compatible API) or a hosted OpenAI audio model. The two clips are sent as
``input_audio`` content parts, interleaved with the prompt's text segments.
"""

from __future__ import annotations

import base64
from typing import Any, Dict

import httpx

from vocencebench.judge.base import extract_json
from vocencebench.prompts import PromptParts, parse_verdict


class OpenAICompatBackend:
    def __init__(self, base_url: str, model: str, api_key: str = "", timeout: float = 180.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def _messages(self, parts: PromptParts, audio_a: bytes, audio_b: bytes) -> list:
        def audio_part(wav: bytes) -> dict:
            return {"type": "input_audio",
                    "input_audio": {"data": base64.b64encode(wav).decode("ascii"),
                                    "format": "wav"}}
        return [
            {"role": "system", "content": parts.system},
            {"role": "user", "content": [
                {"type": "text", "text": parts.intro},
                audio_part(audio_a),
                {"type": "text", "text": parts.mid},
                audio_part(audio_b),
                {"type": "text", "text": parts.outro},
            ]},
        ]

    def compare(self, parts: PromptParts, audio_a: bytes, audio_b: bytes,
                *, temperature: float) -> Dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        payload = {
            "model": self.model,
            "messages": self._messages(parts, audio_a, audio_b),
            "temperature": temperature,
            "max_tokens": 1400,
        }
        with httpx.Client(base_url=self.base_url, timeout=self.timeout, headers=headers) as c:
            r = c.post("/v1/chat/completions", json=payload)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
        obj = extract_json(content)
        if obj is None:
            raise ValueError(f"judge returned no parseable JSON: {content[:200]!r}")
        out = parse_verdict(obj)
        out["raw"] = content
        return out
