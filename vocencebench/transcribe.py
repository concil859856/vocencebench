"""Transcribers for the intelligibility gate.

Returns a callable ``(wav_bytes) -> transcript`` suitable for
``benchmark(..., transcriber=...)``. Loads lazily on first call.
"""

from __future__ import annotations

import io
import os
from typing import Callable, Optional


def whisper_transcriber(model: str = "openai/whisper-large-v3",
                        device: Optional[int] = None) -> Callable[[bytes], str]:
    """Free, local, self-hosted -- but literal/phonetic: it transcribes spoken URLs,
    currency, and phone numbers as heard ('https slash slash...') rather than in the
    written form the target text uses, which can inflate WER on such content even when
    the audio itself is correct. See gemini_transcriber for the more robust option."""
    state: dict = {}

    def _load():
        import torch
        from transformers import pipeline
        dev = device if device is not None else (0 if torch.cuda.is_available() else -1)
        # chunk_length_s enables Whisper's long-form chunk+stitch path; without it, any
        # clip over the native 30s window raises rather than transcribing.
        state["asr"] = pipeline("automatic-speech-recognition", model=model, device=dev,
                                chunk_length_s=30)

    def transcribe(wav: bytes) -> str:
        if "asr" not in state:
            _load()
        import librosa
        y, _ = librosa.load(io.BytesIO(wav), sr=16000, mono=True)
        out = state["asr"]({"array": y, "sampling_rate": 16000})
        return (out.get("text") or "").strip() if isinstance(out, dict) else str(out).strip()

    return transcribe


_GEMINI_STT_SYSTEM = (
    "Transcribe this audio clip verbatim, exactly as spoken, using standard written "
    "formatting for any numbers, dates, phone numbers, and URLs (i.e. write them the way "
    "a human would type them, not phonetically spelled out). Return ONLY the transcript, "
    "no commentary."
)


def gemini_transcriber(model: str = "gemini-3.1-pro-preview",
                       api_key: Optional[str] = None,
                       retries: int = 3) -> Callable[[bytes], str]:
    """Hosted audio-LLM transcriber. Unlike Whisper, it normalises spoken numbers/
    URLs/currency/phone numbers back to written form, avoiding a systematic WER
    inflation Whisper shows on exactly that content. Costs a Gemini API call per
    clip; use whisper_transcriber for a free/local/offline alternative."""
    state: dict = {}

    def _load():
        from google import genai
        state["client"] = genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY"))

    def transcribe(wav: bytes) -> str:
        from google.genai import types
        if "client" not in state:
            _load()
        for _ in range(retries):
            resp = state["client"].models.generate_content(
                model=model,
                contents=[_GEMINI_STT_SYSTEM, types.Part.from_bytes(data=wav, mime_type="audio/wav")],
                config=types.GenerateContentConfig(temperature=0.0))
            text = (resp.text or "").strip()
            if text:
                return text
        return ""

    return transcribe
