"""Whisper transcriber for the intelligibility gate.

Returns a callable ``(wav_bytes) -> transcript`` suitable for
``benchmark(..., transcriber=...)``. Loads lazily on first call.
"""

from __future__ import annotations

import io
from typing import Callable, Optional


def whisper_transcriber(model: str = "openai/whisper-large-v3",
                        device: Optional[int] = None) -> Callable[[bytes], str]:
    state: dict = {}

    def _load():
        import torch
        from transformers import pipeline
        dev = device if device is not None else (0 if torch.cuda.is_available() else -1)
        state["asr"] = pipeline("automatic-speech-recognition", model=model, device=dev)

    def transcribe(wav: bytes) -> str:
        if "asr" not in state:
            _load()
        import librosa
        y, _ = librosa.load(io.BytesIO(wav), sr=16000, mono=True)
        out = state["asr"]({"array": y, "sampling_rate": 16000})
        return (out.get("text") or "").strip() if isinstance(out, dict) else str(out).strip()

    return transcribe
