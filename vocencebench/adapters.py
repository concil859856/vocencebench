"""TTS model adapters.

An adapter turns a speech model into the callable the evaluator expects:
``(text, instruction) -> wav bytes``. Any PromptTTS model plugs in by wrapping it here;
several common shapes are provided. Heavy backends import lazily.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Callable, Optional

TTS = Callable[[str, str], bytes]


def from_callable(fn: TTS) -> TTS:
    """Identity wrapper — use any ``(text, instruction) -> wav bytes`` function directly."""
    return fn


class FileTTS:
    """Serve pre-generated audio from a directory keyed by an id you set per call.

    Set :attr:`current_id` before each call (the runner/CLI does this per sample); the
    adapter returns ``<dir>/<current_id>.<ext>``. Useful for evaluating audio that was
    generated offline by any model or provider.
    """

    def __init__(self, directory: str, ext: str = "wav"):
        self.dir = Path(directory)
        self.ext = ext.lstrip(".")
        self.current_id: Optional[str] = None

    def __call__(self, text: str, instruction: str) -> bytes:
        if self.current_id is None:
            raise RuntimeError("FileTTS.current_id must be set before each call")
        return (self.dir / f"{self.current_id}.{self.ext}").read_bytes()


class OpenAICompatTTS:
    """Call an OpenAI-compatible text-to-speech endpoint (``POST /v1/audio/speech``).

    The voice instruction is sent in the ``instructions`` field where the provider
    supports it (e.g. instruction-steerable TTS models).
    """

    def __init__(self, base_url: str, model: str, voice: str = "",
                 api_key: str = "", response_format: str = "wav", timeout: float = 120.0,
                 send_instruction: bool = True):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.voice = voice
        self.api_key = api_key
        self.response_format = response_format
        self.timeout = timeout
        self.send_instruction = send_instruction

    def __call__(self, text: str, instruction: str) -> bytes:
        import httpx
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        payload = {"model": self.model, "input": text, "response_format": self.response_format}
        if self.voice:
            payload["voice"] = self.voice
        if self.send_instruction and instruction:
            payload["instructions"] = instruction
        with httpx.Client(base_url=self.base_url, timeout=self.timeout, headers=headers) as c:
            r = c.post("/v1/audio/speech", json=payload)
            r.raise_for_status()
            return r.content


class Qwen3TTSAdapter:
    """Adapter for a local qwen-tts voice-design checkpoint directory.

    Loads the model once and renders ``instruction`` as the voice design and ``text`` as
    the content. Requires the ``qwen-tts`` package and the model weights on disk.
    """

    def __init__(self, model_dir: str, language: str = "English", dtype: str = "bfloat16"):
        self.model_dir = str(model_dir)
        self.language = language
        self.dtype = dtype
        self._engine = None

    def _ensure(self):
        if self._engine is None:
            import torch
            from qwen_tts import Qwen3TTSModel
            td = getattr(torch, self.dtype, torch.float32)
            self._engine = Qwen3TTSModel.from_pretrained(
                pretrained_model_name_or_path=self.model_dir,
                device_map="cuda:0" if torch.cuda.is_available() else "cpu",
                dtype=td, attn_implementation="sdpa",
            )
        return self._engine

    def __call__(self, text: str, instruction: str) -> bytes:
        import numpy as np
        import soundfile as sf
        waves, sr = self._ensure().generate_voice_design(
            text=text, language=self.language, instruct=instruction or "Neutral delivery.")
        wave = np.asarray(waves[0], dtype=np.float32)
        if wave.ndim > 1:
            wave = wave.mean(axis=1)
        buf = io.BytesIO()
        sf.write(buf, wave, int(sr), format="WAV", subtype="PCM_16")
        return buf.getvalue()
