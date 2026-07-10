"""Audio helpers — normalise arbitrary audio into WAV bytes and base64."""

from __future__ import annotations

import base64
import io
from typing import Union

Audio = Union[bytes, str]  # WAV/FLAC/... bytes, or a filesystem path


def to_wav_bytes(audio: Audio, sample_rate: int = 24000) -> bytes:
    """Return 16-bit mono WAV bytes for ``audio`` (bytes or a path).

    Bytes that already look like a WAV are returned unchanged; anything else is decoded
    and re-encoded via soundfile.
    """
    if isinstance(audio, str):
        with open(audio, "rb") as fh:
            data = fh.read()
    else:
        data = audio
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return data
    import numpy as np
    import soundfile as sf

    arr, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    if getattr(arr, "ndim", 1) > 1:
        arr = arr.mean(axis=1)
    buf = io.BytesIO()
    sf.write(buf, arr, sr or sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def b64(audio: Audio) -> str:
    return base64.b64encode(to_wav_bytes(audio)).decode("ascii")
