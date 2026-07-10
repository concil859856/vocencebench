"""Model-free acoustic probes: pace, loudness, pitch.

These need only the waveform (librosa/numpy), no downloaded model, so they are cheap
and fully deterministic. Band thresholds are sensible English defaults and are meant to
be recalibrated to a corpus/voice — pass overrides to the constructor.
"""

from __future__ import annotations

import io
from typing import List, Optional, Sequence, Tuple

from vocencebench.probes.base import bucket, ordinal_score
from vocencebench.schema import ProbeResult, Sample


def _load(audio: bytes, sr: int = 16000):
    import librosa
    import numpy as np
    y, _sr = librosa.load(io.BytesIO(audio), sr=sr, mono=True)
    return y, sr, np, librosa


class PaceProbe:
    """Speaking rate = words / speaking-duration, bucketed slow/moderate/fast."""

    trait = "pace"
    order = ("slow", "moderate", "fast")

    def __init__(self, edges: Sequence[Tuple[float, str]] = ((2.2, "slow"), (3.2, "moderate"), (1e9, "fast"))):
        self.edges = edges

    def score(self, sample: Sample, audio: bytes) -> Optional[ProbeResult]:
        req = sample.traits.get(self.trait)
        if not req:
            return None
        y, sr, np, librosa = _load(audio)
        # Voiced/energetic duration only (trim leading/trailing silence).
        yt, _ = librosa.effects.trim(y, top_db=30)
        dur = max(len(yt) / sr, 1e-3)
        words = max(len(sample.text.split()), 1)
        wps = words / dur
        measured = bucket(wps, self.edges)
        return ProbeResult(
            trait=self.trait, requested=req, measured=measured,
            matched=(measured == req), score=ordinal_score(req, measured, self.order),
            detail={"words_per_sec": round(wps, 3), "duration_s": round(dur, 3)},
        )


class LoudnessProbe:
    """RMS level in dBFS, bucketed quiet/normal/loud."""

    trait = "loudness"
    order = ("quiet", "normal", "loud")

    def __init__(self, edges: Sequence[Tuple[float, str]] = ((-30.0, "quiet"), (-18.0, "normal"), (1e9, "loud"))):
        self.edges = edges

    def score(self, sample: Sample, audio: bytes) -> Optional[ProbeResult]:
        req = sample.traits.get(self.trait)
        if not req:
            return None
        y, sr, np, librosa = _load(audio)
        rms = float(np.sqrt(np.mean(y.astype("float64") ** 2)) + 1e-9)
        dbfs = 20.0 * np.log10(rms)
        measured = bucket(dbfs, self.edges)
        return ProbeResult(
            trait=self.trait, requested=req, measured=measured,
            matched=(measured == req), score=ordinal_score(req, measured, self.order),
            detail={"dbfs": round(float(dbfs), 2)},
        )


class PitchProbe:
    """Median voiced F0 in Hz, bucketed low/medium/high.

    Note: absolute F0 is confounded with speaker gender; for gendered corpora prefer
    within-voice relative bands. Kept simple here and configurable.
    """

    trait = "pitch"
    order = ("low", "medium", "high")

    def __init__(self, edges: Sequence[Tuple[float, str]] = ((140.0, "low"), (220.0, "medium"), (1e9, "high"))):
        self.edges = edges

    def score(self, sample: Sample, audio: bytes) -> Optional[ProbeResult]:
        req = sample.traits.get(self.trait)
        if not req:
            return None
        y, sr, np, librosa = _load(audio)
        f0, voiced, _ = librosa.pyin(y, fmin=65, fmax=400, sr=sr)
        vals = f0[~np.isnan(f0)] if f0 is not None else np.array([])
        if vals.size == 0:
            return ProbeResult(trait=self.trait, requested=req, measured=None,
                               score=0.0, detail={"f0_median": None})
        med = float(np.median(vals))
        measured = bucket(med, self.edges)
        return ProbeResult(
            trait=self.trait, requested=req, measured=measured,
            matched=(measured == req), score=ordinal_score(req, measured, self.order),
            detail={"f0_median": round(med, 1)},
        )
