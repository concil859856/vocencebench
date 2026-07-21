"""Continuous age probe (regression).

Age is requested either as a target age in years or as a life-stage category
("child", "young adult", "middle-aged adult", "elderly person" and close variants);
categories resolve to a target-years/tolerance pair below. The probe estimates the
speaker's apparent age from the audio (audEERING wav2vec2 age-gender model, which
has a regression head). The desirability is 1.0 within +/- ``tolerance`` years of
the request and falls off linearly to 0 at 2x tolerance.
"""

from __future__ import annotations

import io
from typing import Optional, Tuple

from vocencebench.schema import ProbeResult, Sample

DEFAULT_MODEL = "audeering/wav2vec2-large-robust-24-ft-age-gender"

# Life-stage categories -> (target years, tolerance years). Keys are matched on the
# lowercased request with the words "person"/"speaker"/"voice" stripped.
AGE_CATEGORIES = {
    "child": (10.0, 6.0),
    "kid": (10.0, 6.0),
    "teenager": (16.0, 4.0),
    "teen": (16.0, 4.0),
    "young adult": (25.0, 8.0),
    "adult": (35.0, 12.0),
    "middle-aged adult": (47.0, 10.0),
    "middle aged adult": (47.0, 10.0),
    "middle-aged": (47.0, 10.0),
    "middle aged": (47.0, 10.0),
    "elderly": (72.0, 12.0),
    "old": (72.0, 12.0),
    "senior": (72.0, 12.0),
}


def resolve_age_request(req: object) -> Optional[Tuple[float, Optional[float]]]:
    """Return (target_years, tolerance_override) for a numeric or categorical
    request, or None if the request is unrecognized. Numeric requests keep the
    probe's default tolerance (override None)."""
    try:
        return float(req), None  # type: ignore[arg-type]
    except (TypeError, ValueError):
        pass
    text = str(req).strip().lower()
    for noise in ("person", "speaker", "voice"):
        text = text.replace(noise, "")
    text = " ".join(text.split())
    if text in AGE_CATEGORIES:
        years, tol = AGE_CATEGORIES[text]
        return years, tol
    return None


def _build_model_class():
    """Define the audEERING age-gender head (published on the model card)."""
    import torch.nn as nn
    from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel

    class _Head(nn.Module):
        def __init__(self, config, num_labels):
            super().__init__()
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.dropout = nn.Dropout(config.final_dropout)
            self.out_proj = nn.Linear(config.hidden_size, num_labels)

        def forward(self, x):
            import torch
            x = self.dropout(x)
            x = torch.tanh(self.dense(x))
            x = self.dropout(x)
            return self.out_proj(x)

    class AgeGenderModel(Wav2Vec2PreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.wav2vec2 = Wav2Vec2Model(config)
            self.age = _Head(config, 1)
            self.gender = _Head(config, 3)
            self.init_weights()

        def forward(self, input_values):
            hidden = self.wav2vec2(input_values)[0].mean(dim=1)
            return self.age(hidden), self.gender(hidden)

    return AgeGenderModel


class AgeProbe:
    trait = "age"

    def __init__(self, model_id: str = DEFAULT_MODEL, tolerance: float = 8.0,
                 device: Optional[int] = None):
        self.model_id = model_id
        self.tolerance = tolerance
        self.device = device
        self._model = None
        self._proc = None

    def _ensure(self):
        if self._model is None:
            import torch
            from transformers import Wav2Vec2Processor
            dev = self.device
            if dev is None:
                dev = 0 if torch.cuda.is_available() else -1
            self._torch = torch
            self._dev = f"cuda:{dev}" if dev >= 0 else "cpu"
            self._proc = Wav2Vec2Processor.from_pretrained(self.model_id)
            self._model = _build_model_class().from_pretrained(self.model_id).to(self._dev).eval()
        return self._model

    def _predict_age_years(self, audio: bytes) -> float:
        import librosa
        y, _ = librosa.load(io.BytesIO(audio), sr=16000, mono=True)
        self._ensure()
        inputs = self._proc(y, sampling_rate=16000, return_tensors="pt")
        with self._torch.no_grad():
            age_norm, _gender = self._model(inputs["input_values"].to(self._dev))
        return float(age_norm.squeeze().cpu()) * 100.0   # model outputs age in [0,1]

    def score(self, sample: Sample, audio: bytes) -> Optional[ProbeResult]:
        req = sample.traits.get(self.trait)
        if not req:
            return None
        resolved = resolve_age_request(req)
        if resolved is None:
            return ProbeResult(trait=self.trait, requested=str(req), measured=None,
                               score=0.0, detail={"error": "unrecognized age request"})
        req_years, tol_override = resolved
        tolerance = tol_override if tol_override is not None else self.tolerance
        try:
            measured = self._predict_age_years(audio)
        except Exception as exc:
            return ProbeResult(trait=self.trait, requested=str(req), measured=None,
                               score=0.0, detail={"error": str(exc)[:160]})
        diff = abs(measured - req_years)
        # 1.0 within tolerance, linear to 0 at 2*tolerance.
        score = max(0.0, 1.0 - max(0.0, diff - tolerance) / tolerance)
        return ProbeResult(
            trait=self.trait, requested=str(req), measured=f"{measured:.0f}",
            matched=(diff <= tolerance), score=round(score, 6),
            detail={"measured_years": round(measured, 1), "requested_years": req_years,
                    "tolerance": tolerance},
        )
