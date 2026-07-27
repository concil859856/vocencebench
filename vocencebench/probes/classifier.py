"""Classifier-based probes: gender, emotion, accent, age.

These wrap trained audio classifiers to read a categorical trait directly from the
signal. Models load lazily and can run on CPU or GPU; each is optional and configurable
(swap the ``model_id`` for a better/faster checkpoint as they appear). Model ids are
defaults, not hard requirements — validate a probe's own agreement on natural speech
before trusting it on synthetic audio.
"""

from __future__ import annotations

import io
from typing import Dict, List, Optional, Sequence

from vocencebench.probes.base import exact_score, ordinal_score
from vocencebench.schema import ProbeResult, Sample


class HFClassifierProbe:
    """Generic probe over a Hugging Face ``audio-classification`` pipeline.

    ``label_map`` maps model labels (lowercased, matched by substring) to canonical
    trait values; unmapped labels are ignored. Ordinal traits get ±1 partial credit.
    """

    def __init__(self, trait: str, model_id: str, label_map: Dict[str, str],
                 *, order: Optional[Sequence[str]] = None, device: Optional[int] = None,
                 top_k: int = 5):
        self.trait = trait
        self.model_id = model_id
        self.label_map = {k.lower(): v for k, v in label_map.items()}
        self.order = tuple(order) if order else ()
        self.device = device
        self.top_k = top_k
        self._pipe = None

    def _ensure(self):
        if self._pipe is None:
            from transformers import pipeline
            dev = self.device
            if dev is None:
                try:
                    import torch
                    dev = 0 if torch.cuda.is_available() else -1
                except Exception:
                    dev = -1
            self._pipe = pipeline("audio-classification", model=self.model_id, device=dev)
        return self._pipe

    def _to_value(self, label: str) -> Optional[str]:
        lab = label.lower()
        for needle, value in self.label_map.items():
            if needle in lab:
                return value
        return None

    def score(self, sample: Sample, audio: bytes) -> Optional[ProbeResult]:
        req = sample.traits.get(self.trait)
        if not req:
            return None
        import librosa
        y, _ = librosa.load(io.BytesIO(audio), sr=16000, mono=True)
        try:
            preds = self._ensure()({"array": y, "sampling_rate": 16000}, top_k=self.top_k)
        except Exception as exc:  # a missing/incompatible model must not abort the sample
            return ProbeResult(trait=self.trait, requested=req, measured=None, score=0.0,
                               detail={"error": str(exc)[:160]})
        measured = None
        conf = 0.0
        for p in preds:  # highest-scoring mappable label
            v = self._to_value(p["label"])
            if v is not None:
                measured, conf = v, float(p["score"])
                break
        if measured is None:
            return ProbeResult(trait=self.trait, requested=req, measured=None, score=0.0,
                               detail={"top": preds[:2]})
        s = (ordinal_score(req, measured, self.order) if self.order
             else exact_score(req, measured))
        return ProbeResult(trait=self.trait, requested=req, measured=measured,
                           matched=(measured == req), score=s,
                           detail={"confidence": round(conf, 3)})


def GenderProbe(model_id: str = "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech",
                **kw) -> HFClassifierProbe:
    return HFClassifierProbe("gender", model_id,
                             {"female": "female", "male": "male"}, **kw)


def EmotionProbe(model_id: str = "superb/hubert-large-superb-er", **kw) -> HFClassifierProbe:
    """Four-class emotion probe (angry/happy/sad/neutral).

    Model choice is measured, not assumed: benchmarked against labeled emotional
    speech, this checkpoint reads 46% on a balanced four-way set (25% chance),
    while several widely-used alternatives sat at chance with ~0.13 confidence —
    i.e. they emit a near-uniform distribution and score nothing. Emotions outside
    these four classes have no reliable open classifier and are therefore not part
    of the scored trait vocabulary.
    """
    return HFClassifierProbe("emotion", model_id, {
        "neu": "neutral", "hap": "happy", "sad": "sad", "ang": "angry",
    }, **kw)


def AccentProbe(model_id: str = "dima806/english_accents_classification", **kw) -> HFClassifierProbe:
    """Accent probe over American / British / Indian.

    ``canada`` maps to American: the classifier splits North American English into
    us/canada and assigns most American speech to the canada class, so merging them
    lifts American recall from 24% to 92% on labeled speech. Australian is absent
    from the mapping on purpose — the model never predicts it for genuinely
    Australian speech (0/40), so it cannot be scored fairly.
    """
    return HFClassifierProbe("accent", model_id, {
        "us": "American", "america": "American", "canada": "American", "canadian": "American",
        "england": "British", "british": "British",
        "indian": "Indian", "india": "Indian",
    }, **kw)


def with_classifiers(base: Optional[List] = None, *, gender: bool = True,
                     emotion: bool = True, accent: bool = True, age: bool = True,
                     device: Optional[int] = None) -> List:
    """Append classifier + age probes to (a copy of) ``base`` (defaults to the acoustic set)."""
    from vocencebench.probes import default_probes
    probes = list(base) if base is not None else default_probes()
    if gender:
        probes.append(GenderProbe(device=device))
    if emotion:
        probes.append(EmotionProbe(device=device))
    if accent:
        probes.append(AccentProbe(device=device))
    if age:
        from vocencebench.probes.age import AgeProbe
        probes.append(AgeProbe(device=device))
    return probes
