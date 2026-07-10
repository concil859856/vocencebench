"""Symmetric head-to-head between two PromptTTS models.

Unlike :func:`vocencebench.evaluate` (one model under test vs a fixed reference), this
treats both sides as equals: every objective trait is **probed on both** models
(absolute control-success, directly comparable), and holistic traits + naturalness are
judged **pairwise** (a vs b) so you get a head-to-head win-rate. The result declares a
winner per axis.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from vocencebench.judge.base import Judge
from vocencebench.schema import Sample, TIE, WINNER_A

TTS = Callable[[str, str], bytes]


@dataclass
class Head2Head:
    label_a: str
    label_b: str
    n: int = 0
    # objective adherence, per trait, per model (absolute, comparable)
    objective_a: Dict[str, float] = field(default_factory=dict)
    objective_b: Dict[str, float] = field(default_factory=dict)
    # pairwise win-rate for model A (>0.5 => A wins that dimension)
    pairwise_a: Dict[str, float] = field(default_factory=dict)
    per_sample: List[dict] = field(default_factory=list)

    def summary(self) -> str:
        oa = _mean(self.objective_a.values())
        ob = _mean(self.objective_b.values())
        nat = self.pairwise_a.get("naturalness")
        lines = [f"head-to-head over {self.n} samples: {self.label_a} vs {self.label_b}",
                 f"  objective adherence:  {self.label_a} {oa:.2f}  vs  {self.label_b} {ob:.2f}"
                 f"   -> {self._who(oa, ob)}"]
        if nat is not None:
            lines.append(f"  naturalness win-rate: {self.label_a} {nat:.2f}"
                         f"   -> {self._who(nat, 1 - nat)}")
        for t in sorted(set(self.objective_a) | set(self.objective_b)):
            a, b = self.objective_a.get(t), self.objective_b.get(t)
            lines.append(f"    {t:10s} {self.label_a} {a if a is None else round(a,2)}"
                         f"  vs  {self.label_b} {b if b is None else round(b,2)}")
        return "\n".join(lines)

    def _who(self, a: float, b: float) -> str:
        if abs(a - b) < 1e-9:
            return "tie"
        return self.label_a if a > b else self.label_b


def _mean(xs) -> float:
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 6) if xs else 0.0


def compare_models(
    dataset: Sequence[Sample],
    model_a: TTS,
    model_b: TTS,
    judge: Judge,
    *,
    probes: Optional[Sequence] = None,
    labels: tuple = ("model_a", "model_b"),
    score_naturalness: bool = True,
    on_audio: Optional[Callable[[str, bytes, bytes], None]] = None,
) -> Head2Head:
    probe_by_trait = {p.trait: p for p in (probes or [])}
    obj_a: Dict[str, List[float]] = defaultdict(list)
    obj_b: Dict[str, List[float]] = defaultdict(list)
    pair: Dict[str, List[float]] = defaultdict(list)
    h = Head2Head(label_a=labels[0], label_b=labels[1], n=len(dataset))

    for s in dataset:
        wav_a = model_a(s.text, s.instruction)
        wav_b = model_b(s.text, s.instruction)
        if on_audio is not None:
            on_audio(s.id, wav_a, wav_b)
        rec = {"id": s.id, "text": s.text, "instruction": s.instruction, "traits": s.traits,
               "objective": [], "pairwise": []}

        for trait, value in s.traits.items():
            probe = probe_by_trait.get(trait)
            if probe is not None:  # objective: score BOTH models absolutely
                ra = probe.score(s, wav_a)
                rb = probe.score(s, wav_b)
                if ra is not None:
                    obj_a[trait].append(ra.score)
                if rb is not None:
                    obj_b[trait].append(rb.score)
                rec["objective"].append({"trait": trait, "requested": value,
                    "a": _pr(ra), "b": _pr(rb)})
            else:  # holistic: judge pairwise
                v = judge.adherence(s.text, s.instruction, trait, value, wav_a, wav_b)
                pair[trait].append(_wv(v))
                rec["pairwise"].append(_vd(f"adherence:{trait}", v))

        if score_naturalness:
            v = judge.naturalness(s.text, wav_a, wav_b, category=s.category)
            pair["naturalness"].append(_wv(v))
            rec["pairwise"].append(_vd("naturalness", v))
        h.per_sample.append(rec)

    h.objective_a = {t: _mean(v) for t, v in obj_a.items()}
    h.objective_b = {t: _mean(v) for t, v in obj_b.items()}
    h.pairwise_a = {d: _mean(v) for d, v in pair.items()}
    return h


def _wv(v) -> float:
    if not v.consistent:
        return 0.5
    return 1.0 if v.winner == WINNER_A else 0.5 if v.winner == TIE else 0.0


def _pr(r):
    if r is None:
        return None
    return {"measured": r.measured, "score": r.score, "detail": r.detail}


def _vd(dim, v):
    return {"dimension": dim, "winner": v.winner, "consistent": v.consistent,
            "score_a": v.score_a, "score_b": v.score_b,
            "reasoning_a": v.reasoning_a, "reasoning_b": v.reasoning_b, "comparison": v.reasoning}
