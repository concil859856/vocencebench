"""Evaluate one pair of clips on all requested traits + naturalness.

The single entry point for comparing two clips of the same prompt. Objective traits are
scored by free deterministic probes; the rest, plus naturalness, go to the audio-LLM
judge — in one combined call by default (``all_at_once=True``), optionally order-swapped
(``swap_eval=True``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from vocencebench.audio import Audio, to_wav_bytes
from vocencebench.judge.base import Judge
from vocencebench.schema import Sample, TIE, Verdict, WINNER_A, WINNER_B


@dataclass
class TraitEval:
    """One trait's outcome for both clips (scores normalised to [0,1])."""

    trait: str
    requested: str
    source: str                 # "probe" | "judge"
    score_a: Optional[float]
    score_b: Optional[float]
    winner: str
    measured_a: Optional[str] = None
    measured_b: Optional[str] = None
    reasoning_a: str = ""
    reasoning_b: str = ""
    consistent: bool = True


@dataclass
class PairResult:
    traits: Dict[str, TraitEval] = field(default_factory=dict)
    naturalness: Optional[Verdict] = None

    def adherence(self) -> Dict[str, float]:
        """Mean absolute adherence per clip over all traits (both can be high)."""
        a = [t.score_a for t in self.traits.values() if t.score_a is not None]
        b = [t.score_b for t in self.traits.values() if t.score_b is not None]
        return {"a": round(sum(a) / len(a), 4) if a else 0.0,
                "b": round(sum(b) / len(b), 4) if b else 0.0}


def _w(a: Optional[float], b: Optional[float]) -> str:
    if a is None or b is None:
        return TIE
    return WINNER_A if a > b else WINNER_B if b > a else TIE


def evaluate_pair(
    clip_a: Audio,
    clip_b: Audio,
    *,
    text: str,
    traits: Dict[str, str],
    judge: Judge,
    instruction: str = "",
    probes: Optional[Sequence] = None,
    category: str = "general",
    swap_eval: bool = False,
    all_at_once: bool = False,
    naturalness: bool = True,
) -> PairResult:
    """Compare two clips on every requested trait + naturalness.

    Args:
        clip_a, clip_b: the two clips (bytes or paths) of the same text.
        text, instruction, traits: the prompt and requested voice attributes.
        judge: an audio-LLM :class:`Judge`.
        probes: trait probes (defaults to the model-free acoustic set).
        swap_eval: also run the judge in the reversed audio order and keep only
            consistent verdicts (default False).
        all_at_once: default False — one focused judge call per judged trait (more
            reliable). Set True to score all judged traits + naturalness in ONE call
            (cheaper, but the judge splits attention across traits).
    """
    from vocencebench.probes import default_probes
    wav_a, wav_b = to_wav_bytes(clip_a), to_wav_bytes(clip_b)
    probe_map = {p.trait: p for p in (default_probes() if probes is None else probes)}
    sample = Sample(id="pair", text=text, instruction=instruction, traits=traits, category=category)

    result = PairResult()
    judged: List[str] = []
    for t, val in traits.items():
        probe = probe_map.get(t)
        if probe is None:
            judged.append(t)
            continue
        ra, rb = probe.score(sample, wav_a), probe.score(sample, wav_b)
        result.traits[t] = TraitEval(
            trait=t, requested=val, source="probe",
            score_a=None if ra is None else ra.score, score_b=None if rb is None else rb.score,
            winner=_w(None if ra is None else ra.score, None if rb is None else rb.score),
            measured_a=None if ra is None else ra.measured,
            measured_b=None if rb is None else rb.measured,
        )

    jtraits = {t: traits[t] for t in judged}
    # evaluate_pair owns the swap decision, so neutralise the Judge's own setting.
    orig_swap = judge.swap
    judge.swap = False
    try:
        if all_at_once:
            _judge_all(result, judge, text, instruction, jtraits, wav_a, wav_b, category,
                       naturalness, swap_eval)
        else:
            for t in judged:
                v = _judge_one(judge, "adherence", text, instruction, t, traits[t], wav_a, wav_b,
                               category, swap_eval)
                result.traits[t] = _to_trait(t, traits[t], v)
            if naturalness:
                result.naturalness = _judge_one(judge, "naturalness", text, instruction,
                                                "naturalness", "", wav_a, wav_b, category, swap_eval)
    finally:
        judge.swap = orig_swap
    return result


# --------------------------------------------------------------------------- helpers
def _to_trait(t: str, requested: str, v: Verdict) -> TraitEval:
    sa = None if v.score_a is None else v.score_a / 3.0
    sb = None if v.score_b is None else v.score_b / 3.0
    return TraitEval(trait=t, requested=requested, source="judge", score_a=sa, score_b=sb,
                     winner=v.winner, reasoning_a=v.reasoning_a, reasoning_b=v.reasoning_b,
                     consistent=v.consistent)


def _judge_one(judge, kind, text, instruction, trait, value, wav_a, wav_b, category, swap_eval):
    """One judged dimension, optionally order-swapped for consistency."""
    def call(a, b):
        return (judge.adherence(text, instruction, trait, value, a, b) if kind == "adherence"
                else judge.naturalness(text, a, b, category=category))
    v1 = call(wav_a, wav_b)
    if not swap_eval:
        return v1
    v2 = call(wav_b, wav_a)  # reversed; winner/scores are flipped
    return _reconcile_verdict(v1, v2)


def _reconcile_verdict(v1: Verdict, v2: Verdict) -> Verdict:
    flip = {WINNER_A: WINNER_B, WINNER_B: WINNER_A, TIE: TIE}
    w2 = flip[v2.winner]
    winner = v1.winner if v1.winner == w2 else TIE
    sa = _avg(v1.score_a, v2.score_b)
    sb = _avg(v1.score_b, v2.score_a)
    return Verdict(dimension=v1.dimension, winner=winner, score_a=sa, score_b=sb,
                   reasoning=v1.reasoning, reasoning_a=v1.reasoning_a, reasoning_b=v1.reasoning_b,
                   confidence=v1.confidence, consistent=(v1.winner == w2))


def _judge_all(result, judge, text, instruction, jtraits, wav_a, wav_b, category, naturalness, swap_eval):
    res = judge.assess_all(text, instruction, jtraits, wav_a, wav_b, category=category)
    if swap_eval:
        res2 = judge.assess_all(text, instruction, jtraits, wav_b, wav_a, category=category)
        for t, v in res["traits"].items():
            res["traits"][t] = _reconcile_verdict(v, res2["traits"].get(t, v))
        if res["naturalness"] and res2["naturalness"]:
            res["naturalness"] = _reconcile_verdict(res["naturalness"], res2["naturalness"])
    for t, v in res["traits"].items():
        result.traits[t] = _to_trait(t, jtraits[t], v)
    if naturalness:
        result.naturalness = res["naturalness"]


def _avg(x, y):
    xs = [v for v in (x, y) if v is not None]
    return int(round(sum(xs) / len(xs))) if xs else None
