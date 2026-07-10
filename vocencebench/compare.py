"""Symmetric head-to-head between two PromptTTS models.

Runs :func:`vocencebench.evaluate_pair` on both models over a dataset: every voice trait
is scored absolutely for each model (objective ones by probes, the rest by the judge),
and naturalness is the pairwise preference. Declares a winner per axis.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from vocencebench.judge.base import Judge
from vocencebench.pair import evaluate_pair
from vocencebench.schema import Sample, TIE, WINNER_A

TTS = Callable[[str, str], bytes]


@dataclass
class Head2Head:
    label_a: str
    label_b: str
    n: int = 0
    objective_a: Dict[str, float] = field(default_factory=dict)
    objective_b: Dict[str, float] = field(default_factory=dict)
    pairwise_a: Dict[str, float] = field(default_factory=dict)   # naturalness win-rate for A (display)
    naturalness_a: float = 0.5   # absolute mean naturalness quality [0,1] for A
    naturalness_b: float = 0.5   # ... for B (used by decide, comparable to adherence)
    per_sample: List[dict] = field(default_factory=list)

    def summary(self) -> str:
        oa, ob = _mean(self.objective_a.values()), _mean(self.objective_b.values())
        nat = self.pairwise_a.get("naturalness")
        out = [f"head-to-head over {self.n} samples: {self.label_a} vs {self.label_b}",
               f"  trait adherence:      {self.label_a} {oa:.2f}  vs  {self.label_b} {ob:.2f}"
               f"   -> {self._who(oa, ob)}"]
        if nat is not None:
            out.append(f"  naturalness win-rate: {self.label_a} {nat:.2f}   -> {self._who(nat, 1 - nat)}")
        for t in sorted(set(self.objective_a) | set(self.objective_b)):
            out.append(f"    {t:10s} {self.label_a} {round(self.objective_a.get(t, 0), 2)}"
                       f"  vs  {self.label_b} {round(self.objective_b.get(t, 0), 2)}")
        return "\n".join(out)

    def _who(self, a, b):
        return "tie" if abs(a - b) < 1e-9 else (self.label_a if a > b else self.label_b)


def _mean(xs):
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
    all_at_once: bool = False,
    swap_eval: bool = False,
    on_audio: Optional[Callable[[str, bytes, bytes], None]] = None,
) -> Head2Head:
    """Compare two models symmetrically. ``all_at_once`` and ``swap_eval`` are passed to
    :func:`evaluate_pair` (per-trait focused judging and no order-swap by default)."""
    obj_a: Dict[str, List[float]] = defaultdict(list)
    obj_b: Dict[str, List[float]] = defaultdict(list)
    nat: List[float] = []                 # win value for A (win-rate display)
    nat_qa: List[float] = []              # absolute naturalness quality for A
    nat_qb: List[float] = []              # ... for B
    h = Head2Head(label_a=labels[0], label_b=labels[1], n=len(dataset))

    for s in dataset:
        wav_a, wav_b = model_a(s.text, s.instruction), model_b(s.text, s.instruction)
        if on_audio is not None:
            on_audio(s.id, wav_a, wav_b)
        pr = evaluate_pair(wav_a, wav_b, text=s.text, traits=s.traits, judge=judge,
                           instruction=s.instruction, probes=probes, category=s.category,
                           swap_eval=swap_eval, all_at_once=all_at_once,
                           naturalness=score_naturalness)
        rec = {"id": s.id, "text": s.text, "instruction": s.instruction, "traits": s.traits,
               "objective": [], "pairwise": []}
        for t, te in pr.traits.items():
            if te.score_a is not None:
                obj_a[t].append(te.score_a)
            if te.score_b is not None:
                obj_b[t].append(te.score_b)
            rec["objective"].append({
                "trait": t, "requested": te.requested, "judged": te.source == "judge",
                "a": {"score": te.score_a, "measured": te.measured_a, "reasoning": te.reasoning_a},
                "b": {"score": te.score_b, "measured": te.measured_b, "reasoning": te.reasoning_b},
            })
        if pr.naturalness is not None:
            v = pr.naturalness
            nat.append(_wv(v))
            qa, qb = _quality(v)
            nat_qa.append(qa)
            nat_qb.append(qb)
            rec["pairwise"].append({"dimension": "naturalness", "winner": v.winner,
                "consistent": v.consistent, "score_a": v.score_a, "score_b": v.score_b,
                "reasoning_a": v.reasoning_a, "reasoning_b": v.reasoning_b, "comparison": v.reasoning})
        h.per_sample.append(rec)

    h.objective_a = {t: _mean(v) for t, v in obj_a.items()}
    h.objective_b = {t: _mean(v) for t, v in obj_b.items()}
    if nat:
        h.pairwise_a = {"naturalness": round(sum(nat) / len(nat), 6)}
        h.naturalness_a = round(sum(nat_qa) / len(nat_qa), 6)
        h.naturalness_b = round(sum(nat_qb) / len(nat_qb), 6)
    return h


def _wv(v) -> float:
    if not v.consistent:
        return 0.5
    return 1.0 if v.winner == WINNER_A else 0.5 if v.winner == TIE else 0.0


def _quality(v):
    """Absolute per-model naturalness quality in [0,1] from the judge's 0-3 scores;
    falls back to the winner if scores are missing."""
    if v.score_a is not None and v.score_b is not None:
        return round(v.score_a / 3.0, 6), round(v.score_b / 3.0, 6)
    if v.winner == WINNER_A:
        return 1.0, 0.0
    if v.winner == WINNER_B:
        return 0.0, 1.0
    return 0.5, 0.5
