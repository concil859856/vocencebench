"""Evaluation runner: generate → probe → judge → aggregate.

Given a model and a reference (anchor) that both turn ``(text, instruction)`` into WAV
bytes, evaluate a dataset and return an :class:`EvalReport`. Objective traits are scored
by deterministic probes where available; the rest, plus naturalness, by the audio-LLM
judge (pairwise, model vs reference). Clip ``a`` is always the model, clip ``b`` the
reference, so a win-rate above 0.5 means the model beats the reference.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

from vocencebench import metrics as _metrics
from vocencebench import traits as _traits
from vocencebench.judge.base import Judge
from vocencebench.schema import (
    EvalReport, ProbeResult, Sample, SampleResult, Verdict, WINNER_A, TIE,
)

TTS = Callable[[str, str], bytes]        # (text, instruction) -> wav bytes
Transcriber = Callable[[bytes], str]     # wav bytes -> transcript (optional, for WER)


def evaluate(
    dataset: Sequence[Sample],
    model: TTS,
    reference: TTS,
    judge: Judge,
    *,
    probes: Optional[Sequence] = None,
    transcriber: Optional[Transcriber] = None,
    score_naturalness: bool = True,
    on_error: Optional[Callable[[str, Exception], None]] = None,
) -> EvalReport:
    probe_by_trait = {p.trait: p for p in (probes or [])}
    results: List[SampleResult] = []

    for sample in dataset:
        sr = SampleResult(sample_id=sample.id)
        try:
            model_wav = model(sample.text, sample.instruction)
            ref_wav = reference(sample.text, sample.instruction)

            # Intelligibility (optional).
            if transcriber is not None:
                sr.wer, sr.intelligible = _wer(transcriber, model_wav, sample.text)

            # Adherence per requested trait: probe if we have one, else judge.
            for trait_name, value in sample.traits.items():
                probe = probe_by_trait.get(trait_name)
                if probe is not None:
                    res = probe.score(sample, model_wav)
                    if res is not None:
                        sr.probes.append(res)
                else:
                    v = judge.adherence(sample.text, sample.instruction, trait_name, value,
                                        model_wav, ref_wav)
                    sr.adherence_verdicts.append(v)

            # Naturalness (pairwise, model vs reference).
            if score_naturalness:
                sr.naturalness = judge.naturalness(
                    sample.text, model_wav, ref_wav, category=sample.category)
        except Exception as exc:  # one bad sample must not abort the run
            sr.note = f"error: {exc}"
            if on_error is not None:
                on_error(sample.id, exc)
        results.append(sr)

    return EvalReport(n=len(results), results=results, metrics=_aggregate(results))


def _wer(transcriber: Transcriber, wav: bytes, text: str) -> tuple:
    import re
    hyp = transcriber(wav)
    norm = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower()).split()
    ref, hyp_w = norm(text), norm(hyp)
    dist = _levenshtein(ref, hyp_w)
    wer = dist / max(len(ref), 1)
    return round(wer, 4), (wer <= 0.15)


def _levenshtein(a: Sequence[str], b: Sequence[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def _aggregate(results: Sequence[SampleResult]) -> Dict:
    all_probes: List[ProbeResult] = [p for r in results for p in r.probes]
    adh_verdicts: List[Verdict] = [v for r in results for v in r.adherence_verdicts]
    nat_verdicts: List[Verdict] = [r.naturalness for r in results if r.naturalness is not None]
    wers = [r.wer for r in results if r.wer is not None]

    metrics: Dict = {}
    # Per-objective-trait control-success (from probes).
    trait_csr = _metrics.trait_control_success(all_probes)
    # Holistic adherence: judge win-rate vs reference, per trait, folded in as a rate.
    holistic_by_trait: Dict[str, List[float]] = {}
    for v in adh_verdicts:
        holistic_by_trait.setdefault(v.dimension, []).append(_win_val(v))
    for t, vals in holistic_by_trait.items():
        trait_csr[t] = round(sum(vals) / len(vals), 6)
    if trait_csr:
        metrics["trait_csr"] = trait_csr

    # Overall adherence = mean over all objective probe scores + holistic win-values.
    adh_components = [p.score for p in all_probes] + [_win_val(v) for v in adh_verdicts]
    if adh_components:
        metrics["adherence_score"] = round(sum(adh_components) / len(adh_components), 6)

    nwr = _metrics.win_rate(nat_verdicts)
    if nwr is not None:
        metrics["naturalness_win_rate"] = nwr
    if wers:
        metrics["wer"] = round(sum(wers) / len(wers), 4)
        metrics["intelligible_rate"] = round(
            sum(1 for r in results if r.intelligible) / len(wers), 4)
    metrics["errors"] = sum(1 for r in results if r.note)
    return metrics


def _win_val(v: Verdict) -> float:
    if not v.consistent:
        return 0.5
    return 1.0 if v.winner == WINNER_A else 0.5 if v.winner == TIE else 0.0
