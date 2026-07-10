"""Aggregate metrics: win-rate, control-success-rate, confusion, bootstrap CI.

Conventions: in a model-vs-reference evaluation, clip ``a`` is always the model under
test and clip ``b`` is the reference/anchor, so a win-rate above 0.5 means the model
beats the reference.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

from vocencebench.schema import ProbeResult, TIE, Verdict, WINNER_A


def win_rate(verdicts: Sequence[Verdict], *, consistent_only: bool = True) -> Optional[float]:
    """Fraction of comparisons the model (clip a) wins; ties count 0.5.

    Returns None if there are no usable verdicts.
    """
    wins = 0.0
    n = 0
    for v in verdicts:
        if consistent_only and not v.consistent:
            continue
        n += 1
        if v.winner == WINNER_A:
            wins += 1.0
        elif v.winner == TIE:
            wins += 0.5
    return round(wins / n, 6) if n else None


def control_success(probes: Sequence[ProbeResult]) -> float:
    """Mean probe score in [0,1] (ordinal traits contribute partial credit)."""
    scored = [p.score for p in probes if p.score is not None]
    return round(sum(scored) / len(scored), 6) if scored else 0.0


def trait_control_success(probes: Sequence[ProbeResult]) -> Dict[str, float]:
    by: Dict[str, List[float]] = defaultdict(list)
    for p in probes:
        by[p.trait].append(p.score)
    return {t: round(sum(v) / len(v), 6) for t, v in by.items() if v}


def confusion(probes: Sequence[ProbeResult], trait: str) -> Dict[Tuple[str, str], int]:
    """(requested, measured) -> count for one trait, for diagnostics."""
    out: Dict[Tuple[str, str], int] = defaultdict(int)
    for p in probes:
        if p.trait == trait and p.measured is not None:
            out[(p.requested, p.measured)] += 1
    return dict(out)


def bootstrap_ci(values: Sequence[float], *, n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 12345) -> Tuple[float, float]:
    """Two-sided (1-alpha) bootstrap CI on the mean of ``values``."""
    import numpy as np

    d = np.asarray(list(values), dtype=np.float64)
    if d.size == 0:
        return (0.0, 0.0)
    if d.size == 1:
        return (float(d[0]), float(d[0]))
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, d.size, size=(n_boot, d.size))].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return (round(lo, 6), round(hi, 6))
