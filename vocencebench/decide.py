"""Decide an overall winner from a comparison — the finalized aggregation.

Pipeline (all steps configurable, sensible defaults):

  1. per dimension  -> desirability in [0,1]           (already produced upstream)
  2. per sample     -> geometric mean of desirabilities x intelligibility gate
  3. per model      -> arithmetic mean of sample composites
  4. A vs B         -> paired bootstrap Lower Confidence Bound of (challenger - incumbent)
  5. winner         -> LCB > dynamic margin, else tie

The geometric mean (step 2) is non-compensatory: a low score in any one dimension
drags the whole sample down, so no single field can dominate or be gamed. The gate is a
hard veto (unintelligible -> 0). The LCB + margin (steps 4-5) only crown a winner on a
statistically real AND perceptually meaningful improvement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# --- defaults (grounded in the research) ------------------------------------------
EPS = 0.01                 # desirability floor: softens the single-zero veto
C_MARGIN = 0.10            # dynamic-margin coefficient (fraction of remaining headroom)
FLOOR_MARGIN = 0.015       # margin floor ~= CMOS 0.09 (just-noticeable difference)
BOOTSTRAP_N = 2000
ALPHA = 0.05               # 5% one-sided LCB (95% confidence)
SEED = 3151662


@dataclass
class Decision:
    labels: tuple
    winner: str                                  # a label, or "tie"
    scores: Dict[str, float] = field(default_factory=dict)     # per-model composite S
    challenger: str = ""
    incumbent: str = ""
    lcb: float = 0.0                             # LCB(challenger - incumbent)
    margin: float = 0.0
    reason: str = ""


def dynamic_margin(incumbent_score: float, *, c: float = C_MARGIN, floor: float = FLOOR_MARGIN) -> float:
    """Headroom-scaled margin: max(floor, c * (1 - incumbent)). Bar rises near saturation."""
    return round(max(floor, c * (1.0 - incumbent_score)), 6)


def geometric_composite(dims: List[tuple], gate: int = 1, *, eps: float = EPS,
                        weights: Optional[Dict[str, float]] = None) -> float:
    """(Weighted) geometric mean of [0,1] desirabilities, times the gate.

    ``dims`` is a list of (name, score) pairs (score in [0,1]); ``gate`` 0 -> composite 0.
    """
    if gate == 0 or not dims:
        return 0.0
    num = den = 0.0
    for name, s in dims:
        w = (weights or {}).get(name, 1.0)
        num += w * math.log(max(min(float(s), 1.0), eps))
        den += w
    return round(math.exp(num / den), 6) if den else 0.0


def _sample_dims(rec: dict, side: str) -> List[tuple]:
    """Collect (name, desirability) for one model on one sample from a per_sample record."""
    dims: List[tuple] = []
    for o in rec.get("objective", []):
        s = (o.get(side) or {}).get("score")
        if s is not None:
            dims.append((o["trait"], s))
    for p in rec.get("pairwise", []):
        if p.get("dimension") == "naturalness":
            sc = p.get("score_a") if side == "a" else p.get("score_b")
            if sc is not None:
                dims.append(("naturalness", sc / 3.0))   # judge 0-3 -> [0,1]
    return dims


def sample_composites(h2h) -> Dict[str, List[float]]:
    """Per-model list of per-sample composites (geometric mean x gate)."""
    la, lb = h2h.label_a, h2h.label_b
    ca: List[float] = []
    cb: List[float] = []
    for rec in h2h.per_sample:
        ca.append(geometric_composite(_sample_dims(rec, "a"), rec.get("gate_a", 1)))
        cb.append(geometric_composite(_sample_dims(rec, "b"), rec.get("gate_b", 1)))
    return {la: ca, lb: cb}


def _paired_lcb(diff: List[float], *, n_boot: int, alpha: float, seed: int) -> float:
    """5%-percentile of the bootstrapped mean of the paired differences."""
    import numpy as np
    d = np.asarray(diff, dtype=np.float64)
    if d.size == 0:
        return 0.0
    if d.size == 1:
        return float(d[0])
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, d.size, size=(n_boot, d.size))].mean(axis=1)
    return float(np.percentile(means, alpha * 100))


def decide(
    h2h,
    *,
    incumbent: Optional[str] = None,
    weights: Optional[Dict[str, float]] = None,
    eps: float = EPS,
    c: float = C_MARGIN,
    floor: float = FLOOR_MARGIN,
    n_boot: int = BOOTSTRAP_N,
    alpha: float = ALPHA,
    seed: int = SEED,
) -> Decision:
    """Decide the winner of a :class:`Head2Head`.

    ``incumbent`` names the model being defended (king). If omitted, the lower-scoring
    model is treated as the incumbent/baseline and the higher-scoring model is the
    challenger. A challenger wins only if the paired bootstrap LCB of its composite
    advantage exceeds the dynamic margin; otherwise it is a tie.
    """
    la, lb = h2h.label_a, h2h.label_b
    # weighted geometric composites per sample (recompute so weights apply here too)
    ca: List[float] = []
    cb: List[float] = []
    for rec in h2h.per_sample:
        ca.append(geometric_composite(_sample_dims(rec, "a"), rec.get("gate_a", 1),
                                      eps=eps, weights=weights))
        cb.append(geometric_composite(_sample_dims(rec, "b"), rec.get("gate_b", 1),
                                      eps=eps, weights=weights))
    comps = {la: ca, lb: cb}
    scores = {la: round(sum(ca) / len(ca), 6) if ca else 0.0,
              lb: round(sum(cb) / len(cb), 6) if cb else 0.0}

    # roles: explicit incumbent, else the lower-scoring model is the baseline to beat
    if incumbent in (la, lb):
        inc = incumbent
    else:
        inc = la if scores[la] <= scores[lb] else lb
    chal = lb if inc == la else la

    diff = [x - y for x, y in zip(comps[chal], comps[inc])]
    lcb = round(_paired_lcb(diff, n_boot=n_boot, alpha=alpha, seed=seed), 6)
    margin = dynamic_margin(scores[inc], c=c, floor=floor)
    winner = chal if lcb > margin else "tie"

    reason = (f"{chal} composite {scores[chal]:.3f} vs {inc} {scores[inc]:.3f}; "
              f"paired LCB({chal}-{inc}) = {lcb:.3f}; "
              f"margin (incumbent {inc}={scores[inc]:.2f}) = {margin:.3f}; "
              f"LCB {'>' if lcb > margin else '<='} margin -> "
              f"{'WIN: ' + chal if winner != 'tie' else 'TIE'}")
    return Decision(labels=(la, lb), winner=winner, scores=scores, challenger=chal,
                    incumbent=inc, lcb=lcb, margin=margin, reason=reason)
