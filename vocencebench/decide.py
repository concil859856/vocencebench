"""Decide an overall winner from a comparison, with configurable weights.

Combines per-trait adherence (correctness) and naturalness (preference) into one
composite per model and picks a winner if it clears a margin. Everything is
configurable; the defaults are sensible starting points, not calibrated constants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

# Default per-trait weights. Unlisted traits default to 1.0; the classifier-noisy ones
# are down-weighted so a bad reading can't swing the verdict before calibration.
DEFAULT_TRAIT_WEIGHTS: Dict[str, float] = {"emotion": 0.5, "accent": 0.5}
# Axis weights: adherence (obeying the prompt) leads, naturalness (sounding good) follows.
DEFAULT_W_ADHERENCE = 0.6
DEFAULT_W_NATURALNESS = 0.4
DEFAULT_MARGIN = 0.03


@dataclass
class Decision:
    labels: tuple
    winner: str                       # a label, or "tie"
    composite: Dict[str, float] = field(default_factory=dict)
    adherence: Dict[str, float] = field(default_factory=dict)
    naturalness: Dict[str, float] = field(default_factory=dict)
    margin: float = DEFAULT_MARGIN
    reason: str = ""


def _weighted_adherence(scores: Dict[str, Optional[float]], weights: Dict[str, float]) -> float:
    num = den = 0.0
    for trait, s in scores.items():
        if s is None:
            continue
        w = weights.get(trait, 1.0)
        num += w * s
        den += w
    return round(num / den, 6) if den else 0.0


def decide(
    comparison,
    *,
    labels: Optional[tuple] = None,
    trait_weights: Optional[Dict[str, float]] = None,
    w_adherence: float = DEFAULT_W_ADHERENCE,
    w_naturalness: float = DEFAULT_W_NATURALNESS,
    margin: float = DEFAULT_MARGIN,
) -> Decision:
    """Pick the winner from a :class:`Head2Head` (dataset) or :class:`PairResult` (one duel).

    composite = w_adherence · adherence + w_naturalness · naturalness, per model. The
    higher composite wins if the gap exceeds ``margin``, else it is a tie. ``adherence``
    is a per-trait-weighted mean of each model's match; ``naturalness`` is each model's
    quality/preference in [0,1].
    """
    weights = dict(DEFAULT_TRAIT_WEIGHTS)
    if trait_weights:
        weights.update(trait_weights)

    scores_a, scores_b, nat_a, nat_b, la, lb = _extract(comparison, labels)
    adh_a = _weighted_adherence(scores_a, weights)
    adh_b = _weighted_adherence(scores_b, weights)
    comp_a = round(w_adherence * adh_a + w_naturalness * nat_a, 6)
    comp_b = round(w_adherence * adh_b + w_naturalness * nat_b, 6)

    if comp_a - comp_b > margin:
        winner = la
    elif comp_b - comp_a > margin:
        winner = lb
    else:
        winner = "tie"

    reason = (f"adherence {la} {adh_a:.2f} vs {lb} {adh_b:.2f} (weight {w_adherence}); "
              f"naturalness {la} {nat_a:.2f} vs {lb} {nat_b:.2f} (weight {w_naturalness}); "
              f"composite {la} {comp_a:.3f} vs {lb} {comp_b:.3f}; "
              f"margin {margin} -> {winner}")
    return Decision(labels=(la, lb), winner=winner,
                    composite={la: comp_a, lb: comp_b},
                    adherence={la: adh_a, lb: adh_b},
                    naturalness={la: nat_a, lb: nat_b},
                    margin=margin, reason=reason)


def _extract(comparison, labels):
    """Pull per-model trait scores + naturalness in [0,1] from either input type."""
    if hasattr(comparison, "objective_a"):  # Head2Head (dataset-level aggregates)
        la, lb = labels or (comparison.label_a, comparison.label_b)
        scores_a = dict(comparison.objective_a)
        scores_b = dict(comparison.objective_b)
        nat_a = comparison.pairwise_a.get("naturalness", 0.5)
        nat_b = round(1.0 - nat_a, 6)
        return scores_a, scores_b, nat_a, nat_b, la, lb

    if hasattr(comparison, "traits"):  # PairResult (single duel)
        la, lb = labels or ("a", "b")
        scores_a = {t: te.score_a for t, te in comparison.traits.items()}
        scores_b = {t: te.score_b for t, te in comparison.traits.items()}
        nat_a, nat_b = _pair_naturalness(comparison.naturalness)
        return scores_a, scores_b, nat_a, nat_b, la, lb

    raise TypeError("decide() expects a Head2Head or PairResult")


def _pair_naturalness(v):
    if v is None:
        return 0.5, 0.5
    if v.score_a is not None and v.score_b is not None:
        return round(v.score_a / 3.0, 6), round(v.score_b / 3.0, 6)
    if v.winner == "a":
        return 1.0, 0.0
    if v.winner == "b":
        return 0.0, 1.0
    return 0.5, 0.5
