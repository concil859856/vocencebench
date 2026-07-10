"""Probe protocol and shared scoring helpers.

A probe measures one trait deterministically from the audio and scores the requested
value against the measurement. Objective, inspectable, and identical across runs and
machines — the reproducible backbone of the adherence score.
"""

from __future__ import annotations

from typing import List, Optional, Protocol, Sequence, Tuple

from vocencebench.schema import ProbeResult, Sample


class Probe(Protocol):
    trait: str

    def score(self, sample: Sample, audio: bytes) -> Optional[ProbeResult]:
        """Return a ProbeResult, or None if this probe's trait was not requested."""
        ...


def bucket(value: float, edges: Sequence[Tuple[float, str]]) -> str:
    """Map a scalar to a labelled bucket. ``edges`` = [(upper_bound, label), ...] with
    the final entry acting as the catch-all (its bound is ignored)."""
    for upper, label in edges[:-1]:
        if value < upper:
            return label
    return edges[-1][1]


def ordinal_score(requested: str, measured: str, order: Sequence[str]) -> float:
    """1.0 exact bucket, 0.5 adjacent bucket, else 0.0 (ordinal ±1 tolerance)."""
    if requested not in order or measured not in order:
        return 1.0 if requested == measured else 0.0
    dist = abs(order.index(requested) - order.index(measured))
    return 1.0 if dist == 0 else 0.5 if dist == 1 else 0.0


def exact_score(requested: str, measured: str) -> float:
    return 1.0 if requested == measured else 0.0
