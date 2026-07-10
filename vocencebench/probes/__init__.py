"""Deterministic trait probes.

``default_probes()`` returns the model-free acoustic probes, which work out of the box
with no downloads. Classifier-based probes (gender, emotion, accent, age) are heavier
and pluggable — add them with :func:`with_classifiers` once their models are available.
"""

from __future__ import annotations

from typing import List

from vocencebench.probes.acoustic import LoudnessProbe, PaceProbe, PitchProbe
from vocencebench.probes.base import Probe

__all__ = ["Probe", "PaceProbe", "LoudnessProbe", "PitchProbe", "default_probes"]


def default_probes() -> List[Probe]:
    """Model-free acoustic probes: pace, loudness, pitch."""
    return [PaceProbe(), LoudnessProbe(), PitchProbe()]
