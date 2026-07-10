"""Canonical trait vocabulary for prompt-driven speech.

Each controllable voice attribute is described once here: its allowed values, whether
it can be measured objectively from the signal (``objective``), needs a listening
judgement (``holistic``), or both (``hybrid``), and the natural-language dimension a
judge is asked to focus on. Everything downstream — prompts, probes, metrics — reads
this registry, so adding a trait is a one-place change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

Kind = str  # "objective" | "holistic" | "hybrid"


@dataclass(frozen=True)
class Trait:
    """One controllable voice attribute."""

    name: str
    kind: Kind
    # Reference value set. Empty tuple = open-vocabulary (free-text instruction).
    values: Tuple[str, ...] = ()
    # What a judge is told to listen for when comparing two clips on this trait.
    dimension: str = ""
    # True for values that lie on an ordered scale (age, pace, pitch, loudness), so a
    # near-miss (adjacent bucket) can be scored more leniently than a far miss.
    ordinal: bool = False

    def is_probeable(self) -> bool:
        return self.kind in ("objective", "hybrid")

    def needs_judge(self) -> bool:
        return self.kind in ("holistic", "hybrid")


_REGISTRY: Dict[str, Trait] = {}


def register(trait: Trait) -> Trait:
    _REGISTRY[trait.name] = trait
    return trait


register(Trait(
    name="gender", kind="objective", values=("male", "female"),
    dimension="whether the speaker's voice matches the requested gender",
))
register(Trait(
    name="age", kind="objective", ordinal=True,
    values=("child", "young adult", "middle-aged adult", "elderly person"),
    dimension="whether the speaker's apparent age matches the requested age group",
))
register(Trait(
    name="pace", kind="objective", ordinal=True,
    values=("slow", "moderate", "fast"),
    dimension="whether the speaking rate matches the requested pace",
))
register(Trait(
    name="pitch", kind="objective", ordinal=True,
    values=("low", "medium", "high"),
    dimension="whether the overall pitch matches the requested level",
))
register(Trait(
    name="loudness", kind="objective", ordinal=True,
    values=("quiet", "normal", "loud"),
    dimension="whether the loudness/energy matches the requested level",
))
register(Trait(
    name="emotion", kind="objective",
    values=("neutral", "happy", "sad", "angry", "fearful", "surprised"),
    dimension="whether the expressed emotion matches the requested emotion",
))
register(Trait(
    name="accent", kind="objective",
    values=("American", "British", "Australian", "Indian"),
    dimension="whether the accent matches the requested accent",
))
register(Trait(
    name="tone", kind="holistic",
    values=("warm", "authoritative", "playful", "serious", "soothing"),
    dimension="whether the overall tone / delivery style matches the requested tone",
))
register(Trait(
    name="environment", kind="hybrid",
    values=("a quiet studio", "a busy street", "a large hall", "over a phone line"),
    dimension="whether the recording environment / acoustic setting matches the request",
))


def get(name: str) -> Trait:
    if name not in _REGISTRY:
        # Unknown traits are treated as open-vocabulary holistic attributes.
        return Trait(name=name, kind="holistic",
                     dimension=f"whether the delivery matches the requested {name}")
    return _REGISTRY[name]


def all_traits() -> List[Trait]:
    return list(_REGISTRY.values())


def names() -> List[str]:
    return list(_REGISTRY.keys())
