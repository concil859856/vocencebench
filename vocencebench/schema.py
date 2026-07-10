"""Core data structures for the evaluation pipeline.

``Sample`` is the unit of a dataset (text + voice instruction + requested traits).
``Verdict`` is one pairwise judge decision. ``ProbeResult`` is one deterministic
trait measurement. ``SampleResult`` and ``EvalReport`` aggregate upward.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Sample(BaseModel):
    """One evaluation case: the text to speak and the requested voice."""

    id: str
    text: str
    # Natural-language voice instruction given to the TTS model (the "prompt").
    instruction: str = ""
    # Structured requested traits, e.g. {"gender": "female", "pace": "fast"}.
    traits: Dict[str, str] = Field(default_factory=dict)
    # Optional grouping for naturalness rubrics / reporting.
    category: str = "general"
    difficulty: int = 0
    language: str = "en"


# ----------------------------------------------------------------------------- verdicts
WINNER_A = "a"
WINNER_B = "b"
TIE = "tie"


@dataclass(frozen=True)
class Verdict:
    """One pairwise comparison outcome from the audio-LLM judge.

    ``winner`` is expressed in caller terms (a vs b) after any internal order-swap has
    been undone, so ``a`` always refers to the first clip the caller passed.
    """

    dimension: str
    winner: str            # WINNER_A | WINNER_B | TIE
    score_a: Optional[int] = None   # 0-3 per-clip score, when the judge provides one
    score_b: Optional[int] = None
    reasoning: str = ""             # the comparison / tie-break text
    reasoning_a: str = ""           # detailed per-clip analysis (clip a = model)
    reasoning_b: str = ""           # detailed per-clip analysis (clip b = reference)
    confidence: float = 0.0
    consistent: bool = True         # True if both audio orders agreed (order-swap check)
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def a_wins(self) -> bool:
        return self.winner == WINNER_A

    @property
    def b_wins(self) -> bool:
        return self.winner == WINNER_B


@dataclass(frozen=True)
class ProbeResult:
    """One deterministic trait measurement on a single clip."""

    trait: str
    requested: str
    measured: Optional[str] = None      # bucket/label the probe read from the audio
    matched: Optional[bool] = None      # exact requested==measured
    score: float = 0.0                  # [0,1]; ordinal traits allow partial credit
    detail: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SampleResult:
    """Everything computed for one sample: intelligibility, probes, judge verdicts."""

    sample_id: str
    wer: Optional[float] = None
    intelligible: Optional[bool] = None
    probes: List[ProbeResult] = field(default_factory=list)
    adherence_verdicts: List[Verdict] = field(default_factory=list)  # per holistic trait
    naturalness: Optional[Verdict] = None
    note: str = ""


@dataclass
class EvalReport:
    """Aggregate over a dataset."""

    n: int = 0
    results: List[SampleResult] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        m = self.metrics
        lines = [f"vocencebench report over {self.n} samples"]
        if "naturalness_win_rate" in m:
            lines.append(f"  naturalness win-rate vs reference: {m['naturalness_win_rate']:.3f}")
        if "adherence_score" in m:
            lines.append(f"  adherence score (mean):            {m['adherence_score']:.3f}")
        for k, v in sorted(m.get("trait_csr", {}).items()):
            lines.append(f"    control-success[{k}]: {v:.3f}")
        if "wer" in m:
            lines.append(f"  mean WER:                          {m['wer']:.3f}")
        return "\n".join(lines)


# ----------------------------------------------------------------------------- dataset io
def load_dataset(path: str | Path) -> List[Sample]:
    """Load a JSONL (or JSON array) dataset of :class:`Sample` records."""
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        rows = json.loads(text)
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    return [Sample(**row) for row in rows]


def save_dataset(samples: List[Sample], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for s in samples:
            fh.write(s.model_dump_json() + "\n")
