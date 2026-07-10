"""Judge & probe calibration against a human gold set.

The mechanics only produce *trusted* scores after they agree with humans. This module
provides the agreement statistics (chance-corrected, not raw accuracy), gold-set
sampling and label i/o, and a per-dimension report that says which dimensions to trust,
drop, or move to a probe — plus a weight suggestion.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from vocencebench.schema import Sample


# ----------------------------------------------------------------- agreement metrics
def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    """Chance-corrected agreement between two raters over aligned nominal labels."""
    n = len(a)
    if n == 0:
        return float("nan")
    labels = set(a) | set(b)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[l] / n) * (cb[l] / n) for l in labels)
    return 1.0 if pe >= 1 else round((po - pe) / (1 - pe), 6)


def krippendorff_alpha(units: Sequence[Sequence[Optional[str]]]) -> float:
    """Nominal Krippendorff's alpha over units × raters (missing labels allowed)."""
    items = [[x for x in u if x is not None] for u in units]
    items = [u for u in items if len(u) >= 2]
    if not items:
        return float("nan")
    do = 0.0
    for u in items:
        c = Counter(u)
        disagree = sum(c[x] * c[y] for x in c for y in c if x != y)
        do += disagree / (len(u) - 1)
    n = sum(len(u) for u in items)
    do /= n
    allc = Counter(x for u in items for x in u)
    total = sum(allc.values())
    de = sum(allc[x] * allc[y] for x in allc for y in allc if x != y) / (total * (total - 1))
    return round(1 - do / de, 6) if de > 0 else 1.0


def accuracy(a: Sequence[str], b: Sequence[str]) -> float:
    n = len(a)
    return round(sum(1 for x, y in zip(a, b) if x == y) / n, 6) if n else float("nan")


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation (numpy only)."""
    import numpy as np
    if len(x) < 2:
        return float("nan")
    rx = np.argsort(np.argsort(np.asarray(x, float)))
    ry = np.argsort(np.argsort(np.asarray(y, float)))
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    return round(float((rx * ry).sum() / denom), 6) if denom else float("nan")


# --------------------------------------------------------------------- gold set io
@dataclass
class GoldItem:
    """One calibration unit: a comparison with human labels and a machine label."""

    sample_id: str
    dimension: str
    human_labels: List[str] = field(default_factory=list)  # winner per rater: a|b|tie
    machine_label: Optional[str] = None                    # judge or probe winner

    def majority_human(self) -> Optional[str]:
        if not self.human_labels:
            return None
        return Counter(self.human_labels).most_common(1)[0][0]


def stratified_comparisons(dataset: Sequence[Sample], *, per_cell: int = 3,
                           dimensions: Sequence[str] = ("naturalness",)) -> List[Dict]:
    """Pick comparison specs stratified by (category, difficulty, dimension).

    Returns lightweight dicts describing which samples to have judged and human-labelled.
    Deterministic given the dataset order.
    """
    cells: Dict[tuple, int] = {}
    out: List[Dict] = []
    for s in dataset:
        for dim in dimensions:
            key = (s.category, s.difficulty, dim)
            if cells.get(key, 0) >= per_cell:
                continue
            cells[key] = cells.get(key, 0) + 1
            out.append({"sample_id": s.id, "dimension": dim, "category": s.category,
                        "difficulty": s.difficulty})
    return out


def export_for_labeling(items: Sequence[Dict], path: str | Path) -> None:
    """Write comparison specs to JSONL for human raters to fill in a ``winner`` field."""
    with open(path, "w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps({**it, "winner": ""}) + "\n")


def load_gold(path: str | Path) -> List[GoldItem]:
    """Load labelled gold items. Each line may carry ``human_labels`` (list) or a single
    ``winner``; multiple rater files can be merged by sample_id+dimension beforehand."""
    items: Dict[tuple, GoldItem] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (row["sample_id"], row["dimension"])
        gi = items.setdefault(key, GoldItem(row["sample_id"], row["dimension"]))
        if "human_labels" in row:
            gi.human_labels.extend(row["human_labels"])
        elif row.get("winner"):
            gi.human_labels.append(row["winner"])
        if row.get("machine_label"):
            gi.machine_label = row["machine_label"]
    return list(items.values())


# ------------------------------------------------------------------------- report
def agreement_report(gold: Sequence[GoldItem]) -> Dict[str, Dict]:
    """Per-dimension: n, human-ceiling alpha, judge accuracy & kappa vs majority human."""
    by_dim: Dict[str, List[GoldItem]] = {}
    for g in gold:
        by_dim.setdefault(g.dimension, []).append(g)

    report: Dict[str, Dict] = {}
    for dim, items in by_dim.items():
        paired = [(g.machine_label, g.majority_human()) for g in items
                  if g.machine_label is not None and g.majority_human() is not None]
        machine = [m for m, h in paired]
        human = [h for m, h in paired]
        rater_matrix = [g.human_labels for g in items if len(g.human_labels) >= 2]
        report[dim] = {
            "n": len(items),
            "n_paired": len(paired),
            "human_alpha": krippendorff_alpha(rater_matrix) if rater_matrix else None,
            "judge_accuracy": accuracy(machine, human) if paired else None,
            "judge_kappa": cohen_kappa(machine, human) if paired else None,
        }
    return report


def suggest_weights(report: Dict[str, Dict], *, floor: float = 0.0) -> Dict[str, float]:
    """Weight each dimension by max(0, judge_kappa); dimensions at/below chance get 0.

    Normalised to sum to 1 over the kept dimensions. Dimensions below ``floor`` kappa are
    flagged for a probe or human review instead of the judge.
    """
    raw = {d: max(0.0, (r.get("judge_kappa") or 0.0)) for d, r in report.items()
           if (r.get("judge_kappa") or -1) > floor}
    total = sum(raw.values())
    return {d: round(w / total, 6) for d, w in raw.items()} if total else {}
