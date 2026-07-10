"""Deterministic corpus generator for trait-controlled evaluation.

Builds ``Sample`` records by combining a text with a set of requested traits and a
rendered natural-language voice instruction. Fully index-driven (no randomness), so the
same call reproduces the same corpus byte-for-byte. Texts are grouped by naturalness
category so the same corpus exercises both adherence (traits) and naturalness (hard
text).
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from vocencebench.schema import Sample

# Original seed texts, grouped by the naturalness dimension they stress.
TEXTS: Dict[str, List[str]] = {
    "general": [
        "The old lighthouse stood watch over the restless grey sea.",
        "Please remember to lock the back gate before it gets dark.",
        "A thin line of smoke rose from the cabin at the edge of the woods.",
    ],
    "emotion": [
        'She threw open the door and cried, "You made it, you actually made it!"',
        'He stared at the empty platform and murmured, "I suppose they are not coming."',
        '"Get back from the ledge!" the guard shouted, his voice cracking with fear.',
    ],
    "questions": [
        "You finished the whole report last night? On your own?",
        "So, remind me, which train leaves first, and does it stop at the harbour?",
        "You are telling me the keys were in your pocket the entire time?",
    ],
    "paralinguistics": [
        "Ugh, fine... I-I guess we can try it your way, but be careful.",
        "Shhh! Did you hear that? It went tap... tap... tap on the window.",
        "Wow, that is ABSOLUTELY the best news I have heard all week!",
    ],
    "syntax": [
        "The letter that the clerk who left early had misfiled was finally found.",
        "Whatever the committee decides tonight, however unpopular, becomes policy.",
        "The bridge, which had been closed for repairs since spring, reopened today.",
    ],
    "pronunciation": [
        "Transfer $1,204.50 to account 0087-QX before 5:45 p.m. on 03/09.",
        "The file lives at example.org/Reports/Q3_final.PDF, all caps after the slash.",
        "NASA and the F.B.I. both confirmed the launch window for GPT-driven telemetry.",
    ],
}

# Trait value banks used to vary the requested voice across samples.
TRAIT_VALUES: Dict[str, List[str]] = {
    "gender": ["male", "female"],
    "age": ["child", "young adult", "middle-aged adult", "elderly person"],
    "tone": ["warm", "authoritative", "playful", "serious", "soothing"],
    "emotion": ["neutral", "happy", "sad", "angry", "fearful", "surprised"],
    "pace": ["slow", "moderate", "fast"],
    "accent": ["American", "British", "Australian", "Indian"],
    "environment": ["a quiet studio", "a busy street", "a large hall", "over a phone line"],
}
_TRAIT_ORDER = list(TRAIT_VALUES.keys())


_AGE_ADJ = {
    "child": "child", "young adult": "young",
    "middle-aged adult": "middle-aged", "elderly person": "elderly",
}


def render_instruction(tr: Dict[str, str]) -> str:
    """Render a natural-language voice instruction from requested traits."""
    age = _AGE_ADJ.get(tr.get("age", ""), tr.get("age", ""))
    who = " ".join(x for x in (age, tr.get("gender")) if x)
    who = f"{who} voice" if who else "voice"
    article = "An" if who[0] in "AEIOUaeiou" else "A"
    parts = [f"{article} {who}"]
    if tr.get("tone"):
        parts.append(f"with a {tr['tone']} tone")
    if tr.get("emotion") and tr["emotion"] != "neutral":
        parts.append(f"sounding {tr['emotion']}")
    if tr.get("pace"):
        parts.append(f"speaking at a {tr['pace']} pace")
    if tr.get("accent"):
        parts.append(f"with a{'n' if tr['accent'][0] in 'AEIOU' else ''} {tr['accent']} accent")
    if tr.get("environment"):
        parts.append(f"as if recorded in {tr['environment']}")
    return ", ".join(parts) + "."


def _traits_for(i: int, keys: Sequence[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, key in enumerate(keys):
        vals = TRAIT_VALUES[key]
        out[key] = vals[(i + k * 7) % len(vals)]
    return out


def build_corpus(n: int, *, categories: Sequence[str] = tuple(TEXTS.keys()),
                 traits: Sequence[str] = tuple(_TRAIT_ORDER), prefix: str = "vb") -> List[Sample]:
    """Generate ``n`` deterministic samples spread across ``categories``."""
    if n < 1:
        raise ValueError("n must be >= 1")
    cats = list(categories)
    samples: List[Sample] = []
    for i in range(n):
        cat = cats[i % len(cats)]
        texts = TEXTS[cat]
        text = texts[(i // len(cats)) % len(texts)]
        tr = _traits_for(i, traits)
        samples.append(Sample(
            id=f"{prefix}-{i:05d}", text=text, instruction=render_instruction(tr),
            traits=tr, category=cat, difficulty=(i // (len(cats) * 3)) % 4,
        ))
    return samples


def build_splits(n: int, *, holdout_every: int = 5, **kw):
    """Split a corpus into a public and a held-out partition, deterministically.

    Every ``holdout_every``-th sample goes to the held-out set. Keep the held-out split
    private so future models cannot be implicitly tuned against a fixed public set; a
    large public-minus-holdout score gap signals overfitting.

    Returns ``(public, holdout)`` lists of :class:`Sample`.
    """
    if holdout_every < 2:
        raise ValueError("holdout_every must be >= 2")
    corpus = build_corpus(n, **kw)
    public, holdout = [], []
    for i, s in enumerate(corpus):
        (holdout if i % holdout_every == 0 else public).append(s)
    return public, holdout
