"""LLM-driven corpus generator.

Each item is a FULLY-SPECIFIED voice: one flowing natural-language ``instruction`` that
weaves in ALL eight controllable traits (age, gender, emotion, pitch, loudness, pace,
accent, tone), paired with a ``text`` script at a target reading difficulty that suits
the persona but never names a trait in words. Every trait mentioned in the instruction is
also emitted as a structured field, so each clip can be probed on every trait.

Trait combinations are sampled deterministically (seeded, balanced so each value appears
about equally). The LLM wording is stochastic, so the *frozen* corpus — written to JSONL
and pinned by content hash — is the reproducible artifact, not the generation run.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence

from vocencebench import traits as _traits
from vocencebench.schema import Sample

# Traits every item specifies, in the order they are presented to the writer.
_TRAIT_ORDER = ("age", "gender", "emotion", "pitch", "loudness", "pace", "accent", "tone")

# Age is continuous (scored within +/- tolerance); sample from a pool spanning child to
# elderly, richer than the registry's reference points for natural variety.
_AGE_POOL = (8, 13, 20, 25, 30, 38, 45, 55, 60, 70, 78)

DIFFICULTIES = ("easy", "normal", "hard")
# How the 3 difficulty levels are mixed across the corpus (hard = fewer, they are stress tests).
_DIFF_WEIGHTS = {"easy": 0.35, "normal": 0.40, "hard": 0.25}
_DIFF_IDX = {"easy": 0, "normal": 1, "hard": 2}
_DIFF_GUIDE = {
    "easy": "a short, simple sentence of common everyday words (6-12 words).",
    "normal": "a natural everyday sentence of moderate length (12-22 words).",
    "hard": ("a sentence that is genuinely hard to read aloud: pick ONE challenge — a"
             " tongue-twister, or numbers/dates/currency/emails/URLs, or foreign words"
             " with code-switching, or deeply nested clauses."),
}

_SYSTEM = (
    "You write items for a prompt-controllable text-to-speech (PromptTTS) benchmark. Each"
    " item is one flowing natural-language VOICE INSTRUCTION describing the whole voice,"
    " plus a SCRIPT (the sentence to be spoken). The instruction must read like a real"
    " person requesting exactly that voice, weaving every requested attribute in naturally"
    " — never a mechanical checklist of the label words. The script must suit the persona"
    ' but must NEVER state a trait in words (never write "he said angrily" or "in a British'
    ' accent"); the trait must come through only in how it is spoken. Output strict JSON.'
)


# --------------------------------------------------------------------- spec sampling
def _balanced_deck(values: Sequence, n: int, rng: random.Random) -> List:
    """A length-n list where each value appears ~equally, order shuffled."""
    reps = (n + len(values) - 1) // len(values)
    deck = list(values) * reps
    rng.shuffle(deck)
    return deck[:n]


def sample_specs(n: int, seed: int = 20250710) -> List[dict]:
    """Deterministically sample ``n`` fully-specified trait combinations.

    Each of the 8 traits is drawn from an independently-shuffled balanced deck, so every
    value gets near-equal coverage while combinations stay varied and uncorrelated.
    """
    rng = random.Random(seed)
    columns = {
        "age": _balanced_deck(_AGE_POOL, n, rng),
        "gender": _balanced_deck(_traits.get("gender").values, n, rng),
        "emotion": _balanced_deck(_traits.get("emotion").values, n, rng),
        "pitch": _balanced_deck(_traits.get("pitch").values, n, rng),
        "loudness": _balanced_deck(_traits.get("loudness").values, n, rng),
        "pace": _balanced_deck(_traits.get("pace").values, n, rng),
        "accent": _balanced_deck(_traits.get("accent").values, n, rng),
        "tone": _balanced_deck(_traits.get("tone").values, n, rng),
    }
    # Difficulty deck honoring the weights.
    diff_deck: List[str] = []
    for d, w in _DIFF_WEIGHTS.items():
        diff_deck += [d] * round(w * n)
    while len(diff_deck) < n:
        diff_deck.append("normal")
    diff_deck = diff_deck[:n]
    rng.shuffle(diff_deck)

    specs = []
    for i in range(n):
        spec = {t: columns[t][i] for t in _TRAIT_ORDER}
        spec["difficulty"] = diff_deck[i]
        specs.append(spec)
    return specs


# ------------------------------------------------------------------- item generation
def _attr_lines(spec: dict) -> str:
    out = []
    for t in _TRAIT_ORDER:
        v = f"{spec[t]} years old" if t == "age" else spec[t]
        out.append(f"  - {t}: {v}")
    return "\n".join(out)


def _item_prompt(spec: dict) -> str:
    d = spec["difficulty"]
    return (
        "Create ONE TTS evaluation item for a voice with EXACTLY these 8 attributes "
        "(weave ALL of them into the instruction naturally; add NO other attributes; omit NONE):\n"
        f"{_attr_lines(spec)}\n\n"
        "1) instruction: one vivid, flowing 1-2 sentence description of this voice, as a real "
        "person would request it (e.g. 'A bubbly twenty-year-old American woman with a bright, "
        "high voice, speaking quickly and cheerfully at a normal volume'). Weave in ALL 8 "
        "attributes but do NOT list them mechanically or reuse the raw label words as a checklist.\n"
        f"2) text: {_DIFF_GUIDE[d]} It must SUIT this persona/emotion and must NOT name any of "
        "the attributes in words.\n"
        'Return strict JSON: {"instruction":"...","text":"..."}'
    )


def _gen_item(client, model: str, spec: dict, retries: int = 4) -> dict:
    from google.genai import types
    from vocencebench.judge.base import extract_json
    o: dict = {}
    for _ in range(retries):
        resp = client.models.generate_content(
            model=model, contents=[_item_prompt(spec)],
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM, temperature=1.0,
                response_mime_type="application/json"),
        )
        cand = extract_json(resp.text or "")
        if isinstance(cand, dict) and cand.get("instruction") and cand.get("text"):
            o = cand
            break
    return {"instruction": str(o.get("instruction", "")).strip(),
            "text": str(o.get("text", "")).strip()}


def _spec_traits(spec: dict) -> Dict[str, str]:
    return {t: str(spec[t]) for t in _TRAIT_ORDER}


def generate_corpus(
    *,
    n: int = 500,
    seed: int = 20250710,
    model: str = "gemini-3.1-pro-preview",
    api_key: Optional[str] = None,
    workers: int = 8,
    progress=None,
) -> List[Sample]:
    """Generate ``n`` fully-specified multi-trait items.

    Deterministic trait combos (via ``seed``); the LLM writes the instruction + script for
    each. Returns a list of :class:`Sample`. Requires ``google-genai`` and a Gemini API key
    (env ``GEMINI_API_KEY`` if not passed). ``progress(done, total)`` is called as items land.
    """
    from google import genai
    client = genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY"))
    specs = sample_specs(n, seed)

    def _one(spec):
        return spec, _gen_item(client, model, spec)

    samples: List[Sample] = []
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_one, specs))   # ordered -> reproducible ids
    for i, (spec, item) in enumerate(results):
        if not item["instruction"] or not item["text"]:
            continue                            # skip any that never filled after retries
        samples.append(Sample(
            id=f"vb-{i:05d}", text=item["text"], instruction=item["instruction"],
            traits=_spec_traits(spec), category="general",
            difficulty=_DIFF_IDX[spec["difficulty"]]))
        done += 1
        if progress:
            progress(done, n)
    return samples


# ------------------------------------------------------------------- freeze / pin
def corpus_hash(samples: Sequence[Sample]) -> str:
    raw = json.dumps([s.model_dump() for s in samples], sort_keys=True,
                     separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def split_holdout(samples: Sequence[Sample], *, holdout_every: int = 5):
    """Deterministic public/held-out split (every k-th item is held out)."""
    public, holdout = [], []
    for i, s in enumerate(samples):
        (holdout if i % holdout_every == 0 else public).append(s)
    return public, holdout


def load_benchmark(name: str = "benchmark_v1") -> List[Sample]:
    """Load a frozen benchmark dataset shipped with the package."""
    from pathlib import Path
    from vocencebench.schema import load_dataset
    path = Path(__file__).parent / "datasets" / f"{name}.jsonl"
    return load_dataset(path)
