"""LLM-driven corpus generator.

Produces a varied, natural benchmark: for each (trait, level, difficulty) cell an LLM
writes several natural-language voice instructions each paired with a fitting script at
the requested reading difficulty. One-factor-at-a-time (OFAT) — each item requests a
single trait so its adherence is cleanly attributable.

The output is frozen to JSONL and pinned by content hash, so results are reproducible
without re-generating (generation is stochastic; the *frozen* corpus is the artifact).
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Optional, Sequence

from vocencebench import traits as _traits
from vocencebench.schema import Sample

DIFFICULTIES = ("easy", "normal", "hard")
_DIFF_GUIDE = {
    "easy": "a short, simple sentence of common everyday words (6-12 words).",
    "normal": "a natural everyday sentence of moderate length (12-20 words).",
    "hard": ("a sentence that is genuinely hard to read aloud: pick ONE challenge — a"
             " tongue-twister, or numbers/dates/currency/emails/URLs, or foreign words"
             " with code-switching, or deeply nested clauses, or dense paralinguistic"
             " cues (interjections, stuttering, CAPS emphasis, elongation)."),
}
_DIFF_IDX = {"easy": 0, "normal": 1, "hard": 2}

_SYSTEM = (
    "You write items for a prompt-controllable text-to-speech (PromptTTS) evaluation. Each"
    " item is a natural-language VOICE INSTRUCTION plus a SCRIPT (the sentence to be"
    " spoken). Instructions must read like a real user request and vary in phrasing. The"
    " script must suit the requested voice but must NEVER state the trait in words (never"
    ' write "he said angrily" or "in a British accent"); the trait must come through only'
    " in how it is spoken. Output strict JSON."
)


def _request_phrase(trait: str, value: str) -> str:
    if _traits.get(trait).numeric:      # age in years
        return f"a {value}-year-old person's voice"
    return f'a voice with {trait} = "{value}"'


def _cell_prompt(trait: str, value: str, difficulty: str, n: int) -> str:
    return (
        f"Produce {n} DIVERSE items requesting {_request_phrase(trait, value)}.\n"
        f"- Vary the instruction wording across the {n} items (do not reuse a template).\n"
        f"- Mention ONLY the {trait} attribute; leave every other voice attribute unstated.\n"
        f"- Each script is {difficulty} difficulty: {_DIFF_GUIDE[difficulty]}\n"
        f"- The script should fit {_request_phrase(trait, value)} in content/register, but"
        f" must not name the {trait}.\n"
        'Return strict JSON: {"items":[{"instruction":"...","text":"..."}, ...]} with exactly'
        f" {n} items."
    )


def _gen_cell(client, model: str, trait: str, value: str, difficulty: str, n: int,
              retries: int = 3) -> List[dict]:
    from google.genai import types
    from vocencebench.judge.base import extract_json
    out: List[dict] = []
    for _ in range(retries):
        resp = client.models.generate_content(
            model=model, contents=[_cell_prompt(trait, value, difficulty, n)],
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM, temperature=1.0,
                response_mime_type="application/json"),
        )
        obj = extract_json(resp.text or "")     # fence-tolerant, with regex fallback
        items = obj.get("items", []) if isinstance(obj, dict) else (obj if isinstance(obj, list) else [])
        out = [{"instruction": str(it["instruction"]).strip(), "text": str(it["text"]).strip()}
               for it in items
               if isinstance(it, dict) and it.get("instruction") and it.get("text")]
        if out:                                 # retry only when the cell came back empty
            break
    return out[:n]


def generate_corpus(
    *,
    trait_names: Optional[Sequence[str]] = None,
    difficulties: Sequence[str] = DIFFICULTIES,
    seeds: int = 2,
    model: str = "gemini-3.1-pro-preview",
    api_key: Optional[str] = None,
    progress=None,
) -> List[Sample]:
    """Generate an OFAT corpus across every (trait, level, difficulty) cell.

    ``seeds`` items per cell. Returns a list of :class:`Sample`. Requires ``google-genai``
    and a Gemini API key (env ``GEMINI_API_KEY`` if not passed).
    """
    from google import genai
    client = genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY"))
    names = list(trait_names) if trait_names else _traits.names()

    samples: List[Sample] = []
    idx = 0
    for tname in names:
        trait = _traits.get(tname)
        for value in trait.values:
            for diff in difficulties:
                items = _gen_cell(client, model, tname, value, diff, seeds)
                for it in items:
                    samples.append(Sample(
                        id=f"vb-{idx:05d}", text=it["text"], instruction=it["instruction"],
                        traits={tname: value}, category="general", difficulty=_DIFF_IDX[diff]))
                    idx += 1
                if progress:
                    progress(tname, value, diff, len(items))
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
