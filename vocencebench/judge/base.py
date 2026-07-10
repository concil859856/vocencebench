"""Judge wrapper: order-swap + vote@k over a raw backend.

A *backend* knows how to send one comparison (four text segments interleaved with two
audio clips) to a specific model and return a parsed verdict dict. The :class:`Judge`
wrapper adds the reliability engineering that makes an audio-LLM verdict trustworthy:

* **order-swap** — every comparison is run in both audio orders; a result only counts
  as consistent when both orders agree, otherwise it is a tie.
* **vote@k** — optionally sample the judge ``k`` times per order and majority-vote,
  bounding run-to-run noise.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Protocol

from vocencebench.audio import Audio
from vocencebench.prompts import PromptParts, adherence, naturalness
from vocencebench.schema import TIE, WINNER_A, WINNER_B, Verdict


class Backend(Protocol):
    """Sends one pairwise comparison to a model and returns a parsed verdict dict."""

    def compare(self, parts: PromptParts, audio_a: bytes, audio_b: bytes,
                *, temperature: float) -> Dict[str, Any]:
        ...


def extract_json(raw: str) -> Optional[dict]:
    """Best-effort JSON object extraction from a model's text response."""
    if not raw:
        return None
    # Strip a leading ```json / ``` fence if present.
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s).rstrip("`").strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    # Grab the outermost {...}.
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def salvage_verdict(raw: str) -> Optional[dict]:
    """Recover a verdict from a truncated/malformed response by regex.

    Verbose judges sometimes overrun the token budget mid-JSON; as long as the winner
    (and ideally the scores) were emitted, we keep a usable decision instead of dropping
    the whole comparison.
    """
    if not raw:
        return None
    w = re.search(r'"winner"\s*:\s*"?(a|b|tie|1|2|0)"?', raw, re.IGNORECASE)
    if not w:
        return None
    sa = re.search(r'"score_a"\s*:\s*(\d)', raw)
    sb = re.search(r'"score_b"\s*:\s*(\d)', raw)
    return {
        "winner": w.group(1), "reasoning_a": "", "reasoning_b": "",
        "comparison": raw[:400], "confidence": 0.0,
        "score_a": int(sa.group(1)) if sa else None,
        "score_b": int(sb.group(1)) if sb else None,
    }


def _flip(winner: str) -> str:
    return WINNER_A if winner == WINNER_B else WINNER_B if winner == WINNER_A else TIE


class Judge:
    """Audio-LLM judge over any :class:`Backend`, with order-swap and voting."""

    def __init__(self, backend: Backend, *, votes: int = 1, swap: bool = True,
                 temperature: float = 0.0):
        if votes < 1:
            raise ValueError("votes must be >= 1")
        self.backend = backend
        self.votes = votes
        self.swap = swap
        self.temperature = temperature if votes == 1 else max(temperature, 0.3)

    # -- factories ---------------------------------------------------------------
    @classmethod
    def local(cls, base_url: str, model: str, api_key: str = "", **kw) -> "Judge":
        """Judge backed by a local OpenAI-compatible endpoint (e.g. a vLLM server)."""
        from vocencebench.judge.openai_compat import OpenAICompatBackend
        return cls(OpenAICompatBackend(base_url=base_url, model=model, api_key=api_key), **kw)

    @classmethod
    def gemini(cls, model: str = "gemini-3.1-pro", api_key: Optional[str] = None, **kw) -> "Judge":
        """Judge backed by a hosted Gemini model with native audio input."""
        from vocencebench.judge.gemini import GeminiBackend
        return cls(GeminiBackend(model=model, api_key=api_key), **kw)

    # -- public comparisons ------------------------------------------------------
    def adherence(self, text: str, instruction: str, trait: str, value: str,
                  audio_a: Audio, audio_b: Audio) -> Verdict:
        return self._compare(adherence(text, instruction, trait, value), trait, audio_a, audio_b)

    def naturalness(self, text: str, audio_a: Audio, audio_b: Audio,
                    category: str = "general") -> Verdict:
        return self._compare(naturalness(text, category), "naturalness", audio_a, audio_b)

    def compare(self, text: str, instruction: str, audio_a: Audio, audio_b: Audio,
                dimension: str) -> Verdict:
        """Generic entry: ``dimension`` is a trait name, 'naturalness', or a category."""
        from vocencebench.prompts import NATURALNESS_RUBRICS
        if dimension == "naturalness" or dimension in NATURALNESS_RUBRICS:
            cat = "general" if dimension == "naturalness" else dimension
            return self.naturalness(text, audio_a, audio_b, category=cat)
        value = ""  # caller may not know the requested value in generic mode
        return self.adherence(text, instruction, dimension, value, audio_a, audio_b)

    # -- core --------------------------------------------------------------------
    def _compare(self, parts: PromptParts, dimension: str, audio_a: Audio, audio_b: Audio) -> Verdict:
        from vocencebench.audio import to_wav_bytes
        wav_a, wav_b = to_wav_bytes(audio_a), to_wav_bytes(audio_b)

        # (audio_first, is_flipped): normal order, plus swapped order when enabled.
        orders = [(wav_a, wav_b, False)]
        if self.swap:
            orders.append((wav_b, wav_a, True))

        tally = {WINNER_A: 0, WINNER_B: 0, TIE: 0}
        sa: List[int] = []
        sb: List[int] = []
        reasons: List[str] = []
        confs: List[float] = []
        per_order_winner: List[str] = []
        for first, second, flipped in orders:
            for _ in range(self.votes):
                res = self.backend.compare(parts, first, second, temperature=self.temperature)
                w = res.get("winner", TIE)
                if flipped:
                    w = _flip(w)
                    # scores were reported for (first=b, second=a); un-flip.
                    res_sa, res_sb = res.get("score_b"), res.get("score_a")
                else:
                    res_sa, res_sb = res.get("score_a"), res.get("score_b")
                tally[w] += 1
                if res_sa is not None:
                    sa.append(res_sa)
                if res_sb is not None:
                    sb.append(res_sb)
                if res.get("comparison"):
                    reasons.append(res["comparison"])
                confs.append(res.get("confidence", 0.0))
            per_order_winner.append(_majority(tally, reset=False))

        winner = _argmax_winner(tally)
        # Consistent iff no order produced the opposite decisive winner.
        consistent = not (tally[WINNER_A] > 0 and tally[WINNER_B] > 0)
        if self.swap and not consistent:
            winner = TIE
        return Verdict(
            dimension=dimension,
            winner=winner,
            score_a=int(round(sum(sa) / len(sa))) if sa else None,
            score_b=int(round(sum(sb) / len(sb))) if sb else None,
            reasoning=reasons[0] if reasons else "",
            confidence=round(sum(confs) / len(confs), 3) if confs else 0.0,
            consistent=consistent,
            raw={"tally": dict(tally)},
        )


def _argmax_winner(tally: Dict[str, int]) -> str:
    a, b, t = tally[WINNER_A], tally[WINNER_B], tally[TIE]
    if a > b and a >= t:
        return WINNER_A
    if b > a and b >= t:
        return WINNER_B
    return TIE


def _majority(tally: Dict[str, int], reset: bool) -> str:
    return _argmax_winner(tally)
