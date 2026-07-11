# The audio-LLM judge

Holistic adherence (`tone`) and **naturalness** cannot be read off the waveform — they
require listening. `vocencebench` scores them with an audio-LLM that hears two clips of the
same script and decides which better satisfies one stated dimension. The judge is
deliberately engineered against the known failure modes of LLM-as-judge, and it is used
sparingly: everything objective is handled by free deterministic [probes](probes.md), so
the judge is reserved for what genuinely needs an ear.

Implementation: [`vocencebench/judge/`](../vocencebench/judge/), rubrics in
[`vocencebench/prompts.py`](../vocencebench/prompts.py).

## Comparison protocol

Each comparison sends the judge four text segments interleaved with the two audio clips: a
**system** role; an **intro** stating the dimension, the requested value, the rubric, and
the bias rules, shown before clip A; a **mid** segment between the clips; and an **outro**
requesting the verdict after clip B. The judge must reason about each clip independently,
then compare, then emit strict JSON:

```json
{"reasoning_a": "...", "reasoning_b": "...", "comparison": "...",
 "score_a": 0-3, "score_b": 0-3, "winner": "a|b|tie", "confidence": 0-1}
```

The `0–3` per-clip scores are what the aggregation consumes (normalised to `[0, 1]` as
`score / 3`); the `reasoning_a` / `reasoning_b` fields give an auditable, per-clip rationale
for every verdict, not just a bare label.

## Reliability engineering

- **Order-swap.** A comparison can be run in *both* audio orders; a verdict counts as
  *consistent* only if both orders agree, otherwise it is recorded as a tie and flagged
  `consistent=False`. This neutralises position bias, the dominant LLM-judge failure on
  close calls. (`swap` on the `Judge`, or `swap_eval=True` at the pair/benchmark level.)
- **vote@k.** With `votes > 1` the judge is sampled `k` times per order at low temperature
  and majority-voted; the spread doubles as an uncertainty signal.
- **Bias rules in the prompt.** The judge is told to ignore recording quality, ignore base
  timbre/identity, and not to reward exaggerated delivery unless the dimension calls for it.
  The transcript is marked information-only to blunt prompt injection.
- **Blinded to identity.** In adherence mode the judge is given the *requested* value and
  asked which clip realises it better — never which clip came from which system.
- **Determinism.** The default temperature is `0`; for reproducible scoring, pin the judge
  model version and prompt version and compare results only within a version boundary.

## Two evaluation modes (and their cost)

A duel scores several judged dimensions (`tone` + `naturalness`, plus any holistic
extensions). There are two ways to obtain them, trading reliability against cost:

| Mode | Calls per duel | Trade-off |
|------|----------------|-----------|
| **Per-dimension** (`all_at_once=False`) | one focused call per judged dimension | most reliable — full attention per dimension; more audio uploads |
| **Combined** (`all_at_once=True`) | **one** call scoring every judged dimension + naturalness | cheapest — the audio is sent once; the judge splits attention across dimensions |

`assess_all` implements the combined path: it sends both clips once and returns
`{"traits": {name: Verdict}, "naturalness": Verdict}` in a single JSON, retrying up to three
times if the response is unparseable or misses a dimension. Because objective traits are
already handled by probes, even the per-dimension path only pays for the *holistic*
dimensions — which is what keeps judge cost bounded.

## Backends

Any model that reasons over raw audio works. A backend is a single class implementing
`compare(parts, audio_a, audio_b, *, temperature) -> dict` (plus `raw(...)` for the combined
path):

```python
# Hosted — strongest reasoning, native audio input.
judge = vb.Judge.gemini(model="gemini-3.1-pro-preview")

# Self-hosted — reproducible, offline; any OpenAI-compatible audio endpoint (e.g. vLLM).
judge = vb.Judge.local(base_url="http://localhost:8003", model="Qwen/Qwen2.5-Omni-7B")
```

**One config change** swaps a hosted judge for a local one — same `Judge` API, same rubrics,
same verdict schema. Use a hosted model where reasoning quality matters most and a
self-hosted model where scores must be reproducible across machines without an external
dependency.

The hosted backend also tracks exact token usage (`backend.usage`) and cost
(`backend.cost_usd()`) from the provider's own accounting, so per-evaluation spend is
measurable to the token rather than estimated.

## Direct use

```python
# One dimension, one pair:
v = judge.compare(text, instruction, wav_a, wav_b, dimension="tone")   # trait or "naturalness"

# Absolute adherence on a named trait:
v = judge.adherence(text, instruction, "tone", "authoritative", wav_a, wav_b)

# Every judged dimension + naturalness in one call:
res = judge.assess_all(text, instruction, traits, wav_a, wav_b)        # cost-optimised
```

Each returns a `Verdict` (`winner`, `score_a`, `score_b`, `reasoning_a`, `reasoning_b`,
`consistent`, `confidence`). In normal use you do not call the judge directly — `benchmark`
and `evaluate_pair` orchestrate probes, judge, gate, and aggregation for you.
