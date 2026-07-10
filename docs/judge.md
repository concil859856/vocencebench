# The audio-LLM judge

Holistic traits (tone, environment) and naturalness are scored by an audio-LLM that
listens to two clips of the same text and decides which better satisfies one stated
dimension. The judge is deliberately engineered against the known failure modes of
LLM judging.

## Comparison protocol

For each comparison the judge receives four text segments interleaved with the two
clips: a **system** role, an **intro** (the dimension, the requested value, the rubric,
the bias rules) shown before clip A, a **mid** segment between the clips, and an
**outro** asking for the verdict after clip B. It must reason about each clip
independently, then compare, then emit strict JSON:

```json
{"reasoning_a": "...", "reasoning_b": "...", "comparison": "...",
 "score_a": 0-3, "score_b": 0-3, "winner": "a|b|tie", "confidence": 0-1}
```

## Reliability engineering

- **Order-swap.** Every comparison is run in both audio orders. A verdict counts as
  *consistent* only if both orders agree; otherwise it is recorded as a tie and flagged
  `consistent=False`. This neutralises position bias, the dominant LLM-judge failure on
  close calls.
- **vote@k.** With `votes>1` the judge is sampled `k` times per order at a low
  temperature and majority-voted; spread is a usable uncertainty signal.
- **Bias rules in the prompt.** The judge is told to ignore recording quality, base
  timbre/identity, and not to reward exaggerated delivery unless the dimension calls
  for it. The transcript is marked information-only to blunt prompt injection.
- **Blinded to the target.** In adherence mode the judge is given the *requested*
  value and asked which clip realises it better — it is never told which clip is the
  model versus the reference.

## Backends

Any judge that reasons over raw audio works:

```python
Judge.gemini(model="gemini-3.1-pro")                    # hosted, best reasoning
Judge.local(base_url="http://localhost:8003",           # self-hosted, reproducible
            model="Qwen/Qwen2.5-Omni-7B")
```

Use a hosted model for reference/calibration and a self-hosted model where scores must
be reproducible across machines. Adding a backend is a single class implementing
`compare(parts, audio_a, audio_b, *, temperature) -> dict`.
