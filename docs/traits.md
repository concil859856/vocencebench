# Trait vocabulary

Every controllable voice attribute is declared once, in
[`vocencebench/traits.py`](../vocencebench/traits.py). This registry is the single source
of truth: the corpus generator, the probes, the judge rubrics, and the aggregation all
read it, so **adding or changing a trait is a one-place edit**.

## The registry

| Trait | Kind | Values | Scoring |
|-------|------|--------|---------|
| `gender` | objective | male, female | exact match |
| `age` | objective · numeric | reference points 8, 13, 20, 30, 45, 60, 78 (years) | tolerance band ±8 yr |
| `pace` | objective · ordinal | slow, moderate, fast | ±1 bucket partial credit |
| `pitch` | objective · ordinal | low, medium, high | ±1 bucket partial credit |
| `loudness` | objective · ordinal | quiet, normal, loud | ±1 bucket partial credit |
| `emotion` | objective | neutral, calm, happy, sad, angry, fearful, disgust, surprised | exact match |
| `accent` | objective | American, British, Australian, Indian, Canadian | exact match |
| `tone` | holistic | warm, authoritative, playful, serious, soothing, cheerful, sarcastic, formal | audio-LLM 0–3 rubric |

## Anatomy of a trait

Each entry is a `Trait` dataclass:

```python
Trait(
    name="pace",
    kind="objective",          # "objective" | "holistic" | "hybrid"
    values=("slow", "moderate", "fast"),
    dimension="whether the speaking rate matches the requested pace",
    ordinal=True,              # values lie on an ordered scale
    numeric=False,             # continuous numeric attribute (age)
    tolerance=0.0,             # ± band that scores 1.0, for numeric traits
)
```

- **`kind` decides who scores it.**
  - `objective` — measured from the signal by a deterministic [probe](probes.md). Identical
    across runs and machines.
  - `holistic` — needs listening; scored by the [audio-LLM judge](judge.md) against the
    `dimension` prompt. Only `tone` is holistic.
  - `hybrid` — both a probe and the judge apply (none in the default vocabulary; supported
    for extensions).
- **`dimension`** is the exact natural-language focus handed to the judge ("whether the
  overall tone / delivery style matches the requested tone"), so holistic scoring is
  explicit and auditable.
- **`ordinal`** marks values on an ordered scale. An adjacent-bucket miss (e.g. requested
  *fast*, measured *moderate*) earns partial credit `0.5` rather than `0.0`, because a
  near-miss on a continuum is genuinely closer than a far miss.
- **`numeric` + `tolerance`** mark a continuous attribute. `age` is scored by a regression
  probe against the requested value with a ±`tolerance`-year band scoring `1.0`, decaying
  linearly outside it (see [probes.md](probes.md#age)). The listed values are reference
  points for corpus sampling, not a closed set — any integer age is valid.

## How a trait becomes a desirability

The registry `kind` routes scoring; the result is always a desirability in `[0, 1]`
(see [methodology.md §2](methodology.md#2-step-1--desirability-normalisation)):

```
   objective ordinal   → 1.0 exact · 0.5 adjacent · 0.0 else
   objective categorical → 1.0 match · 0.0 mismatch
   objective numeric   → max(0, 1 − max(0, |m−r| − τ)/τ)
   holistic            → judge 0–3 rubric score ÷ 3
```

## Adding a trait

1. `register(Trait(...))` in `traits.py` with its `kind`, `values`, and `dimension`.
2. If `objective`, add a [probe](probes.md) that reports its `trait` name; if `holistic`,
   the judge picks it up automatically from `dimension`.
3. Regenerate and re-freeze the [corpus](corpus.md) so the new trait is exercised.

Nothing else changes — aggregation and reporting enumerate the registry.

## Open-vocabulary fallback

`traits.get(name)` for an unregistered name returns a synthesised `holistic` trait with the
prompt *"whether the delivery matches the requested {name}"*. This lets a dataset request
an ad-hoc attribute the judge can still reason about, though registered traits with probes
are always preferred for objective, reproducible scoring.
