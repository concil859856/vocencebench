# Deterministic trait probes

A **probe** measures one objective trait directly from the waveform and scores the
requested value against that measurement. Probes are the reproducible backbone of the
adherence score: they are pure functions of the audio, so their output is identical across
runs and machines, they cost nothing per call, and their reasoning is fully inspectable
(the measured value and intermediate features are returned in `detail`).

```python
import vocencebench as vb
probes = vb.default_probes()          # model-free acoustic: pace, loudness, pitch
probes = vb.with_classifiers()        # + gender, emotion, accent, age
```

Each probe implements one method and returns a `ProbeResult`
([`probes/base.py`](../vocencebench/probes/base.py)):

```python
def score(self, sample, audio) -> ProbeResult | None    # None if its trait wasn't requested
```

```python
ProbeResult(trait, requested, measured, matched, score,  detail={...})   # score ∈ [0,1]
```

## Scoring helpers

Two shared functions convert a measurement into a desirability
([`probes/base.py`](../vocencebench/probes/base.py)):

- **`ordinal_score(requested, measured, order)`** — `1.0` for the exact bucket, `0.5` for
  an adjacent bucket, `0.0` otherwise. Encodes ±1 tolerance on ordered scales.
- **`exact_score(requested, measured)`** — `1.0` on an exact match, else `0.0`. For
  unordered categorical traits.

## Model-free acoustic probes

These need only the waveform (`librosa` / `numpy`), no downloaded model, and are fully
deterministic ([`probes/acoustic.py`](../vocencebench/probes/acoustic.py)). Band thresholds
are sensible English defaults and are constructor-overridable — recalibrate them to a
corpus or voice when needed.

### `PaceProbe` — speaking rate

Trims leading/trailing silence, then `words ÷ voiced-duration` (words per second), bucketed
against edges `(2.2 → slow, 3.2 → moderate, else fast)`. Ordinal scoring. `detail` reports
`words_per_sec` and `duration_s`.

### `LoudnessProbe` — level

RMS amplitude in **dBFS**, bucketed `(−30 → quiet, −18 → normal, else loud)`. Ordinal
scoring. `detail` reports `dbfs`.

### `PitchProbe` — fundamental frequency

Median voiced **F0** via `librosa.pyin` (search range 65–400 Hz), bucketed
`(140 → low, 220 → medium, else high)`. Ordinal scoring. `detail` reports `f0_median`.

> **Caveat.** Absolute F0 is confounded with speaker gender — a "high" male and a "low"
> female can overlap in Hz. For gendered corpora prefer within-voice relative bands; the
> default thresholds are a simple, configurable starting point, not a universal truth.

## Classifier probes

These wrap trained audio classifiers to read a categorical trait from the signal
([`probes/classifier.py`](../vocencebench/probes/classifier.py)). Models load lazily, run on
CPU or GPU (`device=-1` / `0`), and are individually toggleable. Each `model_id` is a
sensible default, not a hard requirement — swap in a better checkpoint as they appear.

| Probe | Trait | Default backbone | Classes | Scoring |
|-------|-------|------------------|---------|---------|
| `GenderProbe` | gender | wav2vec2 gender-recognition | male, female | exact |
| `EmotionProbe` | emotion | wav2vec2 speech-emotion (RAVDESS) | 8-class (neutral … surprised) | exact |
| `AccentProbe` | accent | English-accents classifier | American, British, Australian, Indian, Canadian | exact |

`HFClassifierProbe` maps model labels to canonical trait values by case-insensitive
substring (`label_map`), takes the highest-scoring *mappable* prediction from the top-`k`,
and records its `confidence` in `detail`. A missing or incompatible model degrades
gracefully to `score=0.0` with the error in `detail` rather than aborting the sample.

## Age probe (regression)

<a name="age"></a>`AgeProbe` ([`probes/age.py`](../vocencebench/probes/age.py)) uses a
wav2vec2 regression head to estimate apparent age in **years**, then scores it against the
requested value with the trait's tolerance band `τ` (default 8 years):

```
   score = max( 0,  1 − max(0, |measured − requested| − τ) / τ )
```

So an estimate within ±`τ` years scores `1.0`, decaying linearly to `0` at `2τ` off. This
is the numeric desirability from [methodology.md §2](methodology.md#2-step-1--desirability-normalisation).

## Assembling a probe set

```python
from vocencebench.probes import default_probes, with_classifiers

probes = with_classifiers(               # start from the acoustic set, append classifiers
    gender=True, emotion=True, accent=True, age=True,
    device=0,                            # GPU; -1 for CPU
)
```

Any trait *without* a matching probe is routed to the [judge](judge.md) instead — that is
exactly how `tone` (holistic) is handled. Traits *with* a probe are always scored by the
probe, keeping the objective dimensions deterministic and free.

## Validation caveat

A classifier trained on natural speech may drift on synthetic audio. Before trusting a
probe on TTS output, confirm its agreement with human labels on natural speech first — that
is the probe's noise floor. The optional [validation](validation.md) tooling reports this
per trait.
