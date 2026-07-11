# Corpus: generation, schema, and freezing

The benchmark corpus is a set of **fully-specified, multi-trait evaluation items**. Each
item pins a complete voice — all eight controllable traits at once — expressed both as a
single natural-language instruction and as structured fields, paired with a script to
speak at a controlled reading difficulty.

A frozen corpus ships with the package
([`vocencebench/datasets/benchmark_v1.jsonl`](../vocencebench/datasets/benchmark_v1.jsonl))
and is loaded with `vb.load_benchmark("benchmark_v1")`.

## Dataset schema

A dataset is a JSONL file, one `Sample` object per line
([`vocencebench/schema.py`](../vocencebench/schema.py)):

| Field | Type | Meaning |
|-------|------|---------|
| `id` | string | Unique sample id (`vb-00042`). Names audio files (`<id>.wav`) for CLI eval. |
| `text` | string | The script the model must speak. |
| `instruction` | string | The natural-language voice instruction (the "prompt") given to the model. |
| `traits` | object | Requested attributes, one per registered trait, e.g. `{"gender":"female","age":"20", ...}`. |
| `category` | string | Naturalness grouping for rubric selection (default `general`). |
| `difficulty` | int | Reading-difficulty tier: `0` easy · `1` normal · `2` hard. |
| `language` | string | Language code (default `en`). |

### Example

```json
{"id": "vb-00001",
 "text": "Oh my gosh, you actually brought the puppy to the dorm room? I completely thought you were joking!",
 "instruction": "Generate a voice for a playful twenty-year-old woman with an Indian accent who is loudly expressing her surprise. Her voice should have a high pitch and maintain a steady, moderate pacing.",
 "traits": {"age": "20", "gender": "female", "emotion": "surprised", "pitch": "high",
            "loudness": "loud", "pace": "moderate", "accent": "Indian", "tone": "playful"},
 "category": "general", "difficulty": 1, "language": "en"}
```

## The design contract

Every corpus item obeys three rules that make it a good adherence *and* naturalness probe:

1. **Fully specified.** All eight traits are present as fields, so every clip can be scored
   on every trait — maximal, balanced coverage rather than one-attribute-at-a-time.
2. **Instruction ≡ fields.** Each trait that appears in `traits` is also woven into the
   `instruction`, phrased as a real person would request it ("...maintain a remarkably warm
   tone despite his heartache") rather than as a mechanical checklist. The two views never
   disagree.
3. **The script never names the trait.** The `text` suits the persona but must never state
   an attribute in words (never "he said angrily", never "in a British accent") — the trait
   has to be realised in *how* the clip is spoken, or it does not count.

## Reading-difficulty tiers

The `text` is generated at one of three difficulties so the corpus stresses intelligibility
as well as control:

| Tier | `difficulty` | Character |
|------|--------------|-----------|
| easy | 0 | short, common everyday words |
| normal | 1 | a natural everyday sentence of moderate length |
| hard | 2 | genuinely hard to read aloud — tongue-twisters, or dense numbers / dates / currency / emails / URLs, or foreign words, or deeply nested clauses |

The `hard` tier is deliberately adversarial for ASR and prosody (e.g. *"...exactly
£4,392,810.75 must be transferred ... by Thursday, February 29th, 2024, or proceedings will
commence via www.litigation-registry.co.uk/file_9901"*), giving the [intelligibility
gate](methodology.md#4-the-intelligibility-gate) something real to catch.

## Generation

The generator is [`vocencebench/corpus_llm.py`](../vocencebench/corpus_llm.py):

```python
from vocencebench.corpus_llm import generate_corpus, corpus_hash
from vocencebench.schema import save_dataset

samples = generate_corpus(n=500, seed=20250710)     # needs GEMINI_API_KEY
save_dataset(samples, "benchmark_v1.jsonl")
print(corpus_hash(samples))                         # sha256:...
```

The pipeline is two-stage — a *deterministic* trait sampler, then a *stochastic* language
model that writes the instruction and script for each sampled combination:

1. **Balanced, seeded trait sampling** (`sample_specs`). For each of the eight traits, an
   independently-shuffled *balanced deck* assigns values so every value appears about
   equally, while the eight columns are drawn independently so combinations stay varied and
   uncorrelated. Difficulty follows a fixed easy/normal/hard weighting (0.35 / 0.40 / 0.25).
   With a fixed `seed`, the exact set of trait combinations is reproducible.
2. **Instruction + script writing.** For each combination the language model is asked for a
   vivid instruction weaving in all eight traits and a fitting script at the target
   difficulty, returned as strict JSON. Generation is retried until non-empty; any item
   that never fills is dropped.

Coverage for the shipped `n = 500` sample (496 non-empty) is near-uniform: ~100 items per
accent, ~62 per emotion and per tone, balanced pitch/loudness/pace, ages spread 8→78.

## Freezing and hash-pinning

Language-model generation is stochastic, so the **frozen corpus — not the generator — is
the artifact**. The workflow:

```python
h = corpus_hash(samples)                # canonical sha256 over the sorted records
open("benchmark_v1.sha256", "w").write(h + "\n")
```

The `.jsonl` and its `.sha256` are committed together; `corpus_hash` recomputed on load
must match the pin. This guarantees every evaluator scores the *identical* items, which is a
precondition for the reproducible ranking (see [reproducibility.md](reproducibility.md)).

## Held-out split

```python
from vocencebench.corpus_llm import split_holdout
public, holdout = split_holdout(samples, holdout_every=5)   # every 5th item is held out
```

The split is deterministic. Keep the held-out partition private: a large gap between public
and held-out scores signals a model that has been tuned against the public set rather than
genuinely improved.

## Alternative: deterministic template corpus

For fully offline, dependency-free generation, [`vocencebench/corpus.py`](../vocencebench/corpus.py)
(`build_corpus`) renders samples from fixed text banks and trait templates with no language
model — useful for smoke tests and CI. The language-model generator above produces the
richer, more natural instructions used for the shipped benchmark.
