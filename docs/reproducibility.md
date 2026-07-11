# Reproducibility

`vocencebench` is built so that, given the same inputs, **every evaluator computes the
identical ranking** — a hard requirement for a benchmark whose scores carry weight. This
document lists the determinism guarantees and how to preserve them.

## What is deterministic

| Stage | Guarantee | Mechanism |
|-------|-----------|-----------|
| **Corpus** | every run scores the identical items | generated once, frozen to JSONL, pinned by SHA-256 (`corpus_hash`) |
| **Objective probes** | identical scores across runs and machines | pure functions of the waveform (no sampling, no network) |
| **Aggregation** | identical composites | fixed equal weights, closed-form geometric mean and gate |
| **Bootstrap LCB** | identical confidence bound | fixed seed (`3151662`), fixed `N = 2000`, fixed `α = 0.05` |
| **Decision margin** | identical threshold | closed form `max(0.015, 0.10·(1 − S_inc))`, no free parameters |
| **Held-out split** | identical partition | deterministic "every k-th item" rule |

The only non-deterministic component is the audio-LLM judge (see below); everything else is
a pure function of the frozen corpus and the synthesised audio.

## The frozen corpus

Language-model generation is stochastic, so the **frozen corpus is the artifact**, not the
generator. The `.jsonl` and its `.sha256` are committed together:

```python
from vocencebench import load_benchmark
from vocencebench.corpus_llm import corpus_hash

samples = load_benchmark("benchmark_v1")
assert corpus_hash(samples) == open(".../benchmark_v1.sha256").read().strip()
```

Regenerating the corpus produces *different* items and therefore a *different* hash; that is
expected. A benchmark result is only comparable to another that used the same pinned corpus.

## The judge is the one stochastic dimension

The judge scores only the holistic dimensions (`tone`) and naturalness. To keep it as
reproducible as possible:

- Run at **temperature 0** (the default).
- **Pin the model version and the prompt version**, and log both with every run. Judge
  scores are comparable *only within a version boundary* — a new model or a reworded rubric
  is a new benchmark.
- Use **order-swap** so position bias cannot flip a verdict, and **vote@k** to bound
  residual sampling noise.
- For fully offline, machine-independent scoring, use a **self-hosted** judge
  (`Judge.local`) with a pinned checkpoint.

Because the judge touches only a minority of dimensions and its scores pass through the same
geometric aggregation as the deterministic probes, its residual noise is bounded and
further suppressed by the [paired bootstrap LCB](methodology.md#6-step-4--the-paired-bootstrap-lower-confidence-bound):
a verdict only stands if it survives a pessimistic reading of that noise.

## Reproducing a decision from published scores

The [decision](methodology.md#8-the-decision-rule) is a pure function of the per-sample
composites. A third party who has the published per-sample scores can recompute the winner
without re-running any model or judge:

```python
from vocencebench.decide import decide
decision = decide(head2head)          # same seed → same LCB → same verdict, on any machine
```

## Preserving reproducibility when extending

- Add traits via the [registry](traits.md) and **re-freeze + re-pin** the corpus.
- Keep new probes pure functions of the audio.
- If you change weights, seed, margin, or the judge model/prompt, treat the result as a
  **new benchmark version** — do not compare across the boundary.
