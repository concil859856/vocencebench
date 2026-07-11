<h1 align="center">vocencebench</h1>

<p align="center">
  <b>A reproducible, gaming-resistant benchmark for prompt-controllable text-to-speech.</b>
</p>

<p align="center">
  <a href="#install">Install</a> ·
  <a href="#quickstart">Quickstart</a> ·
  <a href="#the-evaluation-pipeline">Pipeline</a> ·
  <a href="#methodology">Methodology</a> ·
  <a href="docs/">Documentation</a>
</p>

---

## Abstract

Prompt-controllable text-to-speech (**PromptTTS**) systems synthesise speech from *two*
inputs: a script to read and a natural-language *voice instruction* ("a calm elderly
British man, speaking slowly and warmly"). Evaluating them therefore requires answering
two orthogonal questions at once — **does the voice match what was requested** (trait
adherence), and **is the delivery natural and intelligible** (quality) — and then
collapsing many heterogeneous sub-scores into a *single, defensible verdict* when
ranking two systems.

`vocencebench` is a self-contained framework for exactly this. It scores adherence with
deterministic signal-processing **probes** wherever a trait is physically measurable and
with an order-swapped **audio-LLM judge** wherever it is holistic; it gates on
intelligibility with ASR word-error-rate; and it fuses everything through a
**non-compensatory geometric aggregation** guarded by a **paired bootstrap confidence
bound** and a **perceptually-grounded decision margin**. The result is a ranking that is
*reproducible* (same inputs → same verdict, bit-for-bit), *hard to game* (no single
dimension can dominate or be traded off), and *statistically honest* (a winner is
declared only when the improvement is both real and audible).

## Design principles

1. **Two numbers before one.** Adherence and naturalness are measured separately and
   only combined at the final decision — a model can nail the requested voice yet read
   flatly, or read beautifully off-prompt. Neither failure hides.
2. **Measure, don't guess.** Every objective trait (pace, pitch, loudness, gender,
   emotion, accent, age) is read from the signal by a deterministic probe, so those
   scores are identical across runs and machines. The judge is reserved for what genuinely
   needs listening.
3. **Non-compensatory by construction.** Sub-scores are combined with a *geometric* mean,
   not an arithmetic one: a near-zero in any dimension collapses the whole sample. There
   is no weight to over-optimise and no dimension to trade away.
4. **No hidden humans, no hidden weights.** The headline ranking uses fixed, equal weights
   and a fixed random seed. It requires *no* human labelling and *no* per-run tuning, so
   any evaluator recomputes the identical number from the public per-dimension scores.
5. **Declare winners conservatively.** Point estimates lie on small samples. A challenger
   is crowned only when the lower bound of a paired bootstrap on its advantage clears a
   margin scaled to human just-noticeable difference; otherwise the result is an honest
   tie.

## Install

```bash
pip install vocencebench                 # core (schema, aggregation, decision)
pip install "vocencebench[probes]"       # + acoustic & classifier probes (librosa, torch)
pip install "vocencebench[gemini]"       # + hosted audio-LLM judge backend
pip install "vocencebench[all]"          # everything, incl. CLI
```

## Quickstart

```python
import vocencebench as vb

# 1. A judge that reasons over raw audio — hosted, or any local OpenAI-compatible endpoint.
judge = vb.Judge.gemini(model="gemini-3.1-pro-preview")
# judge = vb.Judge.local(base_url="http://localhost:8003", model="Qwen/Qwen2.5-Omni-7B")

# 2. Two PromptTTS models to compare, each a callable (text, instruction) -> wav bytes.
from vocencebench.adapters import Qwen3TTSAdapter
model_a = Qwen3TTSAdapter("/path/to/checkpoint_a")
model_b = Qwen3TTSAdapter("/path/to/checkpoint_b")

# 3. One call: synthesise, probe, judge, aggregate, and decide a winner.
h2h = vb.benchmark(
    vb.load_benchmark("benchmark_v1"),           # frozen, hash-pinned corpus
    model_a, model_b, judge,
    probes=vb.with_classifiers(),                # acoustic + gender/emotion/accent/age
    transcriber=vb.whisper_transcriber(),        # enables the intelligibility gate
    labels=("model_a", "model_b"),
)
print(h2h.summary())
print(h2h.decision.winner, "—", h2h.decision.reason)
```

A single pairwise comparison, when that is all you need:

```python
verdict = judge.compare(
    text="The old lighthouse stood watch over the restless grey sea.",
    instruction="a calm elderly British man, speaking slowly and warmly",
    audio_a=open("a.wav", "rb").read(),
    audio_b=open("b.wav", "rb").read(),
    dimension="tone",                            # any trait, or "naturalness"
)
print(verdict.winner, verdict.score_a, verdict.score_b, verdict.reasoning_a)
```

## The evaluation pipeline

```
  instruction + script
          │
          ▼
   ┌─────────────┐     ┌─────────────┐
   │   model A   │     │   model B   │        two PromptTTS systems synthesise the same item
   └──────┬──────┘     └──────┬──────┘
          │  wav_a            │  wav_b
          ├───────────────────┤
          ▼                   ▼
   ┌──────────────────────────────────┐
   │  intelligibility gate (ASR WER)  │        WER > τ  →  gate = 0  →  sample composite = 0
   └──────────────┬───────────────────┘
                  ▼
   ┌──────────────────────────────────┐
   │  per-trait adherence              │
   │   • objective  → probe  (free)    │        pace·pitch·loudness·gender·emotion·accent·age
   │   • holistic   → judge  (audio)   │        tone
   │  naturalness   → judge  (audio)   │        pairwise quality preference
   └──────────────┬───────────────────┘
                  ▼  desirabilities in [0,1]
   ┌──────────────────────────────────┐
   │  geometric composite × gate       │        non-compensatory: any near-zero → collapse
   └──────────────┬───────────────────┘
                  ▼  per-sample composite Sᵢ
   ┌──────────────────────────────────┐
   │  mean over samples → S(A), S(B)   │
   │  paired bootstrap LCB(B − A)      │        5th-percentile of the paired advantage
   │  compare to dynamic margin        │        margin scaled to JND / remaining headroom
   └──────────────┬───────────────────┘
                  ▼
             winner  /  tie
```

Every judged comparison is optionally run in **both audio orders** and kept only when the
two orders agree, and the judge can be sampled `k` times and majority-voted — the standard
defences against position bias and run-to-run noise. See
[docs/methodology.md](docs/methodology.md) for the full derivation.

## What it measures

| Axis | Method | Per-clip output | Determinism |
|------|--------|-----------------|-------------|
| Adherence — objective traits | DSP / classifier **probe** on the waveform | desirability ∈ [0,1] | exact, reproducible |
| Adherence — holistic (tone) | order-swapped **audio-LLM judge**, 0–3 rubric | desirability ∈ [0,1] | judge-version-pinned |
| Naturalness / quality | pairwise **audio-LLM judge**, per-category rubric | desirability ∈ [0,1] | judge-version-pinned |
| Intelligibility | ASR (Whisper) **word-error-rate** | hard gate {0,1} | exact, reproducible |

## Trait vocabulary

Eight controllable attributes, defined once in [`vocencebench/traits.py`](vocencebench/traits.py):

| Trait | Kind | Values |
|-------|------|--------|
| `gender` | objective | male, female |
| `age` | objective (numeric, ±8 yr) | reference points 8, 13, 20, 30, 45, 60, 78 |
| `pace` | objective (ordinal) | slow, moderate, fast |
| `pitch` | objective (ordinal) | low, medium, high |
| `loudness` | objective (ordinal) | quiet, normal, loud |
| `emotion` | objective | neutral, calm, happy, sad, angry, fearful, disgust, surprised |
| `accent` | objective | American, British, Australian, Indian, Canadian |
| `tone` | holistic | warm, authoritative, playful, serious, soothing, cheerful, sarcastic, formal |

See [docs/traits.md](docs/traits.md) for the registry semantics (ordinal partial credit,
numeric tolerance, objective vs. holistic routing).

## Reproducibility

- The benchmark corpus is **generated once, frozen to JSONL, and pinned by SHA-256**; the
  frozen file — not the stochastic generator — is the artifact under version control.
- Objective probes are pure functions of the waveform.
- The bootstrap and every sampling step run under a **fixed seed**.
- The judge runs at temperature 0 and its **model version + prompt version are logged**;
  scores are comparable only within a version boundary.
- A deterministic **held-out split** keeps a private partition so models cannot be tuned
  against the public set.

Details in [docs/reproducibility.md](docs/reproducibility.md).

## Repository layout

```
vocencebench/
├── traits.py          canonical trait registry (one place to add a trait)
├── schema.py          Sample / Verdict / ProbeResult / report data structures
├── corpus_llm.py      multi-trait corpus generator → freeze + hash-pin
├── probes/            deterministic trait probes (acoustic + classifier + age)
├── judge/             audio-LLM judge: order-swap, vote@k, backends
├── prompts.py         judge rubrics and strict-JSON verdict schema
├── pair.py            evaluate one pair of clips on all traits + naturalness
├── compare.py         symmetric head-to-head over a dataset  (compare_models, benchmark)
├── decide.py          the aggregation: geometric composite → gate → LCB → margin
├── metrics.py         win-rate, control-success, confusion, bootstrap CI
├── transcribe.py      ASR transcriber for the intelligibility gate
├── adapters.py        wrap any TTS model as (text, instruction) → wav
└── cli.py             command-line interface
```

## Documentation

| Document | Contents |
|----------|----------|
| [docs/methodology.md](docs/methodology.md) | The aggregation, in full: desirability → geometric mean → gate → paired bootstrap LCB → dynamic margin. **Start here.** |
| [docs/traits.md](docs/traits.md) | Trait registry: kinds, values, ordinal/numeric scoring. |
| [docs/corpus.md](docs/corpus.md) | Corpus generation, dataset schema, freezing, held-out splits. |
| [docs/probes.md](docs/probes.md) | Deterministic probes: acoustic formulas, classifiers, age regression. |
| [docs/judge.md](docs/judge.md) | The audio-LLM judge: protocol, reliability engineering, backends, cost. |
| [docs/adapters.md](docs/adapters.md) | Wrapping TTS models for evaluation. |
| [docs/cli.md](docs/cli.md) | Command-line reference. |
| [docs/reproducibility.md](docs/reproducibility.md) | Determinism guarantees and how to preserve them. |
| [docs/validation.md](docs/validation.md) | *Optional* judge validation against human labels (diagnostic only). |

## Citation

```bibtex
@software{vocencebench,
  title  = {vocencebench: A Reproducible, Gaming-Resistant Benchmark for
            Prompt-Controllable Text-to-Speech},
  author = {Vocence},
  year   = {2026},
  note   = {https://github.com/vocence-78/vocencebench}
}
```

## License

MIT — see [LICENSE](LICENSE).
