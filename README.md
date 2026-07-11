<div align="center">

# 🎙️ vocencebench

### A reproducible, gaming-resistant benchmark for prompt-controllable text-to-speech

<p>
  <img alt="License" src="https://img.shields.io/badge/license-MIT-2563eb?style=flat-square">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10+-2563eb?style=flat-square&logo=python&logoColor=white">
  <img alt="Judge" src="https://img.shields.io/badge/judge-audio--LLM-7c3aed?style=flat-square">
  <img alt="Scoring" src="https://img.shields.io/badge/scoring-deterministic-059669?style=flat-square">
  <img alt="Status" src="https://img.shields.io/badge/status-research_preview-f59e0b?style=flat-square">
</p>

<p>
  <a href="#-install"><b>Install</b></a> ·
  <a href="#-quickstart"><b>Quickstart</b></a> ·
  <a href="#-the-pipeline"><b>Pipeline</b></a> ·
  <a href="docs/methodology.md"><b>Methodology</b></a> ·
  <a href="#-documentation"><b>Docs</b></a>
</p>

</div>

---

Prompt-controllable text-to-speech (**PromptTTS**) synthesises speech from two inputs — a
script to read and a natural-language *voice instruction* ("a calm elderly British man,
speaking slowly and warmly"). Evaluating it means answering two orthogonal questions at
once — **does the voice match what was requested**, and **is the delivery natural and
intelligible** — then collapsing many heterogeneous sub-scores into one defensible verdict.

`vocencebench` does this with deterministic signal-processing **probes** where a trait is
measurable, an order-swapped **audio-LLM judge** where it is holistic, an ASR
**intelligibility gate**, and a **non-compensatory aggregation** guarded by a **paired
bootstrap confidence bound**.

> [!NOTE]
> The result is a ranking that is **reproducible** (same inputs → same verdict, bit-for-bit),
> **hard to game** (no single dimension can dominate or be traded away), and **statistically
> honest** (a winner is declared only when the improvement is both real and audible).

## ✨ Design principles

|   | Principle | What it means |
|---|-----------|---------------|
| **①** | **Two numbers before one** | Adherence and naturalness are measured separately, fused only at the final decision. A model that nails the voice but reads flatly cannot hide, and neither can the reverse. |
| **②** | **Measure, don't guess** | Every objective trait (pace, pitch, loudness, gender, emotion, accent, age) is read from the signal by a deterministic probe — identical across runs and machines. The judge is reserved for what needs an ear. |
| **③** | **Non-compensatory by construction** | Sub-scores combine with a *geometric* mean: a near-zero in any dimension collapses the whole sample. No weight to over-optimise, no dimension to trade away. |
| **④** | **No hidden humans, no hidden weights** | The headline ranking uses fixed, equal weights and a fixed seed — no human labelling, no per-run tuning. Any evaluator recomputes the identical number. |
| **⑤** | **Declare winners conservatively** | A challenger is crowned only when the lower bound of a paired bootstrap on its advantage clears a margin scaled to human just-noticeable difference. Otherwise: an honest tie. |

## 📦 Install

```bash
pip install vocencebench                 # core — schema, aggregation, decision
pip install "vocencebench[probes]"       # + acoustic & classifier probes  (librosa, torch)
pip install "vocencebench[gemini]"       # + hosted audio-LLM judge backend
pip install "vocencebench[all]"          # everything, incl. CLI
```

## 🚀 Quickstart

```python
import vocencebench as vb

# 1 · A judge that reasons over raw audio — hosted, or any local OpenAI-compatible endpoint.
judge = vb.Judge.gemini(model="gemini-3.1-pro-preview")
# judge = vb.Judge.local(base_url="http://localhost:8003", model="Qwen/Qwen2.5-Omni-7B")

# 2 · Two PromptTTS models to compare, each a callable (text, instruction) -> wav bytes.
from vocencebench.adapters import Qwen3TTSAdapter
model_a = Qwen3TTSAdapter("/path/to/checkpoint_a")
model_b = Qwen3TTSAdapter("/path/to/checkpoint_b")

# 3 · One call: synthesise, probe, judge, aggregate, and decide a winner.
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

<details>
<summary><b>A single pairwise comparison</b>, when that is all you need</summary>

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

</details>

## 🔬 The pipeline

```mermaid
flowchart TD
    I(["instruction + script"]) --> MA["model A"] & MB["model B"]
    MA -->|wav_a| G{{"intelligibility gate · ASR WER"}}
    MB -->|wav_b| G
    G -->|"WER exceeds τ"| Z(["sample composite = 0"])
    G -->|pass| SC["per-trait adherence<br/>objective → probe · holistic → judge<br/>+ naturalness → judge"]
    SC --> GM["geometric composite × gate<br/>non-compensatory"]
    GM --> MN["mean over samples → S(A), S(B)"]
    MN --> LCB["paired bootstrap LCB(B − A)"]
    LCB --> DEC{"LCB exceeds dynamic margin?"}
    DEC -->|yes| W(["🏆 winner"])
    DEC -->|no| T(["🤝 tie"])

    classDef model fill:#2563eb,stroke:#1e40af,color:#fff;
    classDef gate fill:#f59e0b,stroke:#b45309,color:#fff;
    classDef agg fill:#7c3aed,stroke:#5b21b6,color:#fff;
    classDef out fill:#059669,stroke:#047857,color:#fff;
    class MA,MB model;
    class G gate;
    class GM,MN,LCB agg;
    class W,T out;
```

Every judged comparison can run in **both audio orders** (kept only when they agree), and
the judge can be sampled `k` times and majority-voted — the standard defences against
position bias and run-to-run noise. Full derivation in
**[docs/methodology.md](docs/methodology.md)**.

## 📊 What it measures

| Axis | Method | Per-clip output | Determinism |
|------|--------|:---------------:|:-----------:|
| **Adherence** — objective traits | DSP / classifier **probe** on the waveform | desirability&nbsp;∈&nbsp;[0,1] | 🟢 exact |
| **Adherence** — holistic (`tone`) | order-swapped **audio-LLM judge**, 0–3 rubric | desirability&nbsp;∈&nbsp;[0,1] | 🟡 version-pinned |
| **Naturalness** / quality | pairwise **audio-LLM judge**, per-category rubric | desirability&nbsp;∈&nbsp;[0,1] | 🟡 version-pinned |
| **Intelligibility** | ASR (Whisper) **word-error-rate** | hard gate&nbsp;{0,1} | 🟢 exact |

## 🎛️ Trait vocabulary

Eight controllable attributes, defined once in **[`traits.py`](vocencebench/traits.py)**:

| Trait | Kind | Values |
|-------|------|--------|
| `gender` | objective | male · female |
| `age` | objective · numeric&nbsp;(±8&nbsp;yr) | 8 · 13 · 20 · 30 · 45 · 60 · 78 |
| `pace` | objective · ordinal | slow · moderate · fast |
| `pitch` | objective · ordinal | low · medium · high |
| `loudness` | objective · ordinal | quiet · normal · loud |
| `emotion` | objective | neutral · calm · happy · sad · angry · fearful · disgust · surprised |
| `accent` | objective | American · British · Australian · Indian · Canadian |
| `tone` | holistic | warm · authoritative · playful · serious · soothing · cheerful · sarcastic · formal |

See **[docs/traits.md](docs/traits.md)** for ordinal partial-credit, numeric tolerance, and objective-vs-holistic routing.

## 🔁 Reproducibility

> [!IMPORTANT]
> Given the same inputs, **every evaluator computes the identical ranking**. That is a hard
> requirement for a benchmark whose scores carry weight — not a nice-to-have.

- 🧊 The corpus is generated once, **frozen to JSONL, and pinned by SHA-256** — the frozen file, not the stochastic generator, is the artifact.
- 🧮 Objective probes are **pure functions of the waveform**; the bootstrap runs under a **fixed seed**.
- 📌 The judge runs at **temperature 0** with **model + prompt version logged**; scores compare only within a version boundary.
- 🔒 A deterministic **held-out split** keeps a private partition so models cannot be tuned against the public set.

Details in **[docs/reproducibility.md](docs/reproducibility.md)**.

## 🗂️ Repository layout

<details>
<summary>Module map</summary>

```
vocencebench/
├── traits.py          canonical trait registry (one place to add a trait)
├── schema.py          Sample / Verdict / ProbeResult / report data structures
├── corpus_llm.py      multi-trait corpus generator → freeze + hash-pin
├── probes/            deterministic trait probes (acoustic + classifier + age)
├── judge/             audio-LLM judge: order-swap, vote@k, backends
├── prompts.py         judge rubrics and strict-JSON verdict schema
├── pair.py            evaluate one pair of clips on all traits + naturalness
├── compare.py         symmetric head-to-head over a dataset (compare_models, benchmark)
├── decide.py          the aggregation: geometric composite → gate → LCB → margin
├── metrics.py         win-rate, control-success, confusion, bootstrap CI
├── transcribe.py      ASR transcriber for the intelligibility gate
├── adapters.py        wrap any TTS model as (text, instruction) → wav
└── cli.py             command-line interface
```

</details>

## 📚 Documentation

| Document | Contents |
|----------|----------|
| **[methodology.md](docs/methodology.md)** | The aggregation in full — desirability → geometric mean → gate → paired bootstrap LCB → dynamic margin. **Start here.** |
| [traits.md](docs/traits.md) | Trait registry: kinds, values, ordinal / numeric scoring. |
| [corpus.md](docs/corpus.md) | Corpus generation, dataset schema, freezing, held-out splits. |
| [probes.md](docs/probes.md) | Deterministic probes: acoustic formulas, classifiers, age regression. |
| [judge.md](docs/judge.md) | The audio-LLM judge: protocol, reliability engineering, backends, cost. |
| [adapters.md](docs/adapters.md) | Wrapping TTS models for evaluation. |
| [cli.md](docs/cli.md) | Command-line reference. |
| [reproducibility.md](docs/reproducibility.md) | Determinism guarantees and how to preserve them. |
| [validation.md](docs/validation.md) | *Optional* judge validation against human labels (diagnostic only). |

## 📖 Citation

<details>
<summary>BibTeX</summary>

```bibtex
@software{vocencebench,
  title  = {vocencebench: A Reproducible, Gaming-Resistant Benchmark for
            Prompt-Controllable Text-to-Speech},
  author = {Vocence},
  year   = {2026},
  note   = {https://github.com/vocence-78/vocencebench}
}
```

</details>

## ⚖️ License

Released under the **MIT License** — see [LICENSE](LICENSE).
