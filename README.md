# vocencebench

**Prompt-driven text-to-speech evaluation.** `vocencebench` scores how well a
prompt-controllable ("PromptTTS") speech model does two things:

1. **Trait adherence** — does the generated voice match the *requested* attributes
   (gender, age, tone, emotion, pace, accent, environment, …)?
2. **Naturalness** — is the delivery realistic and expressive on challenging text
   (questions, emotional narration, paralinguistics, complex pronunciation, …)?

Both are scored as **pairwise comparisons against a reference clip**, with an
audio-LLM acting as the judge (deep chain-of-thought, strict JSON verdicts). Objective
traits are additionally checked with deterministic signal-processing probes so the
score is inspectable and reproducible, not judge-only.

## Install

```bash
pip install vocencebench                 # core
pip install "vocencebench[all]"          # + probes, Gemini backend, CLI
```

## Quickstart

```python
import vocencebench as vb

# 1. Pick a judge — a hosted model, or any local OpenAI-compatible endpoint.
judge = vb.Judge.gemini(model="gemini-3.1-pro")                # hosted
# judge = vb.Judge.local(base_url="http://localhost:8003",
#                        model="Qwen/Qwen2.5-Omni-7B")         # self-hosted

# 2. Compare two clips of the same text/instruction on one dimension.
verdict = judge.compare(
    text="The old lighthouse stood watch over the restless grey sea.",
    instruction="a calm elderly British man, speaking slowly and warmly",
    audio_a=open("model.wav", "rb").read(),
    audio_b=open("reference.wav", "rb").read(),
    dimension="tone",          # any trait, or "naturalness"
)
print(verdict.winner, verdict.scores, verdict.reasoning)

# 3. Or run a full evaluation of a model against a reference over a dataset.
from vocencebench.adapters import Qwen3TTSAdapter          # or OpenAICompatTTS / from_callable
report = vb.evaluate(
    dataset=vb.load_dataset("mydata.jsonl"),
    model=Qwen3TTSAdapter("/path/to/checkpoint"),          # (text, instruction) -> wav
    reference=reference_tts,                                # the fixed anchor
    judge=judge,
    probes=vb.with_classifiers(),                           # acoustic + gender/emotion/accent
)
print(report.summary())        # win-rates, per-trait control-success, WER
```

Generate a trait-controlled dataset (with a private held-out split) and calibrate the
judge against human labels:

```python
public, holdout = vb.build_splits(256)                     # anti-overfit partition
from vocencebench import calibration as cal
report  = cal.agreement_report(cal.load_gold("labelled.jsonl"))
weights = cal.suggest_weights(report)                      # drop dimensions at chance
```

## What it measures

| Axis | How | Output |
|------|-----|--------|
| Adherence (per trait) | DSP/classifier probe where objective; pairwise audio-LLM judge where holistic | control-success rate + win-rate |
| Naturalness | pairwise audio-LLM judge, per-category rubric, order-swapped | win-rate vs reference |
| Intelligibility | ASR (Whisper) word-error-rate | WER + gate |

Every pairwise comparison is run in **both audio orders** and only counts when
consistent, and the holistic judge can be sampled `k` times and majority-voted to
bound run-to-run noise.

## Design notes

- **Judge-agnostic.** Any judge that reasons over raw audio works — a hosted model
  for reference/calibration, a local model for reproducible, offline scoring.
- **Probes carry the objective traits.** Pitch/rate/loudness/gender/age/accent are
  measured, not guessed, so those scores are deterministic across runs and machines.
- **Two numbers, never one.** Adherence and naturalness are reported separately; a
  model can nail the voice but read flatly, or read beautifully off-prompt.

See [`docs/`](docs/) for the dataset schema, judge rubrics, and calibration protocol.

## License

MIT — see [LICENSE](LICENSE).
