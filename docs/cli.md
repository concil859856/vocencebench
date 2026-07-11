# Command-line interface

Install the CLI extra and invoke `vocencebench`:

```bash
pip install "vocencebench[cli]"
vocencebench --help
```

The CLI covers the offline path — build a corpus, judge two clips, evaluate directories of
pre-generated audio. For in-process, end-to-end comparison (synthesise → probe → judge →
decide) use the Python API (`vb.benchmark`); see the [README](../README.md#quickstart).

## `corpus` — generate a dataset

```bash
vocencebench corpus --out data.jsonl --n 128 [--categories general,emotion]
```

Writes `n` samples to a JSONL dataset. Uses the deterministic template generator
([`corpus.py`](../vocencebench/corpus.py)) so the same call reproduces the same corpus. For
the richer language-model generator used by the shipped benchmark, see
[corpus.md](corpus.md).

## `compare` — judge two clips on one dimension

```bash
vocencebench compare \
  --a model.wav --b reference.wav \
  --text "The old lighthouse stood watch over the restless grey sea." \
  --dimension tone \
  --instruction "a calm elderly British man, speaking slowly and warmly" \
  --judge gemini --model gemini-3.1-pro-preview
```

| Option | Meaning |
|--------|---------|
| `--a`, `--b` | the two clips of the same text |
| `--dimension` | a trait name, `naturalness`, or a category |
| `--judge` | `local` (needs `--base-url`) or `gemini` |
| `--votes` | vote@k sampling (default 1) |
| `--no-swap` | disable order-swap consistency checking |

Prints the JSON verdict (`winner`, `score_a`, `score_b`, `consistent`, `reasoning`).

## `eval` — evaluate pre-generated audio

```bash
vocencebench eval \
  --dataset data.jsonl \
  --model-audios ./model_wavs \
  --reference-audios ./reference_wavs \
  --judge gemini --model gemini-3.1-pro-preview \
  --out report.json
```

Both audio directories contain `<sample_id>.wav` files. Objective traits are scored by
probes (disable with `--no-probes`); the judge scores the holistic dimensions and
naturalness. Prints a report summary (win-rates, per-trait control-success, WER) and, with
`--out`, writes the full metrics JSON.

## `calibrate` — optional judge validation

```bash
vocencebench calibrate --gold labelled.jsonl [--out report.json]
```

Reports per-dimension judge–human agreement from a labelled gold set. This is a
**diagnostic only** — it does not change the benchmark's fixed, equal-weight scoring. See
[validation.md](validation.md).
