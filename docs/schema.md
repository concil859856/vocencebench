# Dataset schema

A dataset is a JSONL file (one JSON object per line) of samples. Load with
`vocencebench.load_dataset(path)`.

## Sample fields

| Field | Type | Meaning |
|-------|------|---------|
| `id` | string | Unique sample id. Used to name audio files (`<id>.wav`) for CLI eval. |
| `text` | string | The text the model must speak. |
| `instruction` | string | The natural-language voice instruction (the "prompt") given to the model. |
| `traits` | object | Requested controllable attributes, e.g. `{"gender":"female","pace":"fast","tone":"warm"}`. |
| `category` | string | Naturalness grouping (`general`, `emotion`, `questions`, `paralinguistics`, `syntax`, `pronunciation`). |
| `difficulty` | int | Optional difficulty tier for reporting. |
| `language` | string | Language code, default `en`. |

## Example

```json
{"id": "vb-00001", "text": "You finished the whole report last night? On your own?",
 "instruction": "A young male voice, with an authoritative tone, sounding surprised, speaking at a slow pace.",
 "traits": {"gender": "male", "age": "young adult", "tone": "authoritative", "emotion": "surprised", "pace": "slow"},
 "category": "questions", "difficulty": 0, "language": "en"}
```

## Generating a corpus

```bash
vocencebench corpus --out data.jsonl --n 256
```

The generator is deterministic (index-driven), so the same call reproduces the same
corpus. Traits are drawn from a fixed bank and spread evenly across samples and
naturalness categories; a matching voice `instruction` is rendered from the traits.

## Traits

The canonical trait vocabulary lives in `vocencebench.traits`. Each trait is either
**objective** (measured by a probe: gender, age, pace, pitch, loudness, emotion,
accent), **holistic** (judged by the audio-LLM: tone), or **hybrid** (environment).
Add a trait by registering it in `vocencebench/traits.py`.
