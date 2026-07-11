# TTS model adapters

An adapter turns any speech model into the callable the evaluator expects:
`(text, instruction) -> wav bytes`. Wrap a model once and it plugs into `vb.benchmark`,
`vb.compare_models`, or `vb.evaluate`.

In a head-to-head, **both** systems are adapters — `vocencebench` compares two models
symmetrically (each trait scored absolutely for each model, naturalness as the pairwise
preference), so there is no privileged "reference"; the roles of incumbent and challenger
are decided by the [decision rule](methodology.md#8-the-decision-rule), not by which
argument you pass.

## Provided adapters

```python
from vocencebench.adapters import from_callable, FileTTS, OpenAICompatTTS, Qwen3TTSAdapter
```

### `from_callable(fn)`
Wrap any function you already have.

```python
model = from_callable(lambda text, instruction: my_model.speak(text, instruction))
```

### `OpenAICompatTTS(base_url, model, voice=..., api_key=...)`
Calls `POST /v1/audio/speech`; the voice instruction is sent in the `instructions` field
for instruction-steerable providers.

```python
model = OpenAICompatTTS(base_url="https://api.example.com", model="tts-1",
                        voice="alloy", api_key="...")
```

### `Qwen3TTSAdapter(model_dir)`
Loads a local Qwen3-TTS voice-design checkpoint and renders `instruction` as the voice
design and `text` as the content. Needs the weights on disk and a GPU.

```python
model = Qwen3TTSAdapter("/path/to/checkpoint")
```

### `FileTTS(directory)`
Serve audio generated offline. Set `current_id` before each call (the CLI does this per
sample); returns `<directory>/<current_id>.wav`. Ideal for evaluating pre-synthesised audio
without re-running a model.

## Using adapters

```python
import vocencebench as vb

h2h = vb.benchmark(
    vb.load_benchmark("benchmark_v1"),
    model_a, model_b, judge,
    probes=vb.with_classifiers(),
    labels=("system_a", "system_b"),
)
print(h2h.summary())
```

Both `model_a` and `model_b` are adapters. To evaluate audio you already generated, point
two `FileTTS` adapters (or the CLI's `--model-audios` / `--reference-audios`) at directories
of `<sample_id>.wav` files.
