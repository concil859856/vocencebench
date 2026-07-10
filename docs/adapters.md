# TTS model adapters

An adapter turns any speech model into the callable the evaluator expects:
`(text, instruction) -> wav bytes`. Wrap your model once and it plugs into
`vocencebench.evaluate`.

## Provided adapters

```python
from vocencebench.adapters import from_callable, FileTTS, OpenAICompatTTS, Qwen3TTSAdapter
```

### `from_callable(fn)`
Use any function you already have.

```python
model = from_callable(lambda text, instruction: my_model.speak(text, instruction))
```

### `OpenAICompatTTS(base_url, model, voice=..., api_key=...)`
Calls `POST /v1/audio/speech`. The voice instruction is sent in the `instructions`
field for instruction-steerable providers.

```python
model = OpenAICompatTTS(base_url="https://api.example.com", model="tts-1",
                        voice="alloy", api_key="...")
```

### `Qwen3TTSAdapter(model_dir)`
Loads a local qwen-tts voice-design checkpoint and renders `instruction` as the voice
design, `text` as the content. Needs the `qwen-tts` package and weights on disk.

```python
model = Qwen3TTSAdapter("/path/to/checkpoint")
```

### `FileTTS(directory)`
Serve audio generated offline. Set `current_id` before each call (the CLI does this per
sample); returns `<directory>/<current_id>.wav`.

## Using an adapter

```python
import vocencebench as vb
report = vb.evaluate(dataset, model, reference, judge, probes=vb.default_probes())
```

`model` and `reference` are both adapters. The reference is your fixed anchor; a
naturalness win-rate above 0.5 means the model beats it.
