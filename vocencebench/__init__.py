"""vocencebench — prompt-driven text-to-speech evaluation.

Score how well a PromptTTS model (a) matches the *requested* voice traits and
(b) reads text naturally, using deterministic probes for objective traits and an
audio-LLM judge (pairwise, order-swapped) for the rest.

Typical use::

    import vocencebench as vb

    judge = vb.Judge.local(base_url="http://localhost:8003", model="Qwen/Qwen2.5-Omni-7B")
    report = vb.evaluate(vb.load_dataset("data.jsonl"), model_fn, reference_fn,
                         judge, probes=vb.default_probes())
    print(report.summary())
"""

from vocencebench import adapters, calibration, corpus, traits
from vocencebench.corpus import build_corpus, build_splits
from vocencebench.judge import Judge
from vocencebench.probes import default_probes, with_classifiers
from vocencebench.runner import evaluate
from vocencebench.schema import (
    EvalReport, ProbeResult, Sample, SampleResult, Verdict,
    load_dataset, save_dataset,
)

__version__ = "0.1.0"

__all__ = [
    "Judge",
    "evaluate",
    "default_probes",
    "with_classifiers",
    "build_corpus",
    "build_splits",
    "adapters",
    "calibration",
    "corpus",
    "traits",
    "Sample",
    "Verdict",
    "ProbeResult",
    "SampleResult",
    "EvalReport",
    "load_dataset",
    "save_dataset",
    "__version__",
]
