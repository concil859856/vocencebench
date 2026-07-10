"""Command-line interface: build corpora, compare two clips, run an evaluation.

Evaluation from the CLI works on pre-generated audio: point ``--model-audios`` and
``--reference-audios`` at directories of ``<sample_id>.wav`` files.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

try:
    import click
except ImportError:  # pragma: no cover
    raise SystemExit("The CLI needs 'click': pip install 'vocencebench[cli]'")

from vocencebench import evaluate, load_dataset, save_dataset
from vocencebench.corpus import build_corpus
from vocencebench.judge import Judge
from vocencebench.probes import default_probes


def _judge(kind: str, model: str, base_url: str, votes: int, swap: bool) -> Judge:
    if kind == "gemini":
        return Judge.gemini(model=model or "gemini-3.1-pro-preview", votes=votes, swap=swap)
    if kind == "local":
        if not base_url:
            raise click.ClickException("--base-url is required for --judge local")
        return Judge.local(base_url=base_url, model=model or "judge",
                           api_key=os.environ.get("JUDGE_API_KEY", ""), votes=votes, swap=swap)
    raise click.ClickException(f"unknown judge kind: {kind}")


@click.group()
@click.version_option(package_name="vocencebench")
def main() -> None:
    """vocencebench — prompt-driven text-to-speech evaluation."""


@main.command()
@click.option("--out", required=True, type=click.Path(dir_okay=False), help="Output JSONL path.")
@click.option("--n", default=128, type=int, help="Number of samples.")
@click.option("--categories", default=None, help="Comma-separated category subset.")
def corpus(out: str, n: int, categories: str) -> None:
    """Generate a deterministic trait-controlled corpus."""
    cats = tuple(c.strip() for c in categories.split(",")) if categories else None
    samples = build_corpus(n) if cats is None else build_corpus(n, categories=cats)
    save_dataset(samples, out)
    click.echo(f"wrote {len(samples)} samples to {out}")


@main.command()
@click.option("--a", "audio_a", required=True, type=click.Path(exists=True))
@click.option("--b", "audio_b", required=True, type=click.Path(exists=True))
@click.option("--text", required=True)
@click.option("--dimension", default="naturalness", help="Trait name, 'naturalness', or a category.")
@click.option("--instruction", default="")
@click.option("--judge", "judge_kind", default="local", type=click.Choice(["local", "gemini"]))
@click.option("--model", default="")
@click.option("--base-url", default="")
@click.option("--votes", default=1, type=int)
@click.option("--no-swap", is_flag=True)
def compare(audio_a, audio_b, text, dimension, instruction, judge_kind, model, base_url, votes, no_swap):
    """Judge two clips of the same text on one dimension."""
    j = _judge(judge_kind, model, base_url, votes, not no_swap)
    v = j.compare(text, instruction, Path(audio_a).read_bytes(), Path(audio_b).read_bytes(), dimension)
    click.echo(json.dumps({"dimension": v.dimension, "winner": v.winner, "consistent": v.consistent,
                           "score_a": v.score_a, "score_b": v.score_b,
                           "confidence": v.confidence, "reasoning": v.reasoning}, indent=2))


@main.command("eval")
@click.option("--dataset", required=True, type=click.Path(exists=True))
@click.option("--model-audios", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--reference-audios", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--judge", "judge_kind", default="local", type=click.Choice(["local", "gemini"]))
@click.option("--model", default="")
@click.option("--base-url", default="")
@click.option("--votes", default=1, type=int)
@click.option("--no-swap", is_flag=True)
@click.option("--no-probes", is_flag=True)
@click.option("--out", default=None, type=click.Path(dir_okay=False), help="Write full report JSON.")
def eval_cmd(dataset, model_audios, reference_audios, judge_kind, model, base_url, votes, no_swap, no_probes, out):
    """Evaluate pre-generated model audio against reference audio."""
    samples = load_dataset(dataset)
    md, rd = Path(model_audios), Path(reference_audios)
    j = _judge(judge_kind, model, base_url, votes, not no_swap)
    probes = None if no_probes else default_probes()

    # Per-sample audio lookup: the current sample id selects <id>.wav from each dir.
    current = {"id": None}
    model_tts = lambda text, inst: (md / f"{current['id']}.wav").read_bytes()
    ref_tts = lambda text, inst: (rd / f"{current['id']}.wav").read_bytes()

    from vocencebench.runner import _aggregate
    from vocencebench.schema import EvalReport

    results = []
    for s in samples:
        current["id"] = s.id
        results.extend(evaluate([s], model_tts, ref_tts, j, probes=probes).results)
    report = EvalReport(n=len(results), results=results, metrics=_aggregate(results))
    click.echo(report.summary())
    if out:
        Path(out).write_text(json.dumps({"n": report.n, "metrics": report.metrics}, indent=2))
        click.echo(f"report written to {out}")


@main.command()
@click.option("--gold", required=True, type=click.Path(exists=True), help="Labelled gold JSONL.")
@click.option("--out", default=None, type=click.Path(dir_okay=False))
def calibrate(gold: str, out: str) -> None:
    """Report judge–human agreement per dimension and suggest weights."""
    from vocencebench import calibration as cal
    report = cal.agreement_report(cal.load_gold(gold))
    weights = cal.suggest_weights(report)
    for dim, r in sorted(report.items()):
        ka = r.get("judge_kappa")
        click.echo(f"{dim:16s} n={r['n']:4d} acc={r.get('judge_accuracy')} "
                   f"kappa={ka} human_alpha={r.get('human_alpha')} weight={weights.get(dim, 0.0)}")
    if out:
        Path(out).write_text(json.dumps({"report": report, "weights": weights}, indent=2))
        click.echo(f"written to {out}")


if __name__ == "__main__":
    main()
