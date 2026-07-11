# Judge validation (optional, diagnostic only)

> **The headline benchmark needs no human labels and no tuning.** Ranking uses fixed,
> equal weights and a fixed seed (see [methodology.md](methodology.md)). The procedure here
> is an *optional diagnostic* for deciding whether to trust a given judge model on a given
> dimension. **It never feeds back into the scoring weights** — those stay deterministic and
> equal by design, so that any evaluator recomputes the identical ranking.

An audio-LLM judge is only worth trusting on a dimension after you have measured its
agreement with human listeners there. Run this once per judge model/version and re-run it
whenever the model or the rubric changes. Its output tells you which dimensions a judge is
reliable on — informing whether to keep judging a dimension, move it to a deterministic
[probe](probes.md), or drop it — not how to weight them.

## Build a gold set

Sample 150–400 comparisons stratified across `(dimension, value, difficulty)`, oversampling
the hard cells (fine emotion, subtle tone). Have 2–3 trained raters label each comparison
with the *same* rubric the judge uses, blind to system identity and to each other.

## Report the right numbers

1. **Human–human agreement first.** It is the ceiling for judge–human agreement. Report a
   chance-corrected statistic (Cohen's κ / Krippendorff's α), not raw accuracy.
2. **Judge–human agreement per dimension.** A dimension where the judge sits at or below
   chance should be moved to a probe or dropped from that judge's remit — not silently
   trusted.
3. **Probe validation.** For each objective trait, confirm the *probe's* own agreement with
   human labels on natural speech first; that is the probe's noise floor and it guards
   against domain shift onto synthetic audio.

## Harness API

```python
from vocencebench import calibration as cal

# 1. Sample comparisons stratified by (category, difficulty, dimension) for labelling.
specs = cal.stratified_comparisons(dataset, per_cell=3, dimensions=["naturalness", "tone"])
cal.export_for_labeling(specs, "to_label.jsonl")      # raters fill the "winner" field

# 2. After labelling, load and report per-dimension agreement.
gold   = cal.load_gold("labelled.jsonl")
report = cal.agreement_report(gold)     # per dimension: n, human_alpha, judge_accuracy, judge_kappa
```

Metrics are chance-corrected: `cohen_kappa` (judge vs. majority human), `krippendorff_alpha`
(human ceiling), `spearman` (ordinal). From the CLI:

```bash
vocencebench calibrate --gold labelled.jsonl
```

## What to do with the results

- Where a dimension's judge–human κ is healthy, keep judging it.
- Where it is at or below chance, prefer a deterministic probe or drop the dimension from
  that judge.
- Where a probe and the judge disagree *systematically*, surface it — that disagreement is
  diagnostic; do not average it away.

Again: these are decisions about **which measurement to trust for a dimension**, taken once,
by hand, when configuring a judge. They are not per-run weight tuning, and the benchmark's
aggregation weights remain fixed and equal regardless.
