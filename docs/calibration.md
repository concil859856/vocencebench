# Calibrating the judge

An audio-LLM judge is only trustworthy after you have measured it against human labels.
Do this once per judge model/version and re-run it whenever either changes.

## Build a gold set

Sample 150–400 comparisons stratified across `(trait/category, value, difficulty)`,
oversampling the hard cells (fine emotion, pace, emphasis). Have 2–3 trained raters
label each comparison with the *same* rubric the judge uses, blind to model identity
and to each other.

## Report the right numbers

1. **Human–human agreement first** — it sets the ceiling for judge–human agreement.
   Report a chance-corrected statistic (Cohen's κ / Krippendorff's α), not raw
   accuracy.
2. **Judge–human agreement per dimension.** A dimension where the judge sits at or
   below chance should be dropped from the score or moved to a deterministic probe.
3. **Probe validation.** For each objective trait, confirm the probe's own agreement
   with human labels on natural speech first — that is the probe's noise floor and it
   guards against domain shift onto synthetic audio.

## Turn the results into weights

- Keep, for each dimension, whichever of {probe, judge} correlates best with humans.
- Where a probe and the judge disagree systematically, surface it — that disagreement
  is diagnostic; do not average it away.
- Fit the relative weight of each dimension to maximise agreement with the gold set,
  rather than using hand-picked weights.

## Bounding noise

- Prefer pairwise/binary decisions over absolute scores.
- Run both audio orders and count only consistent verdicts.
- Pin and log the exact judge model version, temperature, and prompt version with every
  run so results are comparable only within a version boundary.

## Harness API

`vocencebench.calibration` provides the tooling:

```python
from vocencebench import calibration as cal, build_splits

# 1. Sample comparisons stratified by (category, difficulty, dimension).
specs = cal.stratified_comparisons(dataset, per_cell=3,
                                   dimensions=["naturalness", "tone"])
cal.export_for_labeling(specs, "to_label.jsonl")   # raters fill the "winner" field

# 2. After labelling (and recording each machine_label), load and report.
gold = cal.load_gold("labelled.jsonl")
report = cal.agreement_report(gold)     # per dimension: n, human_alpha, judge_accuracy, judge_kappa
weights = cal.suggest_weights(report)   # kappa-weighted; dimensions at/below chance get 0
```

Metrics are chance-corrected: `cohen_kappa` (judge vs majority human), `krippendorff_alpha`
(human-ceiling), `spearman` (ordinal). A held-out split (`build_splits`) keeps a private
partition so models cannot be tuned against a fixed public set.

From the CLI:

```bash
vocencebench calibrate --gold labelled.jsonl
```
