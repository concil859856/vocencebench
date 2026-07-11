# Methodology: from per-dimension scores to a single verdict

This document derives the aggregation at the heart of `vocencebench`: how many
heterogeneous per-dimension scores — some absolute, some pairwise — are fused into one
reproducible, gaming-resistant decision between two PromptTTS systems. The implementation
is [`vocencebench/decide.py`](../vocencebench/decide.py); this is its rationale.

## 1. The problem

Evaluating a single item produces a bag of dissimilar numbers:

- **Objective adherence** (pace, pitch, loudness, gender, emotion, accent, age) — each an
  absolute *"does this clip match the requested value"* desirability in `[0, 1]`, produced
  by a deterministic probe. Both systems can score high simultaneously.
- **Holistic adherence** (tone) — an absolute `0–3` rubric score from the audio-LLM judge,
  normalised to `[0, 1]`.
- **Naturalness / quality** — inherently a *pairwise preference* ("which sounds better"),
  which the judge also expresses as absolute `0–3` scores per clip.
- **Intelligibility** — a word-error-rate, used as a hard pass/fail gate.

A naïve headline number — say, a weighted arithmetic mean — fails in three ways that
matter for an *incentivised, reproducible* benchmark:

1. **Domination.** A zero-sum metric (a win = 1, a loss = 0) has a full unit of range,
   while a graded absolute metric that clusters near 0.7–0.9 has far less. In a weighted
   sum the binary metric swamps the graded ones. Naturalness, expressed as win/lose, would
   dictate the verdict on its own.
2. **Compensation.** An arithmetic mean lets a high score in one dimension *buy back* a
   failure in another. A system that ignores the requested accent entirely but is very
   loud can match one that honours every trait. That is precisely the trade-off an
   adversary optimises.
3. **Over-confidence.** Point estimates on a few hundred items are noisy. Declaring a
   winner on `S(B) > S(A)` alone crowns a champion on differences well inside the noise.

The scheme below addresses each in turn.

## 2. Step 1 — desirability normalisation

Every sub-score is first expressed on a common **desirability** scale `d ∈ [0, 1]`, where
`0` is "requirement not met at all" and `1` is "fully met" (after Derringer & Suich's
desirability functions). This makes heterogeneous metrics comparable *before* they are
combined, and removes the domination failure (§1.1) at the source — there is no longer a
unit-range binary metric fighting a compressed graded one; all dimensions share the same
`[0, 1]` range.

| Dimension | Raw form | → desirability |
|-----------|----------|----------------|
| ordinal probe (pace, pitch, loudness) | measured bucket | `1.0` exact · `0.5` adjacent · `0.0` else |
| categorical probe (gender, emotion, accent) | measured label | `1.0` match · `0.0` mismatch |
| numeric probe (age) | measured value `m` vs requested `r`, tolerance `τ` | `max(0, 1 − max(0, |m−r| − τ) / τ)` |
| holistic judge (tone) | `0–3` rubric score | `score / 3` |
| naturalness | `0–3` rubric score per clip | `score / 3` |

Probes and rubrics are documented in [probes.md](probes.md) and [judge.md](judge.md). The
key point here is only that everything downstream sees numbers in `[0, 1]` with the same
meaning.

## 3. Step 2 — the per-sample composite (non-compensatory)

For one sample and one system, let the desirabilities be `d₁ … d_k` (the `k` dimensions
scored for that item). The composite is a **weighted geometric mean**, multiplied by the
intelligibility gate `g ∈ {0, 1}`:

```
                   ⎛  Σ wⱼ · ln(max(dⱼ, ε))  ⎞
   S  =  g · exp   ⎜  ─────────────────────  ⎟
                   ⎝         Σ wⱼ            ⎠
```

with default equal weights `wⱼ = 1`, and a small floor `ε = 0.01` inside the logarithm.

**Why geometric, not arithmetic.** The geometric mean is *non-compensatory*: because it is
a product (in log-space, a sum of logs), a single small factor drags the whole result
toward zero, and no amount of excellence elsewhere buys it back. This is the property that
kills the compensation attack (§1.2). Concretely, for a system that scores `1.0` on eight
dimensions and `0.05` on the ninth:

```
   arithmetic mean = (8·1.0 + 0.05) / 9 = 0.894      ← failure nearly invisible
   geometric  mean = (1.0⁸ · 0.05)^(1/9) = 0.717      ← failure clearly penalised
```

The gap widens as the failure deepens. An adversary can no longer ignore a hard trait and
compensate with an easy one; every requested attribute must actually be delivered.

**The ε floor.** A literal zero anywhere would send `ln(d)` to `−∞` and force the composite
to exactly `0`, making one probe misfire catastrophic. Flooring `d` at `ε = 0.01` keeps a
single dropout severe but survivable, while preserving the non-compensatory character. It
is a soft veto, not an infinite one.

**Weights.** The default is uniform. Equal weighting is deliberate: it needs no human
calibration, cannot be quietly re-tuned between runs, and is the choice most robust to
overfitting when no ground-truth "importance" ranking exists. Weights *can* be supplied
(`decide(..., weights={...})`) for analysis, but the headline benchmark does not use them.

## 4. The intelligibility gate

Adherence and naturalness are only meaningful if the words are actually intelligible; a
melodious clip that says the wrong thing is worthless. Each clip is transcribed by ASR and
scored by word-error-rate against the reference text:

```
   g = 1  if  WER ≤ τ        (default τ = 0.15)
   g = 0  if  WER >  τ       → sample composite S = 0
```

The gate is a **hard, non-compensatory veto**: no trait or naturalness score can rescue an
unintelligible clip. It sits *outside* the geometric mean (as a `{0,1}` multiplier) rather
than inside it, so failing the gate zeroes the sample outright instead of being averaged
against other dimensions.

## 5. Step 3 — aggregate over samples

Each system's benchmark score is the arithmetic mean of its per-sample composites:

```
   S(A) = mean_i  S_A(i)          S(B) = mean_i  S_B(i)
```

Arithmetic is correct *here* — across independent items we want an unbiased estimate of
expected per-item quality, and one bad item should not zero the whole benchmark. The
non-compensation lives *within* a sample (§3), where trading traits is the exploit;
*across* samples, averaging is the right estimator.

## 6. Step 4 — the paired bootstrap lower confidence bound

`S(A)` and `S(B)` are point estimates. To decide a winner we need to know whether the gap
is *real* given the sample size — and we should exploit the fact that both systems are
evaluated on the *same items*.

For every sample `i`, form the **paired difference** of composites:

```
   δ_i = S_challenger(i) − S_incumbent(i)
```

Pairing removes item-difficulty variance: a hard item drags both systems down together, so
`δ_i` isolates the head-to-head advantage far more tightly than comparing two unpaired
means. We then bootstrap the mean of `δ`:

1. Resample the `δ_i` with replacement `N = 2000` times (fixed seed `3151662`).
2. Take the mean of each resample → a distribution of the mean advantage.
3. The **lower confidence bound** is its 5th percentile (`α = 0.05`, a one-sided 95% LCB):

```
   LCB(challenger − incumbent) = percentile_5( { mean(resample_b) : b = 1..N } )
```

Using the **lower** bound, not the point estimate, is what makes the verdict conservative
and reproducible. It answers *"even in a pessimistic reading of the evidence, does the
challenger still lead?"* — and because the seed is fixed, every evaluator computes the
identical bound. (`decide` treats a single-sample input as its own bound and a zero-length
input as `0`, so the function is total.)

## 7. Step 5 — the dynamic decision margin

A statistically significant difference can still be *perceptually irrelevant*. We require
the challenger's advantage to clear a margin that scales with how much room to improve
remains:

```
   margin(S_inc) = max( floor,  c · (1 − S_inc) )        c = 0.10,  floor = 0.015
```

- **Headroom scaling `c · (1 − S_inc)`.** Near the bottom of the scale, a fixed gap is easy
  and cheap; near saturation, the same nominal gap represents a much larger real
  improvement and should be worth more. Tying the margin to remaining headroom `(1 − S_inc)`
  makes the bar rise as systems get good, so late-stage wins must be substantive.
- **The floor.** Grounded in listening-test just-noticeable difference. On comparison-MOS
  (CMOS) scales, one unit ≈ `0.167` on `[0,1]`; the JND is on the order of `0.1` CMOS ≈
  `0.017`. The floor of `0.015` (≈ 0.09 CMOS) sits just below that JND, so a "win" never
  rests on a difference below the threshold of human audibility.

## 8. The decision rule

```
   winner = challenger   if   LCB(challenger − incumbent) > margin(S_incumbent)
   winner = tie          otherwise
```

**Roles.** If an `incumbent` is named (e.g. a reigning king in a king-of-the-hill setting),
it is the baseline to beat. Otherwise the lower-scoring system is treated as the incumbent
and the higher-scoring one as the challenger — so the burden of proof always falls on the
system claiming superiority.

**Ties are first-class.** Requiring `LCB > margin` — rather than `point estimate > 0` —
means roughly an order of magnitude more evidence is needed to crown a winner than to
observe a raw lead. Close calls resolve to an honest tie instead of a coin-flip champion.
This is the correct behaviour for an incentivised benchmark: it refuses to reward noise.

## 9. Worked example

Two systems over a small run; per-sample composites already computed (gate applied):

```
   sample   S_A     S_B     δ = S_B − S_A
   ──────   ────    ────    ─────────────
     1      0.71    0.78       +0.07
     2      0.66    0.69       +0.03
     3      0.80    0.74       −0.06
     4      0.62    0.71       +0.09
     5      0.75    0.77       +0.02
   ──────   ────    ────
   mean     0.708   0.738     point gap +0.030
```

`B` leads on the point estimate. The incumbent is `A` (lower mean), so
`margin = max(0.015, 0.10·(1 − 0.708)) = 0.0292`. Bootstrapping the five `δ` values yields
a 5th-percentile `LCB ≈ −0.01`. Since `LCB (−0.01) < margin (0.029)`, the verdict is
**tie** — the point lead is inside the noise for a sample this small. The same computation
on hundreds of items, with the advantage holding up, would push the LCB above the margin
and crown `B`.

## 10. Why this resists gaming

| Attack | Defence |
|--------|---------|
| Over-optimise one loud dimension | geometric mean gives it no extra leverage; all dimensions share `[0,1]` |
| Ignore a hard trait, compensate with an easy one | non-compensatory product penalises any near-zero (§3) |
| Produce pretty audio that says the wrong words | intelligibility gate zeroes the sample (§4) |
| Win by a hair on a lucky sample | paired bootstrap LCB + margin demand a real, audible, repeatable lead (§§6–8) |
| Re-tune the aggregation to taste | fixed equal weights, fixed seed, fixed margin — no free parameters at ranking time |

## 11. Reproducibility of the decision

Given the same per-dimension scores, `decide` is a pure function: the only stochastic step
is the bootstrap, and it runs under a fixed seed, so the LCB — and therefore the verdict —
is identical on every machine. Any third party can recompute the ranking from the published
per-sample composites and must arrive at the same winner. See
[reproducibility.md](reproducibility.md).

## References (methods)

The scheme composes standard, well-established techniques — desirability functions
(Derringer & Suich), geometric-mean aggregation in multi-criteria decision analysis,
non-parametric bootstrap confidence intervals (Efron), paired difference testing, and
comparison-MOS / just-noticeable-difference norms from speech-quality assessment — into a
single deterministic decision rule tailored to PromptTTS ranking.
