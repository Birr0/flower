↑ [Back to README](README.md)

# Planning

## 6. Limitations

- `ΔBase` requires a full reverse-ODE integration *per pathway* (conditional and unconditional
  separately) for every sample — 2× the integration cost of a single likelihood evaluation, and
  this cost is per-sample, not amortized.
- The base distributions are not shared/coupled in the current CFG implementation (§2.2), so the
  simpler pure-flux form of PMI is not available without a separate architecture change (noted as
  future work, out of scope here).
- Divergence computation cost scales with latent dimensionality; where the exact trace is
  infeasible (spectra-scale latents), the Hutchinson estimator's own variance is convolved with
  the MI estimate's variance and must be reported separately, not conflated.
- PMI as defined here is with respect to whatever `y` is fed to the conditional pathway. If `y`
  bundles multiple factors, the resulting MI is joint MI with the whole bundle — per-factor
  attribution requires running the estimator with each factor conditioned individually.
- No finite-sample theoretical guarantee (consistency proof, non-asymptotic error bound) currently
  exists for this estimator, unlike N-MIENF (Related Work), which has both. Solver tolerance and
  Hutchinson-estimator variance are empirically controllable but not backed by a proof; this is a
  real gap relative to the closest competing method, not just a minor omission.

## Causal Interpretation (and Its Limits)

**Read the caveat before the applications.** Everything in §2 is an *observational* quantity.
`p(x|y)` from a conditional flow trained on observational data is the *seeing*-conditional, not
the *doing*-conditional `p(x | do(y))`. By Pearl's ladder of causation, you cannot climb from the
joint distribution to interventional or counterfactual claims *using the joint alone* — distinct
causal mechanisms can produce the identical joint. So the slogan "just write down the probability
of everything" is necessary but underdetermined: what buys causal content is the *causal
factorization* (how each variable is generated from its direct causes), not the joint itself. Any
statement of the form "high PMI(latent, factor) ⟹ the factor causally drives the latent" is a
rung-1-masquerading-as-rung-2 error and must not appear in this work unqualified.

That said, flower's setup has three legitimate — and honestly bounded — points of contact with
causal inference, listed strongest-first:

1. **The conditional flow's direction is already a causal assumption.** Conditioning on `y` and
   generating `x` (`y → x`) commits the architecture to `y` being upstream of `x`; a conditional
   flow is thus a *learned structural causal model*, and generating with a clamped `y` is `do(y)`
   **in the model's world**. Whether that matches the real world's `do(y)` is exactly the
   no-unobserved-confounding question — but the point worth stating in the doc is that the causal
   directionality is not something we add later, it is baked into the model we already train.

2. **The residual-structure machinery is a confounder/ignorability *test* — the most defensible
   causal contribution.** Flower's premise ("what does `y` not explain?") makes the residual
   latent precisely the candidate space for *unobserved confounders*. The permutation-null /
   independence test already in §4.4 is the tool: if `MI(residual_latent, y) ≈ 0`, the residual
   cannot confound the `y → x` edge and the causal reading is more credible; if it is clearly
   nonzero, we have *detected* a violation of ignorability. The honest framing is therefore not
   "we estimate causal effects" but "we provide a testable *necessary condition* for the
   conditional effect to be causal, and flag when it fails." This is genuinely useful and rare, and
   reuses machinery already in the benchmark plan.

3. **dSprites is a bona fide causal-effect demonstration, by construction.** Its factors are
   sampled *independently* and then deterministically rendered — independent randomization plus a
   known mechanism means no confounding, so the observational conditional exactly equals the
   interventional one. On dSprites, PMI(latent, factor) genuinely measures each factor's causal
   influence, not by assumption but because the data-generating process is known. This gives a
   clean "our estimator recovers causal effect where ground-truth causality is known" result that
   slots directly into the existing dSprites experiments (§4.5), and serves as the positive control
   for the ignorability test in point 2.

A fourth, more speculative extension: a good **conditional-MI estimator is a conditional-
independence oracle** (`MI(A,B|C) = 0 ⟺ A ⊥ B | C`), the bottleneck primitive for
constraint-based causal-discovery algorithms (PC, FCI) in continuous high-dimensional spaces. MINDE
(Related Work) already leans on conditional MI and data-processing-inequality self-consistency, so
the generative-MI → CI-testing path has precedent; pursuing it here would be a separate project,
noted for completeness rather than scoped in the timeline.

**Bottom line**: this does not make the work "causal inference" on real spectra — the confounding
problem there is real and unsolved by any amount of joint-distribution modeling, and the flatness
caveat (§2.3 geometry) compounds it for per-object claims. What it legitimately supports is (a) a
causal-effect recovery result on dSprites where the mechanism is known, (b) an ignorability *test*
on real data via residual independence, and (c) a CI-testing primitive for discovery. Points 1 and
2 are the ones to build on: both are honest and both use machinery this design already includes.

## 7. Conclusion & Positioning

Given the prior-art landscape in Related Work above, this splits into two claims of very different
strength, and they should not be conflated:

1. **Estimator quality — a narrow, targeted claim, not a general one.** N-MIENF (Butakov et al.)
   is the closest, most mechanistically distinct competitor, has proven consistency and
   non-asymptotic error bounds we don't currently have, and its cheap variant is a Gaussian-
   restricted lower bound with a documented failure mode (Related Work, point 2). A believable
   version of this claim is *not* "we beat SOTA MI estimation" but: our estimator, obtained at zero
   marginal training cost from a model flower already trains, tracks true MI in the specific
   non-Gaussian/deterministic-coupling regime where N-MIENF's cheap bound is known to saturate —
   and is otherwise competitive with KSG/MINE/InfoNCE on the standard `bmi` suite. That is a much
   narrower, more defensible, and still genuinely useful claim.
2. **Geometric interpretability — a variant formulation, honestly scoped.** Time-resolved and
   pointwise information decompositions from conditional generative models already exist in the
   diffusion/MMSE formulation (Kong et al., MMG — Related Work), so this cannot be claimed as a new
   capability class. What is ours: the deterministic probability-flow-ODE flux formulation (§2.3,
   §4.5) — exact per individual sample, carrying a literal continuity-equation/transport reading —
   plus the §2.4 displacement identity for the CFG mismatched-prior case, a small original lemma.
   This is worth reporting as a complementary formulation with per-sample resolution, and does not
   depend on winning any accuracy comparison — but it must be positioned alongside Kong et al./MMG,
   not above them.

**Publishability, in short**: unlikely to clear the bar as a general "new SOTA MI estimator" paper
— that space is crowded and partially ahead of us on theory. More likely to clear the bar as (a) a
targeted empirical result in the specific regime identified above, framed honestly against N-MIENF
rather than around it, plus (b) an applied-interpretability contribution — MI and a geometric
diagnostic falling out for free from a production scientific pipeline (dSprites/spectra), which is
a legitimate contribution for an applied-ML or domain-workshop venue even without a clean win on
raw estimator accuracy.

If the targeted-regime result holds up, it's directly reusable across every dataset flower already
supports, and ties naturally into the model-consolidation work tracked in issue #7 (a single shared
flow/likelihood loader would make this estimator a drop-in diagnostic for any future flower
experiment, not a one-off analysis).

## 8. Timeline / Effort Estimate

Assumes one engineer already familiar with the codebase, working solo, full-time-equivalent days.
Ranges reflect genuine uncertainty, not false precision — CNF likelihood work in particular has a
history of surprising people with numerical issues. Does not yet include the additional
targeted-regime experiment against N-MIENF proposed in Related Work/§7 — add ~1–2 days for
implementing N-MIENF (or adapting Butakov et al.'s reference code, if available) as a fourth
baseline once the design above is confirmed as the right one to pursue.

| # | Task | Estimate | Why |
|---|---|---|---|
| 1 | Wire `log_prob(x,y)` via `compute_likelihood` for `LightningFlowMatching` (dSprites first) | 2–3 days | Straightforward per §3, but first integration always surfaces shape/dtype/`model_extras` plumbing surprises against the shared `VelocityField`/`WrappedModel`. |
| 2 | Extend to RGBMNIST | 0.5–1 day | Same pattern, mostly copy/adapt once (1) works. |
| 3 | Validation gate: fit flow on a `bmi` task, confirm `compute_likelihood` recovers analytic density (both `exact_divergence` settings) | 1–2 days | New `synthetic_bmi_Flow` config + training run + comparison script. Biggest early risk: if this doesn't converge cleanly, everything downstream stalls — don't proceed past this gate on a hope. |
| 4 | Integrate `bmi` (KSG/MINE/InfoNCE/NWJ) and `mibenchmark` (BSC trick for RGBMNIST ground truth) as baselines | 1.5–2 days | Mostly dependency wiring per §4.6, not algorithm work. |
| 5 | Run full `bmi` task-suite benchmark (dense/sparse/transformed, multiple dims, multiple seeds) + compute bias/variance metrics | 3–4 days | Dominated by training+eval compute across the task grid, plus writing the metrics/aggregation script once. |
| 6 | dSprites experiments (positive/negative controls, permutation-null and factor sanity checks) | 2–3 days | Reuses existing config; mostly experiment-running + sanity-check scripting. |
| 7 | RGBMNIST experiments (BSC-trick ground truth, comparison to baselines) | 2–3 days | Same shape as (6), new dataset wiring. |
| 8 | Interpretability case studies (§4.5 flux-profile plots across factors) | 1–2 days | Visualization + qualitative write-up, lower risk but open-ended by nature. |
| 9 | Results write-up / doc update | 1–2 days | Compiling into this design doc or a follow-up results doc. |

**Subtotal:** ~15–22 engineering days (excluding the N-MIENF baseline noted above).

**Risk buffer: +25–40%** for CNF-likelihood-specific failure modes — reverse-ODE numerical
instability, the Hutchinson estimator's variance being worse than expected at higher latent
dimensionality, or the exact-divergence path being too slow on RGBMNIST/spectra-scale latents and
needing rework. This is the single biggest source of schedule risk in the whole plan, more than
any individual task above.

**Total: ~4–6 weeks full-time solo**, or roughly **8–10 weeks** at typical part-time/shared-
attention pace.

One thing not captured in "engineering days": actual training/HPC queue wait time (task 5 in
particular trains many small flows across the `bmi` task grid). On a shared cluster that can add
real calendar time without adding work — worth checking queue times before committing to a
calendar date rather than a days-of-effort number.
