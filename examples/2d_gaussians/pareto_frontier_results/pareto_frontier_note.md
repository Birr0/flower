# Removal vs preservation — Flower's own frontier on the 2D toy

Companion to `pareto_frontier.{csv,png,txt}` (produced by `pareto_frontier.py`).
Answers **R1 §W4 / issue #27 (E8)**: *"the paper should better quantify the tradeoff
between condition removal and preservation of residual information."* Built
2026-07-26 from results already on disk — no new compute.

Axes: removal = mode accuracy under an MLP probe (chance 0.25, **lower better**);
preservation = distance R² under an MLP probe (**higher better**). Both use the
nonlinear probe deliberately: a linear probe reads this toy's radial distance at
R² ≈ 0 whatever you do (`omega_probe_note.md` §3), so a linear preservation axis
would be uninformative for every method at once.

## What is new here

Every previous frontier figure we have (`examples/mnist/plot_combined_frontier.py`)
sweeps the *baselines* over their knob and shows Flower as a single point. That
understates the answer to W4. This traces **Flower over both of its own knobs**:

- **β** — the training-time knob (19 values × 3 seeds, at ω=1), merging the wide
  log-spaced `beta_sweep_results/` with `beta_sweep_transition_results/`
  (β = 0.02–0.3). **Use both**: the wide sweep alone jumps 0.01 → 0.05 → 0.1 and
  misrepresents a continuous transition as a step (see finding 1).
- **ω** — the inference-time guidance weight (11 values × 3 seeds,
  `omega_probe_results/`), at β=1.

The two sweeps come from different scripts with slightly different probe settings,
so they are only comparable if they agree where they overlap. They do — at the
shared setting (β=1, ω=1) removal differs by **0.010** and preservation by
**0.009**, within the seed spread. That check is printed every run; if it ever
widens, the figure is invalid.

## Findings

**1. β is a real dial over one decade, then saturates — but no point on the dial is
one you would choose.** *(Corrected 2026-07-27: an earlier version of this note said
"β is a switch, not a dial". That was an artifact of the wide sweep's log spacing,
which jumps 0.01 → 0.05 → 0.1. Merging `beta_sweep_transition_results/`
(β = 0.02–0.3, same script, same probes, same 3 seeds) fills the gap and the
transition is plainly continuous.)*

Removal falls smoothly and monotonically across β ≈ 0.01–0.15 — 0.984, 0.797, 0.610,
0.425, 0.367, 0.322, 0.298 at β = 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15 — and then
saturates: β ≥ 0.15 is flat to within 0.02 across four further orders of magnitude
(0.295, 0.298, 0.294, 0.282 … 0.279 at β = 0.2 … 100).

What survives from the "switch" framing, and is the point worth making, is that
**every transition point is dominated on both axes by the saturated region**: β=0.15
gives (0.298, 0.964) against β=1's (0.282, 0.978), and mid-transition preservation
dips to **0.673** at β=0.03. So β is a genuine graded control over removal, but not a
useful *trade-off* dial — there is no operating point in the transition that trades
preservation for removal advantageously. It is a constraint you turn up until it
saturates. **Do not** revive an operating-point recommendation in the transition
region; that was retracted separately (R3 §A4b, on the coupling-artefact grounds in
`residual_floor_analysis.md`) and this data agrees with the retraction.

**2. ω moves removal smoothly but is not a frontier.** Removal falls monotonically
(1.000 → 0.272), but preservation is U-shaped — 0.977 at ω=0, down to **0.424** at
ω=0.4, back to 0.987 at ω=1. Every intermediate ω is **dominated by ω=1 on both
axes at once**, so the curve is a path, not a trade-off. Cause and the two ruled-out
explanations are in `omega_probe_note.md` §4; the same guardrail applies here —
phrase the dip as *recoverable* residual, never as information destroyed.

**3. ⚠ Flower does not win on this toy, and we should not pretend otherwise.**
The Pareto front is a **single point and it is a baseline**: FastICA residB at
(0.250, 1.000) dominates everything, including every Flower setting. iVAE residB
(0.263, 0.994) is next. This is not a weak-probe artifact — the probe-free
corroboration in `correlation_metrics_results/` is unambiguous: residB has
η = 0.044 against a null of 0.024, and **MCC = 1.000**, i.e. it recovers the
ground-truth seed exactly.

**Why, and why it is not damaging.** In this toy the condition enters as a pure
per-class additive mean offset (`data.py`: `x = μ_class + δx`), and the residual
*is* the mean-subtracted displacement. Subtracting a conditional mean therefore
**inverts the generative process exactly**. No method can beat exact, and a method
that matched it would only be rediscovering the generator. The toy is a
*calibration* case — it tells us the metrics are sane and that Flower lands within
0.03 removal / 0.02 preservation of the exactly-optimal solution **without being
given the generative form** — not a discriminating one.

The discriminating cases are the ones where the condition is *not* an additive mean
offset, and there the ordering reverses hard: on cMNIST the same residB
construction reads 0.114 to a linear digit probe and **0.858** to an MLP (iVAE
residB: **0.999**), against Flower's 0.748/0.792 (`results_index.md` §1).

## Guardrails

- **Never show this panel without the cMNIST/spectra pairing.** Alone it reads as
  "a linear baseline beats Flower". With the pairing it makes a stronger point than
  either does separately: the baseline is *exactly right* when its assumption holds
  and *collapses* when it does not, which is precisely the argument for a method
  that does not assume the form of the condition.
- Do not describe the ω curve as a frontier (finding 2).
- β *is* a graded control of removal over β ≈ 0.01–0.15 — but say in the same breath
  that the whole transition is dominated by the saturated region, or it reads as an
  invitation to tune β, which the retraction in R3 §A4b forbids (finding 1).
- Both knobs dip in preservation mid-transition (β: 0.673 at β=0.03; ω: 0.424 at
  ω=0.4) and recover at the endpoints. Tempting to call this one phenomenon; we have
  not established that, so describe them separately.
- Baseline points are **single-draw** (`correlation_metrics.py` runs at one
  `RANDOM_STATE`), while the Flower curves are 3-seed means with s.d. bars. Do not
  read small baseline-vs-Flower gaps as significant. See `TODO.md`.

## Not covered — the cMNIST β-frontier, descoped

Flower's β-frontier on cMNIST is **not** a re-plot, contrary to the "Low (re-plot)"
estimate in `rebuttal_issue_plan.md`. It needs one embedding per β, and
`$DATA_ROOT/rgbmnist/` holds only `rgbmnist_Flow_cond_prior` (single run
`7518770_0`), `rgbmnist_Flow_smoke_test` and `rgbmnist_VAE` — no β-ablation
embeddings, despite the `rgbmnist_Flow_beta_ablation` config existing. Producing
that panel means training runs.

**Descoped rather than deferred.** The training cost is out of proportion to what a
second frontier adds: the toy is the only setting where *both* of Flower's knobs are
swept end to end, so it is the only place the frontier shape is informative, and the
cMNIST answer already exists in a different form —
`examples/mnist/combined_frontier.png`, baselines swept over k with Flower as a point.

**This does not weaken the pairing requirement above.** The pairing that finding 3
demands is the cMNIST/spectra *matched-removal* comparison, not a cMNIST frontier —
`examples/mnist/ivae_sweep_results/` and `examples/spectra/ivae_sweep_paper_eval_results/`,
both committed. Those are what show the residB construction collapsing (digit 0.114
linear vs 0.858 MLP) where it was exact here.

## Re-run

```bash
cd examples/2d_gaussians
python pareto_frontier.py 2>&1 | tee pareto_frontier_results/pareto_frontier.log
```

Pure re-plot, seconds. Inputs: `beta_sweep_results/results.csv`,
`omega_probe_results/omega_probe.csv`,
`correlation_metrics_results/correlation_metrics.csv`.
