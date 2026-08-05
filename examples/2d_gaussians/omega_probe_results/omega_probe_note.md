# Fig 3b explainer — does guidance *inject* the residual, or just linearise it?

Companion to `omega_probe.{csv,png,txt}` (produced by `omega_probe.py`).
Answers **R3 §A5 / issue #25 (E6)**. Run 2026-07-26, 3 seeds × 10 000 samples,
shipped checkpoint `checkpoints/cond_fm.pth` — the same weights behind Fig 3b.

Reviewer 3:

> "The metric R² increases as ω moves from 1 to 0, which seems to imply that the
> flow-matching process injects more information about the distance variable.
> This appears counterintuitive."

Three separate things are wrong or missing in that reading. Numbers below are
mean ± s.d. over 3 seeds.

---

## 1. The ω axis reads the other way (already known, now measured)

| ω | δx₀ R², linear probe |
|---|---|
| 0.0 | **0.361 ± 0.022** |
| 1.0 | **0.997 ± 0.000** |

R² *decreases* as ω goes 1 → 0, the opposite of the reviewer's stated direction.
This confirms from a script what `rebuttal_issue_plan.md` had confirmed by eye
off the rendered notebook (yellow ω=1 on top at t=0). Our caption is correct.

## 2. "More R² " is not "more information" — the raw data is the ceiling

The residual is fixed at generation time: `diff` is each point's displacement
from its own cluster mean. The flow cannot create it. The test for injection is
therefore the **raw-data ceiling** — the seed cannot contain more of the residual
than `x₁` did:

| representation | δx₀ linear | δx₀ MLP |
|---|---|---|
| raw data `x₁` | 0.057 ± 0.011 | **0.996 ± 0.004** |
| seed, ω=0 | 0.361 ± 0.022 | 0.972 ± 0.007 |
| seed, ω=1 | 0.997 ± 0.000 | **0.997 ± 0.000** |

The nonlinear probe on the seed never materially exceeds the raw-data ceiling
(0.997 vs 0.996 at ω=1 — equal within the ±0.004 seed spread). **Nothing is
injected.** What changes across ω is the *linear* probe, which climbs 0.361 →
0.997 to meet a nonlinear reading that was already ~0.97 at ω=0.

Note the raw data itself: δx₀ is 0.996 to an MLP but **0.057** to a linear probe.
In `x₁` the residual is fully present and almost entirely linearly *inaccessible*.
That is the cleanest statement of what guidance does here — it rotates an
existing quantity into linear view.

## 3. The reviewer's "distance variable" is not what Fig 3b plots

Fig 3b's legend is δx₀ / δx₁ — the two **components** of the displacement. The
scalar distance `d = ‖δx‖` behaves completely differently:

| ω | `d` linear R² | `d` MLP R² |
|---|---|---|
| 0.0 | 0.0005 | 0.977 |
| 0.4 | 0.0003 | 0.424 |
| 1.0 | −0.0006 | 0.987 |

`d` is linearly unreadable at **every** ω, ω=1 included — it is radially
symmetric, so no linear function of the coordinates can track it (the same
failure mode documented for this toy in `CLAUDE.md`: max |ρ| at chance while an
MLP reaches R² 0.999). Had Fig 3b plotted the distance, the curve would be flat
at zero and there would be no trend to find counterintuitive. Worth stating
plainly in the response, and it doubles as the definition R3 §A6 asks for.

---

## 4. ⚠ What did *not* come out as predicted: the nonlinear curve is not flat

The design predicted MLP R² ≈ flat and high across ω. It is not. It dips hard at
intermediate ω and recovers:

| ω | 0.0 | 0.1 | 0.2 | 0.4 | 0.7 | 1.0 |
|---|---|---|---|---|---|---|
| δx₀ MLP R² | 0.972 | **0.557** | 0.565 | 0.675 | 0.902 | 0.997 |
| mode acc (MLP) | 0.999 | 0.843 | 0.716 | 0.530 | 0.349 | 0.272 |

Two causes ruled out:

- **Not probe capacity.** At ω = 0.1/0.2/0.4 a 256×128 MLP and a 10-NN regressor
  land where the 64×32 MLP did (ω=0.1: 0.574 / 0.536 vs 0.556). See
  `omega_probe_capacity.csv`.
- **Not solver resolution.** ω=0.2 at 800 integration steps gives δx₀ MLP 0.550,
  identical to 100 steps.

The interpretation we can defend: a blended field `(1−ω)u_null + ω u_cond` is the
transport of *neither* model, so only the endpoints are coherent maps. **Do not
say information is destroyed** — the inversion is a deterministic ODE map and so
invertible in principle; what is measured is recoverability by standard probes.
Phrase every claim here as "recoverable residual", never "residual present".

**This is usable, not damaging.** Intermediate ω is *dominated* by ω=1 on both
axes at once — worse removal (mode 0.843 at ω=0.1 vs 0.272 at ω=1) **and** worse
residual recovery (0.557 vs 0.997). The shipped operating point ω=1 is not a
cherry-picked spot on a trade-off curve; it is the best point on both metrics
simultaneously.

---

## What to say to R3

Politely note the axis direction; then make the substantive point, which does not
depend on the direction at all: the residual is an intrinsic property of the
data, the raw-data ceiling shows the seed never carries more of it than `x₁` did,
and the ω-trend in Fig 3b is a statement about *linear accessibility*. Add that
the plotted quantity is δx₀/δx₁, not the scalar distance — and define both in the
revision (R3 §A6).

**Guardrails:**
- Don't claim the nonlinear curve is flat — it dips at intermediate ω. If we show
  this figure, we show the dip and explain it.
- Don't claim information is destroyed at intermediate ω (see §4).
- 2D-Gaussian toy only, single shipped checkpoint; seeds vary the data draw and
  the probe fits, not the trained model. `--train` re-fits per seed if a reviewer
  presses on model variance.

## Re-run

```bash
cd examples/2d_gaussians
python omega_probe.py --capacity-omegas 0.1 0.2 0.4 2>&1 \
    | tee omega_probe_results/omega_probe.log
```

No training — the checkpoint is loaded, so the cost is ODE inversions plus probe
fits, dominated by the 256×128 capacity ladder. Drop `--capacity-omegas` for the
main table alone. Constants and the exact invocation are in
`omega_probe_results/params.json`.
