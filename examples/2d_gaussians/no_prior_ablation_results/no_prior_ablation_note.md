# What does the conditional prior buy? — 2D-Gaussian ablation

Companion to `results.csv` / `summary.txt` (produced by `no_prior_ablation.py`).

Three arms, identical optimiser, iterations, batch size, warmup and CFG layout; the only
difference is where `x_0` comes from. 3 seeds each, 20,001 iterations.

## Expectation

A sufficiently expressive velocity field can absorb the base: whatever a learned
conditional base supplies, the network can supply instead, so `p(x_0|y)` is expected to
approach `N(0, I)` and the learned-base and fixed-base arms are expected to land at
parity on any measure that scores them on the same problem. This toy is small enough for
that regime, so parity is the predicted result rather than a null one.

## Read `cfm_cond_fixed`, not `cfm_cond` or straightness

`cfm_cond` and the straightness integral `S` score each arm against *its own* base, so an
arm whose base sits nearer the data is graded on a shorter transport problem. They are not
comparable across arms and no cross-arm claim should rest on them.

`cfm_cond_fixed` scores every arm's conditional pathway from a **common** `N(0, I)` base
against the true `y`, so all three solve the identical transport problem and only the
velocity field varies. It is the only conditional column where a difference is
attributable to the model.

| Arm | cfm_cond (own base) | **cfm_cond_fixed** (common base) | cfm_uncond | S | KL | class acc (MLP) | dist R² |
|---|---|---|---|---|---|---|---|
| `no_prior` — `x_0 ~ N(0,I)`, no KL | 1.113 | **1.116** | 4.140 | 2.203 | — | 0.275 | 0.985 |
| `learned_prior_b0.1` — learnable, β=0.1 | 0.360 | **2.221** | 4.142 | 0.722 | 1.392 | 0.315 | 0.933 |
| `learned_prior_b1` — shipped, β=1 | 0.967 | **1.129** | 4.140 | 1.938 | 0.036 | 0.280 | 0.978 |

At the shipped β=1 the two arms agree to 0.014 on `cfm_cond_fixed` (1.116 vs 1.129),
which is the expected parity. The KL column shows why: β=1 drives the learned base to
KL 0.036, i.e. essentially back to `N(0, I)`. At β=0.1 the base is left far from the
prior (KL 1.392) and consequently does worse when scored from a common base.

Note the direction of the metric's bias: `cfm_cond_fixed` is `no_prior`'s own training
distribution and off-distribution for the learned-prior arms, so it favours `no_prior`
slightly. The 1.2% gap at β=1 sits within that.

**Removal is unaffected by the base.** Class accuracy is 0.275–0.315 against 0.25 chance
for all three arms, and `dist_r2_mlp` is 0.93–0.98. Whatever the base contributes here, it
is not condition removal.

**`cfm_uncond` is identical across arms by construction** (4.140–4.142), which doubles as
a sanity check that the arms differ only where intended.

## Reproducing

```bash
cd examples/2d_gaussians
python no_prior_ablation.py --variants no_prior --learned-betas 0.1 1.0 \
    --outdir no_prior_ablation_results
```

GPU; 3 arms x 3 seeds x 20k iterations, ~20 min.

**The bare `python no_prior_ablation.py` does not reproduce this table.** Its default arms
are `learned_prior_b1` / `learned_prior_b0` / `no_prior` — β=0 rather than the β=0.1 arm
quoted above. Use the command as written.

Probes, `S` and `cfm_loss` reuse `beta_sweep.py`'s definitions verbatim, so these rows
concatenate with its `results.csv`.
