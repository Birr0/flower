↑ [Back to README](README.md)

# Method

## 1. Background & Notation

- `x` — the CFM model's latent target (the VAE latent / flow endpoint `z`), per CLAUDE.md's
  description of the encoder → CFM pipeline.
- `y` — a single known conditioning variable drawn from a dataset's `y_catalog` (e.g. rotation in
  dSprites), embedded via `ConditionEmbedder` (existing, `src/flower/models/modules.py:158-170`).
- `v(x, t; y)` — the shared `VelocityField` network (existing, `src/flower/models/modules.py`),
  evaluated with either the true condition embedding (`v_cond`) or the learned null-condition
  embedding `null_y` (existing, `VelocityField.null_y`, `modules.py:216-219`) to produce
  `v_uncond`. **One network, two inputs** — confirmed in `WrappedModel.forward`
  (`modules.py:55-84`), which batch-doubles at inference and combines
  `v_uncond + cfg_scale * (v_cond - v_uncond)`.
- CFG training (existing, `LightningFlowMatching.base_step`, `modules.py:340-367`): every batch
  is deterministically split 50/50 — the conditional half draws `x_0 ~ N(μ(y), σ(y)²)` from a
  learned conditional prior; the unconditional half draws `x_0 ~ N(0, I)`, an independent
  standard-normal sample. This is **not** stochastic per-sample dropout with probability `p`,
  and — important for §2 below — **the two halves have different base distributions and are not
  coupled to a shared noise draw.**
- PMI(x, y) = log p(x|y) − log p(x); MI(X, Y) = E_{(x,y)~p(x,y)}[PMI(x,y)].

## 2. Theory

### 2.1 Per-pathway log-likelihood

For a CNF/CFM model, the instantaneous change-of-variables formula gives, for a trajectory
`x(t)` running from base sample `x(0) = z₀` to data point `x(1) = x`:

```
log p₁(x) = log p₀(z₀) − ∫₀¹ ∇·v(x(t), t) dt
```

This holds separately for the conditional pathway (`v = v_cond`, `p₀ = N(μ(y), σ(y)²)`) and the
unconditional pathway (`v = v_uncond`, `p₀ = N(0, I)`), each integrated by reverse-solving the
model's own ODE from `x` back to its own `z₀`.

### 2.2 PMI decomposition

Subtracting:

```
PMI(x, y) = ΔBase + ΔFlux

ΔBase  = log p₀,cond(z₀,cond) − log p₀,uncond(z₀,uncond)
ΔFlux  = −∫₀¹ [∇·v_cond(x_cond(t), t) − ∇·v_uncond(x_uncond(t), t)] dt
```

**This is an exact identity** (up to ODE-solver numerical error), not a bound. Both priors are
closed-form Gaussians (`N(μ(y), σ(y)²)` and `N(0, I)` respectively, per §1), so `ΔBase` costs
nothing beyond evaluating a Gaussian log-density at each pathway's recovered `z₀`.

Note the base distributions are **not** shared or coupled in the current implementation (§1) —
an earlier version of this design assumed CFG training implies a shared base distribution across
pathways; that assumption does not hold as implemented, so we keep the general two-term form
rather than the reduced pure-flux form. This is a deliberate scope decision (see §6): unifying
the priors to unlock the pure-flux simplification is future work, not part of this experiment.

### 2.3 Geometric interpretation

By the continuity equation, `∂p/∂t + ∇·(pv) = 0`, so `∇·v` is exactly the local rate at which a
probability-carrying parcel expands or contracts as it's advected by the flow — flux density in
the literal vector-calculus sense. `ΔFlux`'s integrand, `∇·v_cond(t) − ∇·v_uncond(t)`, is
therefore a **time-resolved profile** of where along the generative trajectory conditioning on
`y` causes the model to compress or expand probability mass relative to the unconditional model.
Integrating it gives a single scalar (part of PMI); *not* integrating it — i.e. plotting the
integrand against `t` — gives a mechanistic, per-timestep account of when the conditioning signal
is injected. None of the classical baselines in §4.2 (KSG/MINE/InfoNCE) offers an analogue of
this; among generative approaches, Kong et al. and MMG (Related Work) provide decompositions of
the same kind in the diffusion/MMSE formulation — ours differs in being exact per sample along a
deterministic trajectory, not in existing at all.

### 2.4 `ΔBase` as displacement, not correction

`ΔBase` need not be treated as an opaque bookkeeping term. Because both base densities are
Gaussian (§1), it decomposes exactly into a displacement (work) term plus a volume term, giving
the whole of PMI a unified transport/potential reading rather than "flux term + leftover."

Whiten the conditional pathway's recovered base point through its own prior:
`ε_cond = Σ(y)^{-1/2}(z₀,cond − μ(y))`, so `ε_cond` is standard normal, matching the coordinate
system of `z₀,uncond` (already `~N(0,I)`, no whitening needed). In these shared coordinates, both
log-densities are the same quadratic potential `U(z) = ½‖z‖²`
(`log N(z;0,I) = −U(z) − (d/2)log 2π`), so:

```
ΔBase = −U(ε_cond) + U(z₀,uncond) − ½ log det Σ(y)
```

Apply `‖a‖² − ‖b‖² = (a−b)·(a+b)` to the two `U` terms:

```
ΔBase = Δ · z̄  −  ½ log det Σ(y)

Δ  = z₀,uncond − ε_cond          (displacement between the two recovered base points, whitened coords)
z̄  = (z₀,uncond + ε_cond) / 2    (their midpoint)
```

`Δ · z̄` is exactly the work done moving a particle by displacement `Δ` through the quadratic
potential `U` — since `∇U(z) = z` is linear, the mean-value identity
`∫₀¹ ∇U(z̄ + (s−½)Δ) · Δ ds = Δ · z̄` holds with no approximation, not just to leading order. The
remaining `−½ log det Σ(y)` term is the whitening map's constant Jacobian (a volume/dilation
contribution from the conditional prior's anisotropic scaling), and is separate from the
displacement itself.

So the full identity reads:

```
PMI(x, y) = ΔFlux  +  Δ·z̄  −  ½ log det Σ(y)
```

three physically legible pieces: flux work accumulated along the two actual ODE trajectories,
displacement work through the ambient quadratic potential between where those trajectories land
in base space, and a volume-dilation term from the learned conditional prior's covariance. This
is a meaningfully better story than "flux term plus a leftover base-density correction."

**Caveat, stated precisely so it isn't overclaimed**: this decomposition is *exact* only because
`U` is exactly quadratic, which follows from the Gaussian-base assumption already in place (§1,
§2.2). If either base distribution weren't Gaussian, `Δ·z̄` would be the leading (linear) term of
a Taylor expansion of the true potential difference, not an exact identity — this reframing is a
direct consequence of the closed-form-Gaussian prior, not a property that would survive relaxing
that assumption.

### 2.5 Connection to non-equilibrium free-energy estimation (Jarzynski equality)

The energy reading isn't just terminology. Under the Boltzmann/energy-based-model convention
`E(z) := −log p(z)` (so `p(z) ∝ exp(−E(z))`), PMI is by definition a negative energy difference,
`PMI(x,y) = −[E_cond(x) − E_uncond(x)]`. §2.4's decomposition gives that energy difference the
same structure as a **non-equilibrium free-energy relation** from statistical mechanics, and the
correspondence is precise enough to name term-by-term.

The classical **Jarzynski equality** (Jarzynski 1997, *Phys. Rev. Lett.* 78, 2690) and **Crooks
fluctuation theorem** (Crooks 1999, *Phys. Rev. E* 60, 2721) relate a free-energy difference `ΔF`
between two equilibrium states A, B to the work `W` done while driving a system between them along
a stochastic, non-equilibrium protocol: `exp(−βΔF) = ⟨exp(−βW)⟩_A`, averaged over noisy
realizations of the protocol starting from equilibrium in A. **Targeted free energy perturbation**
(Jarzynski 2002, *Phys. Rev. E* 65, 046122) specializes this to the case where A and B are
connected by a deterministic, invertible map `Ψ`: then the identity holds with the log-Jacobian of
`Ψ` playing the role of the work term, and — because there's no thermal noise to average over —
the relation becomes **exact for every single sample**, not just in expectation. This is exactly
the machinery behind **Boltzmann generators** (Noé, Olsson, Köhler & Wu 2019, *Science* 365,
eaaw1147) and **stochastic normalizing flows** (Wu, Köhler & Noé, NeurIPS 2020), where a
normalizing flow's accumulated log-Jacobian is used as the deterministic work term to estimate
molecular free-energy differences.

Our construction is a direct instance of this, term for term:

- **States A and B** are the two base ("reference") ensembles — `N(0,I)` for the unconditional
  pathway and `N(μ(y),Σ(y))` for the conditional one — each a harmonic/quadratic-potential system
  in the physics sense.
- **The protocol connecting them** is provided by the two reverse-time flow-matching ODEs: running
  `v_uncond`/`v_cond` backward from the same data point `x` is exactly "pulling" x back to its
  respective reference ensemble along a designed, deterministic trajectory.
- **`ΔFlux`** (§2.1–2.3) is exactly the deterministic work/log-Jacobian term from targeted free
  energy perturbation — the same object that Boltzmann generators use as their reweighting
  correction, just computed via `compute_likelihood`'s continuous-time divergence integral (§3)
  instead of a discrete coupling flow's analytic Jacobian.
- **`Δ·z̄ − ½ log det Σ(y)`** (§2.4) is exactly the closed-form free-energy difference between the
  two reference ensembles themselves (two harmonic oscillators — one isotropic, one anisotropic
  and shifted), which is analytically tractable precisely because both are Gaussian.

Because our maps are deterministic ODE flows rather than noisy Langevin/MCMC steps, our identity
holds exactly *per sample* — the cleaner, deterministic regime of targeted FEP, not the noisier
general Jarzynski/Crooks setting that requires averaging over stochastic realizations. That's a
meaningful bonus, not just a restatement: it means every single PMI evaluation is already the
"debiased" quantity that stochastic free-energy estimators need many samples to approximate.

**Scope of this connection, stated honestly**: the generative-likelihood ↔ free-energy/Jarzynski
link is *foundational to this model family, not a discovery of this work*. Diffusion models were
explicitly built on Jarzynski's equality and annealed importance sampling from their inception
(Sohl-Dickstein et al. 2015, *"Deep Unsupervised Learning using Nonequilibrium Thermodynamics"*,
ICML 2015; AIS itself — Neal 2001 — is Jarzynski's identity in statistical form), and the
flow-log-Jacobian-as-work identity is standard practice in the Boltzmann-generator literature
cited above. What this section adds is only the explicit term-by-term specialization to the CFG
**mismatched-prior** case — two different reference ensembles, connected to the same data point by
two different deterministic protocols — with §2.4's displacement/volume split of the
reference-ensemble free-energy difference as the one piece for which we found no precedent. Cite
the lineage; present §2.4 as a small original lemma within it; do not claim the
information–energy relation itself as novel (see Related Work, point 5).

## 3. Implementation Prerequisite: Log-Likelihood

**Not custom code — already available in a dependency already in use.** The installed
`flow_matching` package (v1.0.10, already a dependency; its `ODESolver` is already used in
`predict_step`/`test_step`) ships `ODESolver.compute_likelihood`
(`flow_matching/solver/ode_solver.py:106-195`), which does exactly what §2.1 needs:

```python
sol, log_p1 = solver.compute_likelihood(
    x_1=x,                # target sample
    log_p0=log_p0_fn,     # closed-form log-density of the base distribution
    step_size=None,
    time_grid=torch.tensor([1.0, 0.0]),  # must run 1 -> 0 for likelihood
    exact_divergence=False,  # True = exact autograd trace; False = Hutchinson estimator
    method="dopri5",
    **model_extras,        # e.g. y for the conditional pass, null_y for the unconditional pass
)
```

It reverse-integrates the ODE from `t=1` to `t=0` while augmenting the state with a running
divergence integral — exactly the instantaneous-change-of-variables computation in §2.1 — and
exposes the exact/Hutchinson choice flagged in §2.3/§6 as a single boolean:

- `exact_divergence=True`: per-dimension `torch.autograd` trace (`ode_solver.py:158-162`) — exact,
  cost scales linearly with latent dimensionality (one backward pass per dimension). Suitable for
  dSprites/RGBMNIST-scale latents.
- `exact_divergence=False`: Hutchinson estimator via a fixed Rademacher random projection
  `z ∈ {−1,+1}ᵈ` (`ode_solver.py:145-147, 163-173`), `E[zᵀ D_x(v_t) z]` — O(1) backward passes
  regardless of dimension, at the cost of estimator variance from the random projection. Needed
  for higher-dimensional latents (spectra).

So the actual implementation work is thin wiring, not a from-scratch CNF likelihood: a
`log_prob(x, y)` method on `LightningFlowMatching` (or a small new module,
`flower/models/likelihood.py`) that (a) supplies `log_p0` as the closed-form conditional-prior
Gaussian for the conditional pass and the closed-form `N(0,I)` for the unconditional pass (§1,
§2.2), (b) calls `compute_likelihood` twice — once per pathway, passing `y` vs. `null_y` through
`model_extras` into the shared `VelocityField` — and (c) subtracts to get PMI per §2.2.

Two things to still verify/decide, not assume:
- **Validation before trusting this on real data**: fit a flow to a distribution with a known
  closed-form density (a `bmi` task, §4.6) and confirm `compute_likelihood` recovers the analytic
  density to solver tolerance, for both `exact_divergence` settings. This is a correctness gate on
  our wiring, not on the library, but still a required gate before any reported MI number.
- **Which `exact_divergence` setting per dataset**, and whether Hutchinson's extra variance is
  small enough at RGBMNIST/spectra latent dimensionality to trust — this should be an empirical
  check (compare both settings on the low-dimensional datasets where exact is affordable, before
  committing to Hutchinson-only for the higher-dimensional ones), not a default assumption.
- Solver settings (`method`, `step_size`/`time_grid`, `atol`/`rtol`) trade numerical accuracy for
  compute cost directly and must be reported alongside every MI estimate.
