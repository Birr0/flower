# Geometric Estimation of Mutual Information via Conditional Flow Matching

Status: design draft, not yet implemented. No code referenced here exists yet unless explicitly
marked "(existing)".

## Abstract

Flower trains a VAE encoder together with a conditional flow-matching (CFM) model over the
learned latent space, conditioned on known factors of variation (`y_catalog`). This setup lets
us ask a question that is central to interpretability and scientific discovery alike: *how much
of the latent structure does a known factor actually explain, and how much is left over?* That
question is a mutual information (MI) question — MI(latent, factor) — but the standard tools for
answering it are unsatisfying. Non-parametric estimators (KSG) don't scale to high-dimensional
latents. Neural variational estimators (MINE, InfoNCE/CPC) require training a *separate* critic
network, are known to underestimate MI badly and become high-variance once the true MI is large,
and give a single scalar with no mechanistic account of *where* the dependence lives.

We propose to estimate MI directly from log-likelihoods that fall out of models we are already
training, using the identity PMI(x,y) = log p(x|y) − log p(x), computed from the conditional and
unconditional pathways of a single classifier-free-guidance (CFG) trained CFM model. Because CFM
log-likelihood is computed via the instantaneous change-of-variables formula (an integral of the
velocity field's divergence along the probability-flow ODE), the resulting PMI decomposes into a
closed-form base-density term and a term with a genuine physical reading: the accumulated
difference in probability *flux* between the conditional and unconditional flows. This gives us
an estimator that costs no extra training, and — uniquely among MI estimators — a time-resolved,
geometric account of when and where along the generative trajectory the conditioning information
is injected.

Recent work (MINDE, Butakov et al.'s MIENF, a normalizing-flow difference-of-entropies
estimator, and diffusion-based information decompositions — Kong et al.'s interpretable-diffusion
work and the MMSE-gap estimator MMG; see Related Work below) has already explored
generative-model-based MI estimation, including pointwise estimates and per-noise-level
decompositions, so the contribution here is narrower than "a new way to estimate MI": it is
(a) an estimator that falls out, at zero extra training cost, of a conditional generative model
already being trained for generation/embedding in an existing scientific pipeline, and (b) a
*deterministic-ODE, divergence/flux* formulation of the time-resolved decomposition — exact per
sample, rather than an expectation over noise as in the diffusion/MMSE formulations — together
with a displacement/work identity for the mismatched-prior CFG case (§2.4) that we have not found
elsewhere. Whether that is enough to constitute a publishable contribution — and where,
specifically, it might outperform the closest prior work — is addressed directly in Related Work
& Novelty Positioning below.

## Related Work & Novelty Positioning

**This section exists to keep the claims above honest.** The core mechanism proposed here — PMI
as a difference of generative-model log-likelihoods along conditional vs. unconditional pathways
— is not new in the abstract; it has been published, in different concrete forms, at least five
times recently, including with pointwise and decomposed variants. Any framing of this work as "we
invented flow-based MI estimation" would be inaccurate and would not survive review. What follows
is a precise accounting of the closest prior work, and where a genuine edge might actually be
found.

### Closest prior work

- **MINDE** (Franzese et al., ICLR 2024, [arXiv:2310.09031](https://arxiv.org/abs/2310.09031)) —
  uses score-based diffusion models and a Girsanov-theorem argument to estimate a KL divergence
  (hence MI) as a difference between conditional and joint/marginal diffusion score functions.
  Same high-level move (generative-model internals → MI via a subtractive construction) applied to
  diffusion SDEs rather than flow-matching ODEs. Reports passing self-consistency tests (data
  processing inequality, additivity under independence) that are known pain points for classical
  estimators — a bar our design should also test against (§4.4 already includes some of these;
  worth adding data-processing-inequality and additivity checks explicitly).
- **Neural Difference-of-Entropies** (2025, [arXiv:2502.13085](https://arxiv.org/abs/2502.13085))
  — parametrizes conditional and marginal densities with normalizing flows and explicitly
  constructs "tractable PMI, Monte Carlo averaged" — essentially the same formal target as our
  §2.2, via an entropy-difference framing rather than a continuity-equation/flux framing.
- **Butakov et al., "Mutual Information Estimation via Normalizing Flows" (MIENF / N-MIENF)**
  (NeurIPS 2024, [arXiv:2403.02187](https://arxiv.org/abs/2403.02187)) — read in full (not just
  abstract) to get this right, since it's the most mechanistically distinct of the three and the
  one the user asked about directly:
  - Trains **two separate marginal diffeomorphisms** `f_X`, `f_Y` (bijective normalizing flows
    with tractable analytic Jacobians — RealNVP/Glow-style, not continuous-time ODEs) that each
    transform *one* variable toward a space where the transformed pair's MI is easier to estimate.
    `X` and `Y` are treated symmetrically; there is no conditional generative model `p(x|y)`
    anywhere in their construction — MI invariance under injective smooth maps (their Theorem 2.1)
    is the entire mechanism.
  - **General MIENF**: approximate the transformed joint density with a family `Q`; the resulting
    MI estimate is consistent as `Q` → universal, but with bias exactly `D_KL(p_{ξ,η} ‖ q_{ξ,η})`
    (their Corollary 3.2) — nonzero, and larger for a coarser `Q`.
  - **N-MIENF** (their cheap, practical variant): restrict `Q` to multivariate Gaussians. This
    yields a **closed-form MI estimate directly from a covariance matrix**, `O(d)` extra learnable
    parameters, and — critically — a **provable lower bound on true MI** whenever the marginals
    are Gaussian (their eq. 8, Corollary 4.3/4.4), with a known, quantifiable gap. Their own
    Remark 4.5 shows this bound can be *tight in one specific failure case*: `ξ ~ N(0,1)`,
    `η = (2B−1)·ξ` for `B ~ Bernoulli(1/2)` independent of `ξ` — a deterministic, non-Gaussian
    (sign-flip) coupling where the Gaussian-restricted bound saturates far below the true MI.
  - Benchmarked on their own high-dimensional synthetic data with known ground truth (not, as far
    as retrieved, the `bmi`/Czyż suite from §4.6 — worth checking their appendix for the exact
    benchmark before claiming direct comparability).
- **Kong et al., "Interpretable Diffusion via Information Decomposition"** (2023, ICLR 2024,
  [arXiv:2310.07972](https://arxiv.org/abs/2310.07972)) — derives *exact* expressions for MI and
  conditional MI directly from a (conditional) denoising diffusion model, including **pointwise
  estimates** and a **non-negative decomposition** fine-grained enough to attribute informative
  relationships between individual words and pixels. This is the closest prior art to our
  interpretability claim: PMI from a conditional diffusion model, with spatial attribution,
  published three years ago. It does not use the deterministic probability-flow-ODE divergence
  formulation, but it firmly establishes "PMI + information decomposition from a conditional
  generative model" as existing territory.
- **MMG, "Mutual Information Estimation via the MMSE Gap in Diffusion"** (2025,
  [arXiv:2509.20609](https://arxiv.org/pdf/2509.20609)) — estimates MI as an integral over noise
  levels of a per-level MMSE gap, via the I-MMSE relation. Because the estimator *is* an integral
  of a per-noise-level information density, it constitutes a **time-resolved MI decomposition in
  the diffusion/MMSE formulation** — directly analogous in role to our `ΔFlux` integrand profile
  (§2.3), differing in mechanism (stochastic MMSE/score quantities, in expectation) rather than in
  kind.

### Where the edge might actually be

1. **Zero marginal training cost, not a bespoke estimator.** N-MIENF requires training two new
   diffeomorphism networks specifically for each `(X, Y)` pair being analyzed. Our PMI is read off
   a CFG-trained CFM model that flower is *already training* for generation/embedding — no
   additional model, no additional training run. This is a real structural advantage specifically
   in flower's regime (a conditional generative model already exists for every dataset), not a
   general-purpose claim ("better MI estimator for any two random vectors") — it should be framed
   as the latter, more defensible claim.
2. **Exact identity vs. bias-by-restriction.** Our §2.2 decomposition is exact given exact
   reverse-ODE integration (no approximating family `Q` involved at all) — the only error sources
   are ODE solver tolerance and, if used, Hutchinson-estimator variance (§3), both controllable and
   reportable. N-MIENF's cheap variant is a Gaussian-restricted *lower bound* with a bias term that
   can be arbitrarily loose, as their own Remark 4.5 demonstrates on a deterministic non-Gaussian
   coupling. **This gives a concrete, targeted experiment**: construct exactly this kind of
   regime — non-Gaussian, deterministic or near-deterministic couplings; the `bmi` suite's
   sparse-interaction and Student-t (long-tailed) tasks in §4.6 are close analogues already in
   scope — and check whether our estimator tracks the true MI where N-MIENF's bound is known to
   saturate. A blanket accuracy sweep across all benchmark tasks is much less likely to show a
   clean win than this specifically-targeted regime.
3. **Time-resolved geometric interpretability (§2.3) — a formulation difference, not unprecedented
   territory.** An earlier draft of this document claimed no prior method offers a time-resolved
   decomposition; that claim does not survive contact with Kong et al. (pointwise MI + fine-grained
   information decomposition from conditional diffusion) and MMG (MI as an integral of a
   per-noise-level MMSE gap — functionally a time-resolved profile). What genuinely remains ours:
   the decomposition via the **deterministic probability-flow ODE's divergence/flux integrand**,
   which (a) is exact per individual sample rather than an expectation over diffusion noise, and
   (b) carries the literal continuity-equation/transport reading of §2.3 rather than an
   MMSE/estimation-theoretic one. That is a real difference in mechanism and in what can be read
   off a *single* trajectory, but it must be presented as a variant formulation alongside
   Kong et al./MMG, not as a new capability class. Diff-of-Entropies and N-MIENF remain static
   scalar estimators with no trajectory to decompose.
4. **Honest gap in the other direction.** N-MIENF comes with formal consistency proofs and
   non-asymptotic error bounds (their Theorem 3.1, Corollary 3.2–3.4). Our method does not
   currently have an analogous finite-sample theoretical guarantee — solver tolerance and
   Hutchinson variance are empirically controllable but not proven bounds. A reviewer comparing
   the two directly would reasonably ask for this; either derive one (harder, may not be tractable
   for CFM's ODE-based likelihood) or scope the contribution as empirical/applied rather than
   theoretical, and say so explicitly rather than let the gap go unaddressed.
5. **The free-energy connection (§2.5) is useful framing, not a differentiator.** The
   generative-likelihood ↔ free-energy/Jarzynski link is foundational to this model family, not a
   discovery of ours: diffusion models were *built* on Jarzynski's equality and annealed importance
   sampling from the start (Sohl-Dickstein et al. 2015, "Deep Unsupervised Learning using
   Nonequilibrium Thermodynamics"; AIS itself, Neal 2001, is Jarzynski's identity in statistical
   form), and Boltzmann generators / stochastic normalizing flows made the flow-log-Jacobian-as-
   work identity standard practice. What §2.5 contributes is only the *explicit, term-by-term
   specialization to the CFG mismatched-prior case* — including the `Δ·z̄ − ½ log det Σ(y)`
   displacement/volume split (§2.4), for which we found no precedent. Correct use of this material:
   cite the lineage, present §2.4's identity as a small original lemma within it, and do not claim
   the information–energy relation itself as novel.

**Bottom line**: this is very unlikely to clear the bar as "a new, more accurate MI estimator" in
a top-tier ML venue given points 1–2 and 4 above — that framing is already crowded and partially
outgunned on theory (N-MIENF has proven bounds we don't), and the interpretability claim must be
scoped as a variant formulation given Kong et al./MMG (point 3). It is most likely to clear the
bar as an applied contribution: MI estimation and a per-sample-exact flux diagnostic, obtained for
free from models flower already trains, validated on the specific regime (point 2) where the
closest competing cheap estimator is known to fail, with §2.4's displacement identity as a small
original theoretical lemma. That reframing should propagate into the abstract and conclusion (§7).

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

## 4. Benchmark Design

### 4.1 Datasets

| Dataset | Role | Ground truth |
|---|---|---|
| Synthetic tasks from `bmi` (see §4.6) | Calibration | Closed-form MI via the Czyż et al. 2023 40-task suite (`bmi.benchmark.BENCHMARK_TASKS`) — normal, Student-t, and diffeomorphically-transformed variants, dense and sparse correlation structure, dimensionality 1×1–25×25, MI up to ~2–5 nats. Requires a new minimal flow config (`src/conf/experiment/synthetic_bmi_Flow/`, new) trained on `bmi`-generated `(x, y)` pairs. |
| dSprites | Known-factor structure | No closed-form MI, but known generative factors (shape, scale, rotation, x, y — existing dataset/config) allow constructing both positive controls (y = a real factor) and negative controls (y = an independent/shuffled factor, expected MI ≈ 0). |
| RGBMNIST | Real-data scaling test | No closed-form ground truth by default; adopt the `mibenchmark` binary-symmetric-channel trick (§4.6) to construct a *known*, dialable ground-truth MI on MNIST-like pairs instead of relying on qualitative comparison only. |

### 4.2 Baseline estimators

- **KSG** (k-NN, non-parametric) — no training required, standard ground-truth-free reference.
- **MINE** (neural variational lower bound) — widely cited, known to underestimate at high MI.
- **InfoNCE/CPC** (contrastive lower bound) — common in representation-learning MI literature.

All three run on the *same* paired `(x, y)` samples as the flow-based estimator for direct
comparison. See §4.6 for existing packages that ship these estimators and their ground-truth
tasks, so §4.1's synthetic case does not need a bespoke implementation.

### 4.6 Reuse existing MI benchmark suites instead of building from scratch

Several maintained benchmark suites already provide exactly the ground-truth-known synthetic
tasks and reference-estimator implementations §4.1/§4.2 call for. Prefer these over hand-rolling
KSG/MINE/InfoNCE and the correlated-Gaussian sweep:

- **[`bmi`](https://github.com/cbg-ethz/bmi) (Benchmarking Mutual Information, CBG-ETH Zürich)**
  — PyPI package `benchmark-mi`, MIT licensed. Ships `KSGEnsembleFirstEstimator`, and
  JAX-implemented `DonskerVaradhanEstimator`, `MINEEstimator`, `InfoNCEEstimator`, `NWJEstimator`,
  plus CCA — i.e. all three baselines in §4.2 plus one extra (NWJ) for free. Ground-truth tasks
  are addressed by name with known MI attached:
  ```python
  import bmi
  task = bmi.benchmark.BENCHMARK_TASKS["1v1-normal-0.75"]
  ground_truth_mi = task.mutual_information
  x, y = task.sample(1000, seed=42)
  ```
  This implements the 40-task suite from Czyż et al. 2023, *"Beyond Normal: On the Evaluation of
  Mutual Information Estimators"* (arXiv:2306.11078) — bivariate and multivariate normal (dense
  and sparse correlation structure), multivariate Student-t at various degrees of freedom, and
  diffeomorphic transforms of each (Gaussian-CDF/uniform-margin, half-cube, asinh, spiral,
  "wiggly" non-uniform-lengthscale mappings) at dimensionality 1×1 up to 25×25, spanning MI up to
  ~2 nats in the main suite and up to 5 nats in a documented high-MI extension. This directly
  supersedes the bespoke synthetic-Gaussian sweep proposed in an earlier draft of §4.1 — reuse
  `bmi`'s tasks (including its sparse/transformed variants, which stress-test estimators harder
  than plain correlated Gaussians) rather than rebuilding a narrower version of the same thing.
  A relevant documented finding from the underlying paper: KSG degrades sharply on the *sparse*
  2-pair-interaction tasks even though it's accurate on dense multivariate normal — worth
  specifically including sparse tasks in our sweep, since that is exactly the regime where a
  flow-based estimator might differentiate itself.
- **[`mibenchmark`](https://github.com/kyungeun-lee/mibenchmark)** (Lee & Rhee, NeurIPS 2024,
  *"A Benchmark Suite for Evaluating Neural Mutual Information Estimators on Unstructured
  Datasets"*, arXiv:2410.10924) — extends ground-truth MI construction to *real* unstructured
  data (MNIST, CIFAR-10/100, IMDB/BERT text embeddings) via same-class positive pairing plus a
  binary-symmetric-channel trick that lets you dial the true MI to a chosen value even on real
  data. Relevant to our RGBMNIST track (§4.1): this gives a way to get a real, non-synthetic
  ground-truth MI benchmark on MNIST-like data, which our current design otherwise lacks (RGBMNIST
  in §4.1 has no ground truth at all today). Worth adopting the BSC-trick construction for
  RGBMNIST specifically, upgrading it from "no ground truth" to "known ground truth."
- **[`mutinfo`](https://github.com/VanessB/mutinfo)** (2025, *"Towards Diverse and Comprehensive
  Benchmarks for Mutual Information Estimation"*, arXiv:2607.03487, CC-BY-4.0) — the most recent
  suite, combining a copula-based synthetic generator (independently controllable MI,
  dimensionality, and marginal complexity) with a "marginals-first" real-image track built on the
  same-class-pairing idea. Its headline finding — *no estimator dominates uniformly; ranking
  flips by task category (non-parametric vs. discriminative vs. generative)* — is directly
  relevant to how we should report results in §4.3/§4.4: report performance broken out by task
  category rather than a single aggregate ranking, since that is the axis the field has converged
  on as the meaningful one.

**Revision to §4.1**: replace the bespoke "new `synthetic_gaussian_Flow` config, sweep ρ
manually" plan with training/evaluating the flow estimator directly on `bmi.benchmark.BENCHMARK_TASKS`
samples (still requires a `synthetic_*_Flow` Hydra config to train a flow on `bmi`-generated
`(x, y)` pairs, but the task definitions, ground-truth MI, and baseline-estimator implementations
come from the package rather than being reimplemented). Similarly, replace the ad hoc
KSG/MINE/InfoNCE implementations referenced in §4.2 with `bmi`'s estimator classes directly.

### 4.3 Success criteria

Comparative, not a fixed absolute threshold: on the synthetic Gaussian sweep, the flow-based
estimator should be **closer to the analytic ground truth than KSG/MINE/InfoNCE at matched
sample budget**, with particular attention to the high-true-MI end of the sweep (ρ ≥ 0.9), where
MINE/InfoNCE are documented to degrade. On dSprites/RGBMNIST (no ground truth), success is
cross-estimator agreement plus passing the sanity checks in §4.4.

### 4.4 Metrics & sanity checks

- Bias vs. true MI (Gaussian only).
- Variance across seeds/bootstrap resamples, at fixed sample budget.
- Sample efficiency: estimate quality as a function of `n` paired samples.
- Wall-clock / compute cost per estimate (the flow estimator pays ODE-integration cost per
  sample — expect this to be more expensive per-point than KSG, cheaper than training MINE/CPC
  from scratch).
- High-MI degradation behavior (does accuracy fall off the way MINE is known to, or differently?).
- Non-negativity: MI ≥ 0 always.
- Permutation null: shuffling the `(x, y)` pairing should collapse the estimate toward 0.
- Negative control on dSprites: y = an independent/irrelevant factor should give MI ≈ 0.

### 4.5 Interpretability evaluation (qualitative)

Plot `∇·v_cond(t) − ∇·v_uncond(t)` against `t` for individual `(x, y)` pairs on dSprites, across
different conditioning factors (e.g. rotation vs. scale), to check whether the time-resolved flux
profile differs meaningfully by factor (e.g. does a "sharper," more localized-in-time signal
correspond to a factor with cleaner generative structure?). This is a case-study/qualitative
track, not a numeric pass/fail benchmark — it supports the interpretability half of the
contribution claim in §7 and has no baseline to compare against by construction.

## 5. Experimental Protocol

- **Training**: reuse existing Hydra CFG-trained flow configs as-is (`rgbmnist_Flow`,
  `dsprites_Flow`, both existing under `src/conf/experiment/`); add one new
  `synthetic_bmi_Flow` config (§4.1/§4.6) for the calibration case.
- **Likelihood validation gate**: run the closed-form-density check from §3 before any MI number
  from this pipeline is trusted.
- **Held-out evaluation set**: MI is estimated on a held-out split, never on training data, to
  avoid optimistic bias from the flow having memorized training pairs.
- **Repeats**: each configuration (dataset × estimator × sample budget) run across multiple
  seeds to report variance, not just a point estimate.

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
