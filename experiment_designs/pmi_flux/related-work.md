↑ [Back to README](README.md)

# Related work

## Related Work & Novelty Positioning

**This section exists to keep the claims above honest.** The core mechanism proposed here — PMI
as a difference of generative-model log-likelihoods along conditional vs. unconditional pathways
— is not new in the abstract; it has been published, in different concrete forms, at least six
times recently, including with pointwise and decomposed variants, and — with FMMI (below) — in a
flow-matching form by name. Any framing of this work as "we
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
- **Butakov et al., "FMMI: Flow Matching Mutual Information Estimation"** (2025,
  [arXiv:2511.08552](https://arxiv.org/abs/2511.08552)) — same group as MIENF above (Butakov,
  Frolov, Oseledets), and the most direct challenge to this design's framing because it puts *flow
  matching* and MI estimation in the same title. Reframes the **discriminative** (classifier-based,
  MINE/InfoNCE-style) route: instead of training a classifier to tell the joint `p(x,y)` from the
  product of marginals `p(x)p(y)`, it learns a (flow-matching) normalizing flow that *transforms
  one distribution into the other*, and reads MI off that transformation. Claims computational
  efficiency, high precision, and good scaling to high dimension across a wide range of
  ground-truth MI. **Caveat on this entry:** the OpenReview forum and the arXiv PDF body were both
  inaccessible at write time (bot challenge / failed text extraction), so this is written from the
  abstract + author list only — verify the mechanism specifics below against the full text before
  relying on them, exactly as done for MIENF.
  - **What it does *not* appear to change about our edge (point 1 below):** like MIENF, FMMI trains
    a *purpose-built* flow for the `(X, Y)` pair being measured (its target is the joint→marginal
    transport), not a conditional generative model already trained for generation/embedding. So the
    "zero marginal training cost — read off a model flower already trains" argument survives against
    FMMI as it does against MIENF. *Verify*: confirm FMMI is per-pair and non-reusable, since the
    abstract does not state this explicitly.
  - **What it *does* narrow:** "flow matching for MI estimation" is no longer open territory — the
    same competing group now owns that phrase, so our abstract/§7 must not imply flow-matching-based
    MI is novel in itself. Our mechanism stays distinct — a *generative* conditional-vs-unconditional
    log-likelihood difference `PMI = log p(x|y) − log p(x)` from a single CFG-CFM model via the
    probability-flow-ODE divergence (§2.2) — where FMMI is a *discriminative density-ratio* reframing
    via a joint↔marginal flow. A real mechanistic difference, but the phrasing edge is gone.
  - **Verify before any accuracy comparison**: whether FMMI (a) uses continuous-time/ODE flow
    matching vs. discrete coupling layers, (b) evaluates on the `bmi`/Czyż 2023 suite (§4.6) — if so
    it becomes a direct, must-cite baseline for the §4.3 claim — and (c) carries the
    consistency/error-bound guarantees MIENF has (point 4 below), which would extend the honest
    theory gap to this newer method too.
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
