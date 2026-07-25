# Geometric Estimation of Mutual Information via Conditional Flow Matching

Status: design draft, not yet implemented. No code referenced here exists yet unless explicitly
marked "(existing)".

This design is split across five files; this README is the entry point. See the outline below for where each section lives.

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

## Section outline

**[`related-work.md`](related-work.md)**

- [Related Work & Novelty Positioning](related-work.md#related-work--novelty-positioning)

**[`method.md`](method.md)**

- [1. Background & Notation](method.md#1-background--notation)
- [2. Theory](method.md#2-theory)
- [3. Implementation Prerequisite: Log-Likelihood](method.md#3-implementation-prerequisite-log-likelihood)

**[`evaluation.md`](evaluation.md)**

- [4. Benchmark Design](evaluation.md#4-benchmark-design)
- [5. Experimental Protocol](evaluation.md#5-experimental-protocol)

**[`planning.md`](planning.md)**

- [6. Limitations](planning.md#6-limitations)
- [Causal Interpretation (and Its Limits)](planning.md#causal-interpretation-and-its-limits)
- [7. Conclusion & Positioning](planning.md#7-conclusion--positioning)
- [8. Timeline / Effort Estimate](planning.md#8-timeline--effort-estimate)
