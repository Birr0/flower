# Conditional Flow Matching over a Pretrained Light-Curve Latent: Isolating Residual Stellar Variability in TESS

*Experiment design — feasibility spike*
*Status: draft for discussion. Scope agreed: broad TESS variable catalog · frozen pretrained encoder · staged conditioning (timescales → astrophysical factors) · ablation-based validation.*

This design is split across five files; this README is the entry point. See the outline below for where each section lives.

## Abstract

Stellar light curves encode a superposition of physical processes — rotation, pulsation,
granulation, spot evolution, binarity, flares, instrumental systematics — that unfold across a
wide range of timescales. Modern self-supervised encoders (e.g. StarCLR, Astromer-2) compress a
light curve into a fixed latent vector that is excellent for classification and retrieval, but the
latent entangles *known* astrophysical drivers (effective temperature, characteristic period,
amplitude) with the *residual* structure that those drivers do not explain — which is precisely
where new physics and rare behaviour live.

We propose applying **Flower** — a conditional flow matching (CFM) model over a learned latent
space — to TESS light curves, reusing the exact pattern Flower already implements for galaxy
spectra (`PretrainedSpender` → shared `LightningFlowMatching`). A **frozen pretrained light-curve
encoder** supplies the latent `z`; a CFM model then learns the conditional distribution
`p(z | y)` where `y` is a catalog of known factors of variation. The distinctive contribution is
the **conditioning set built from timescales**: (i) catalogued *physical periods* and (ii) a
*multi-scale decomposition* of each light curve into timescale-band power (wavelet / PSD).
Because Flower's `y_catalog` supports a native `drop_variables` mechanism, we can *condition out*
factors one at a time and measure how the residual latent distribution changes — an
information-theoretic ablation that directly quantifies how much of the encoder's representation is
"explained" by each known factor, and characterises the structure that remains.

This document scopes a **feasibility spike**: verify that a frozen TESS encoder + CFM trains on a
broad TESS variable sample, and that the staged conditioning + `drop_variables` ablation produces a
measurable, interpretable residual signal. It is deliberately lean — one clean pipeline and one
convincing ablation result — before committing to a full-survey study.

## Section outline

**[`related-work.md`](related-work.md)**

- [Related Work & Novelty Positioning](related-work.md#related-work--novelty-positioning)

**[`method.md`](method.md)**

- [1. Background & Notation](method.md#1-background--notation)
- [2. Theory: what "residual structure" means here](method.md#2-theory-what-residual-structure-means-here)
- [3. Implementation Outline](method.md#3-implementation-outline)

**[`evaluation.md`](evaluation.md)**

- [4. Benchmark / Validation Design](evaluation.md#4-benchmark--validation-design)

**[`planning.md`](planning.md)**

- [5. Risks & Open Questions](planning.md#5-risks--open-questions)
- [6. Timeline (feasibility spike)](planning.md#6-timeline-feasibility-spike)
- [Appendix: Mapping to existing Flower code](planning.md#appendix-mapping-to-existing-flower-code)
