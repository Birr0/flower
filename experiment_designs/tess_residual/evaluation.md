↑ [Back to README](README.md)

# Evaluation

## 4. Benchmark / Validation Design

Validation is **ablation on conditioning** (agreed). Everything below is computed on a held-out
test split.

### 4.1 Primary: conditioning-gain ablation

Using the `drop_variables` sweep, train/evaluate the flow with nested conditioning sets and report
`ΔLL(S)` (§2.1) for each factor and factor group:
- timescales only (`P`, `B`) vs unconditional;
- each astrophysical factor added on top of timescales;
- full `y` vs each single-variable drop (leave-one-out importance).

**Success:** `ΔLL` is positive, statistically resolved (bootstrap CI over test set), and *ordered
sensibly* — e.g. period and amplitude carry large gain; metallicity small. A monotone, interpretable
ordering is the core evidence that the conditional density is capturing real factor→latent
dependence rather than noise.

### 4.2 Secondary: residual-structure characterisation

- **Residual spread map:** per-dimension / PCA spread of `p(z | y_full)` vs `p(z)`; identify the
  directions that stay broad after full conditioning.
- **Residual correlation probe:** correlate the residual coordinates against a factor deliberately
  *not* conditioned on (a held-out `θ`, or a proxy for spot evolution / secondary period). A
  non-trivial correlation shows the residual carries recoverable physics, not just encoder noise.
- **Class colouring (qualitative):** colour a 2-D projection of the residual by the catalog
  variability class; coherent structure = the residual is astrophysically organised.

### 4.3 Baselines / sanity checks

- **Shuffle control:** permute `y` across samples; `ΔLL` must collapse to ≈0 (guards against the
  flow memorising the marginal).
- **Embedding-space baseline:** compare residual-based anomaly ranking against a plain
  distance/OOD score in the raw encoder embedding (à la StarEmbed) to show the conditional model
  adds signal.
- **Encoder swap:** repeat the core ablation with the fallback TSFM encoder to check conclusions are
  not an artefact of one encoder.

### 4.4 What would make the spike a "go" for a full study

A positive, interpretable `ΔLL` ordering (4.1) **and** at least one non-trivial residual signal
(4.2 correlation probe or coherent class structure). If timescale conditioning alone leaves a large,
structured, recoverable residual, the full-paper version (large TESS sample, multi-encoder,
anomaly-recovery benchmark) is justified.
