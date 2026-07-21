# Conditional Flow Matching over a Pretrained Light-Curve Latent: Isolating Residual Stellar Variability in TESS

*Experiment design — feasibility spike*
*Status: draft for discussion. Scope agreed: broad TESS variable catalog · frozen pretrained encoder · staged conditioning (timescales → astrophysical factors) · ablation-based validation.*

---

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

---

## Related Work & Novelty Positioning

### Pretrained light-curve encoders (the frozen `z`)

The last two years produced several viable frozen encoders that fit Flower's encoder contract
(`encode(X) -> {"z": ...}`), analogous to how `PretrainedSpender` wraps a frozen `spender` galaxy
autoencoder:

| Model | Pretraining | Output | Weights | Fit for the spike |
|---|---|---|---|---|
| **StarCLR** ([arXiv:2604.24516](https://arxiv.org/abs/2604.24516), ApJ 2026) | Contrastive, **TESS-native**; positive pairs from overlapping sub-sequences | Single projection/embedding vector | [Inference repo](https://github.com/dj-y/StarCLR-Inference) + [Zenodo weights](https://doi.org/10.5281/zenodo.19728042) | **Primary.** TESS-native, single-vector output — closest analog to `spender`. |
| **Astromer-2** ([arXiv:2502.02717](https://arxiv.org/html/2502.02717)) | Masked-modeling transformer on MACHO/ATLAS single-band | Per-element embeddings (pool to vector) | [astromer-science](https://github.com/astromer-science) (TF; PyPI) | **Fallback.** Not TESS-native, TensorFlow → integration friction. |
| **General TSFMs** (Chronos, MOIRAI) | Generic large-scale time series | Token embeddings (pool) | HF | **Zero-shot baseline** — [StarEmbed](https://arxiv.org/pdf/2510.06200) benchmarks these on variable stars. |

### Conditional / generative modeling of the latent (the novel part)

- **AstroM³** ([arXiv:2411.08842](https://arxiv.org/abs/2411.08842)) — CLIP-style multimodal
  embeddings; does anomaly detection and similarity search in *embedding space*, but no conditional
  generative model and no explicit conditioning-out of factors.
- **StarEmbed** ([arXiv:2510.06200](https://arxiv.org/pdf/2510.06200)) — benchmarks TSFM embeddings
  for clustering / classification / OOD detection. Establishes that these latents are informative;
  does not model `p(z | y)`.
- **Astronomical anomaly / OOD** work generally operates on distances or reconstruction error in a
  static embedding, not on a *conditional density* that can be marginalised or ablated.
- **Flower's own spectra pathway** (`spender_I_flow`, `spender_II_flow`) — CFM over a frozen
  spectral latent conditioned on redshift. This is the direct methodological template; it has not
  been applied to time-domain photometry, nor with a *timescale-structured* conditioning set.

### Where the novelty is

To our knowledge, no prior work learns a **conditional flow / density over a pretrained
light-curve latent** in order to (a) *condition out* known astrophysical and timescale factors and
(b) characterise the **residual variability structure** that remains. The specific combination —
frozen TESS encoder + CFM + a conditioning catalog whose leading variables are *physical period*
and *multi-scale band-power* + a `drop_variables` ablation — is new. The contribution is
methodological (residual-structure modeling for time-domain astronomy) and, if the spike succeeds,
a reusable diagnostic: "how much of a light-curve foundation-model embedding is explained by each
known factor of variation?"

*Novelty verdict: passes.* The pieces exist individually; their composition for residual
variability does not. Even if a residual-density approach were later found in an unpublished form,
the timescale-decomposition conditioning and the Flower-native ablation remain a distinct,
defensible angle.

---

## 1. Background & Notation

A TESS light curve is a sampled flux series `X = {(t_i, f_i, σ_i)}` over one or more sectors
(~27 d each, 2-min or FFI cadence, with gaps at momentum dumps and downlinks). A frozen encoder
`E` maps `X` to a latent `z = E(X) ∈ R^d`.

The conditioning vector `y` is assembled from a **catalog** describing each variable — its `size`,
whether it is `continuous`, and which entries are in `drop_variables`. Flower computes the
effective conditional dimensionality from this catalog
(`get_conditional_len`, `get_no_of_continuous_variables` in `flower.models.modules`), so the
science lives in the *catalog design*, not the model code.

We distinguish:
- **Physical periods** `P` — catalogued characteristic periods (rotation, pulsation, orbital),
  entered as continuous conditioning (log-period, plus optional sin/cos phase encoding to respect
  periodicity and to be robust to alias/harmonic ambiguity).
- **Multi-scale band-power** `B = (B_1, …, B_K)` — the light curve's power in `K` timescale bands
  from a wavelet or PSD decomposition (e.g. octave-spaced bands from cadence to sector length),
  normalised per-curve. This gives the model a *timescale fingerprint* independent of any single
  catalogued period, and is well-defined even for quasi-periodic / stochastic variables where a
  single `P` is meaningless.
- **Astrophysical factors** `θ = (T_eff, log g, [Fe/H], R⋆, TESS mag, colour, RMS amplitude)` —
  from TIC / Gaia cross-match. Staged in *after* the timescale-only spike.

The residual object of interest is the conditional distribution `p(z | y)` and, in ablation, the
*change* in that distribution as variables are added to `drop_variables`.

---

## 2. Theory: what "residual structure" means here

Flower's CFM learns a velocity field transporting a conditional prior to `p(z | y)`. Two
quantities make the residual notion concrete:

### 2.1 Conditional density and the explained-variance analog

For a trained model we can evaluate (or bound, via the flow's likelihood) `log p(z | y)`. Define
the **conditioning gain** of a variable set `S` as the improvement in held-out latent
log-likelihood when `S` is included versus dropped:

```
ΔLL(S) = E_test [ log p(z | y_full) − log p(z | y_drop(S)) ]
```

`ΔLL(S) ≥ 0` (up to estimation noise) measures how much of the latent's variation is statistically
explained by `S`. This is the density-model analog of an explained-variance / conditional-mutual-
information term and is the primary ablation metric.

### 2.2 The residual as a displacement in latent space

Conditioning the flow on `y` reshapes where probability mass sits. The residual structure is what
the conditional prior `p(z | y)` *cannot collapse*: directions in `z` along which the conditional
distribution stays broad even after all known `y` are supplied. Concretely, after training we can
compare the per-dimension (or PCA-of-residual) spread of samples from `p(z | y)` against the
marginal `p(z)`; dimensions that remain broad carry structure unexplained by known factors — the
candidate home of leftover physics (spot lifetime, differential rotation, activity cycles,
multi-periodicity) and of genuine anomalies.

### 2.3 Timescales as the interesting axis

Because the leading conditioning variables *are* timescales (`P`, `B`), the ablation answers a
physically framed question: **once we tell the model the star's periods and its power-per-timescale,
how much latent structure is left, and what is it correlated with?** A large residual after
timescale conditioning that then shrinks under `θ` conditioning localises the leftover structure to
astrophysical parameters; a residual that persists under *everything* is the anomaly / new-physics
signal.

---

## 3. Implementation Outline

The spike reuses Flower's existing machinery almost verbatim; the new code is an encoder wrapper, a
data module, and a catalog.

### 3.1 Frozen encoder (`flower.models.lightcurve`)

Mirror `flower.models.spectra`:

```python
class LightningFlowMatching(LightningFlowMatchingBase):
    pass

class PretrainedLightCurveEncoder(nn.Module):
    def __init__(self, model, latent_dim):
        super().__init__()
        self.model = load_starclr(model)          # frozen
        self.latent_dim = latent_dim
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode(self, X):
        return {"z": self.model.embed(X)}          # {z: ...} contract; no mu/logvar
```

*Integration risk to retire first:* StarCLR ships as an inference repo + Zenodo weights; confirm it
loads in-process and produces a deterministic fixed-`d` vector from a padded/masked TESS segment. If
the framework mismatch (TF) or preprocessing coupling is too costly, fall back to a TSFM
(Chronos/MOIRAI) embedding, which StarEmbed shows is serviceable zero-shot. **This load-and-embed
check is the first day's work and gates everything else.**

### 3.2 Data module (`flower.data.tess`)

Follow `flower.data.sdss.SDSS` structure (HF `datasets`, train/val/test split, per-source `Dataset`
wrapped by `FlowerDataset`/`FlowerDataLoader`). Responsibilities:
- Fetch the **broad TESS variable catalog**. *Pinned choice:* the **Fetherolf et al. 2023
  Variability Catalog of Stars Observed during the TESS Prime Mission**
  ([ApJS, 10.3847/1538-4365/acdee5](https://iopscience.iop.org/article/10.3847/1538-4365/acdee5)),
  hosted as the [TESS-SVC HLSP on MAST](https://archive.stsci.edu/hlsp/tess-svc). This single
  product supplies three conditioning ingredients directly: periodogram-derived **periods** (`P`),
  per-star **variability amplitudes** (`rms_amp`), and **TICv8 stellar parameters** (T_eff, radius,
  luminosity → log g) matchable by **TIC or Gaia ID**. Optionally augment with **Gaia DR3**
  variability classes (12.4M stars, 22 types) for the class-coloured residual diagnostic (§4.2).
  *Rejected alternatives:* TARS (1M rotation periods within 500 pc) — excellent but rotator-only,
  too narrow for the broad scope; single-class catalogs (δ Scuti, sdB) — too narrow. Light curves
  come from TESS 2-min SPOC (or FFI for fainter targets).
- Standardise each light curve to the encoder's expected input (segment length, cadence handling,
  normalisation, gap/skyline-style masking — cf. SDSS `get_skyline_mask`).
- Compute the **multi-scale band-power** `B` per curve (wavelet or Lomb–Scargle/PSD; store as
  features so it is cheap at train time).
- Attach the `y_catalog`.

### 3.3 The `y_catalog` (staged)

Stage A (spike core — timescales only):
```yaml
y_catalog:
  variables:
    log_period:      {size: 1, continuous: 1}
    period_phase:    {size: 2, continuous: 2}   # sin/cos, optional
    band_power:      {size: K, continuous: K}   # multi-scale decomposition
  drop_variables: []
```
Stage B (add astrophysical factors):
```yaml
    teff:      {size: 1, continuous: 1}
    logg:      {size: 1, continuous: 1}
    feh:       {size: 1, continuous: 1}
    radius:    {size: 1, continuous: 1}
    tess_mag:  {size: 1, continuous: 1}
    rms_amp:   {size: 1, continuous: 1}
```

`n_layers` for the velocity field is set in `model.yaml`, not hardcoded (per repo contract).

### 3.4 Config tree

Add `src/conf/experiment/tess_flow/` composing `meta.yaml` (data_name `tess`, paths),
`model.yaml` (`_target_: flower.models.lightcurve.LightningFlowMatching`, `catalog: ${data.y_catalog}`,
`n_layers`), `train.yaml` / `embed.yaml`, and `sweeps.yaml` whose `drop_variables` list drives the
ablation — exactly as `spender_I_flow` sweeps `drop_variables` today.

### 3.5 Tests

Extend `tests/test_integration.py` with a `TestTessFlowConfig` that composes the config on CPU with
a tiny batch and instantiates the `lightning_loader` graph (catches `_target_` / catalog drift).
Add a unit test that the encoder wrapper returns `{"z": tensor}` of shape `(B, d)`.

---

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

---

## 5. Risks & Open Questions

- **Encoder input coupling** — StarCLR's preprocessing/segmenting may be tightly bound to its
  training pipeline; the frozen-embed check (§3.1) must pass early or we switch to a TSFM.
- **Period-catalog quality / aliasing** — catalogued `P` carries harmonic/alias ambiguity;
  sin/cos phase encoding and band-power `B` reduce reliance on a single exact period.
- **Heterogeneity of a broad catalog** — mixing aperiodic and periodic classes weakens `P` for some
  stars; `B` (defined for all) is the safeguard, and class-coloured diagnostics will reveal if the
  flow needs per-class treatment later.
- **Contradictory-answer reconciliation** — conditioning was scoped as "all factors" *and* "just
  timescales for now"; resolved as staged (A→B). Confirm before Stage B.
- **Likelihood estimation** — CFM likelihoods require an ODE/trace estimate; reuse whatever
  likelihood path the spectra pathway already relies on, and treat `ΔLL` comparatively (differences,
  not absolute nats) to cancel estimator bias.

---

## 6. Timeline (feasibility spike)

| Phase | Work | Est. |
|---|---|---|
| 0 | **Encoder load-and-embed gate:** StarCLR (or TSFM fallback) produces deterministic fixed-`d` vectors from TESS segments in-process | 2–3 days |
| 1 | `flower.data.tess` module: fetch broad variable catalog + TIC/Gaia cross-match, preprocessing, band-power `B`, `y_catalog` | 4–6 days |
| 2 | `flower.models.lightcurve` wrapper + `tess_flow` config tree + integration/unit tests | 2–3 days |
| 3 | Train Stage-A (timescales-only) CFM; smoke-test on a few sectors | 2–3 days |
| 4 | `drop_variables` ablation harness + `ΔLL` computation + shuffle/baseline controls | 3–4 days |
| 5 | Stage-B (add `θ`), residual-structure diagnostics, go/no-go write-up | 4–5 days |

**Total: ~3–4 weeks** for a decision-quality spike, front-loaded on the two real risks (encoder
integration, data/catalog assembly). Model and training code are near-free by reuse of the existing
shared flow.

---

## Appendix: Mapping to existing Flower code

| New piece | Modeled on |
|---|---|
| `PretrainedLightCurveEncoder` | `flower.models.spectra.PretrainedSpender` |
| `flower.models.lightcurve.LightningFlowMatching` | `flower.models.spectra.LightningFlowMatching` (thin subclass) |
| `flower.data.tess.TESS` | `flower.data.sdss.SDSS` |
| `tess_flow` config tree | `src/conf/experiment/spender_I_flow/` |
| `drop_variables` ablation | `spender_I_flow` `sweeps.yaml` / `train.yaml` drop-variable sweep |
| Integration test | `tests/test_integration.py::TestDspritesFlowConfig` |

---

### Sources

- StarCLR — [arXiv:2604.24516](https://arxiv.org/abs/2604.24516) · [inference repo](https://github.com/dj-y/StarCLR-Inference) · [Zenodo weights](https://doi.org/10.5281/zenodo.19728042)
- Astromer-2 — [arXiv:2502.02717](https://arxiv.org/html/2502.02717) · [astromer-science](https://github.com/astromer-science)
- StarEmbed (TSFM benchmark on variable stars) — [arXiv:2510.06200](https://arxiv.org/pdf/2510.06200)
- AstroM³ — [arXiv:2411.08842](https://arxiv.org/abs/2411.08842)
- TESS variability catalog (Fetherolf et al. 2023) — [ApJS 10.3847/1538-4365/acdee5](https://iopscience.iop.org/article/10.3847/1538-4365/acdee5) · [TESS-SVC HLSP on MAST](https://archive.stsci.edu/hlsp/tess-svc)
