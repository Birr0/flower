↑ [Back to README](README.md)

# Method

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
