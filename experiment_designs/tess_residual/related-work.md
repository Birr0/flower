↑ [Back to README](README.md)

# Related work

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
