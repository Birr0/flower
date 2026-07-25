↑ [Back to README](README.md)

# Planning

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
