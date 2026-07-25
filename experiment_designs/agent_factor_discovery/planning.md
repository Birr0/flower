↑ [Back to README](README.md)

# Planning

## 5. Risks & Open Questions
- **pmi_flux dependency (blocking, #0).** MI-gain needs the CFG `log_prob`/PMI estimator from
  [pmi_flux](../pmi_flux/README.md). Mitigation: a held-out **ΔLL fallback** from the same
  `compute_likelihood` wiring, so the loop is not blocked if pmi_flux slips.
- **Amortization ≠ retraining.** `drop_variables` only drops *trained* factors; a new named feature
  is OOD and needs retrain (§3.3). Risk: too-narrow a bank → many slow retrains; keep it rich and
  track retrain frequency.
- **Residual = encoder noise?** Broad directions may be artefacts — guarded by the shuffle control
  and encoder-noise-floor baseline (§eval 4.3).
- **LLM naming reproducibility.** Correlation is the gate, the LLM only the interpreter, so a wrong
  name cannot pass the MI gate; log all tool-augmented naming.
- **Case-study risk.** The headline may yield no novel factor; the calibration control (§eval 4.1)
  keeps it a demonstrable methods result regardless.

## 6. Timeline (~4–5 weeks, front-loaded on gate + orchestrator; model/training reuse near-free)
| Phase | Work | Est. |
|---|---|---|
| 0–1 | MI gate (or ΔLL fallback validated on a known density) + feature-bank extension to `flower.data.sdss` | 7–11 d |
| 2–3 | `spender_I_discovery` config + tests; `flower.discovery.agent` (propose→gate→correlate→name→human-gate) | 7–10 d |
| 4–5 | Calibration control (planted + synthetic) + case study + baselines/ablations + write-up | 8–11 d |

## Appendix: Mapping to existing Flower code
| New piece | Reuses / modeled on |
|---|---|
| Amortized CFG model | `flower.models.spectra.LightningFlowMatching` (unchanged) |
| Feature bank | `flower.data.sdss.SDSS` + `y_catalog` (extended) |
| `drop_variables` subset search | `spender_I_flow` `sweeps.yaml` / CFG `null_y` at eval |
| MI-gain / ΔLL gate | pmi_flux `log_prob` (`ODESolver.compute_likelihood`) |
| Integration test | add `TestSpenderDiscoveryConfig` to `tests/test_integration.py` |

### Sources
[CAAFE](https://arxiv.org/abs/2305.03403) ·
[Automation→Autonomy survey](https://github.com/HKUST-KnowComp/Awesome-LLM-Scientific-Discovery) ·
[Agentic-scientists critique](https://arxiv.org/html/2605.08956v1)
