↑ [Back to README](README.md)

# Planning

## 5. Risks & Open Questions
- **Label acquisition is now the central bet.** v1 is a *learned* preference model (committed), not a
  frozen rubric, so it lives or dies on getting astronomer labels. Evidence-level scoring (dense,
  credit-assigned) and cluster-level verdicts (one judgment → many signals) are the leverage;
  the elicited-prior warm start is what keeps the system useful *before* labels arrive. **Trigger:** if
  within-session labels stay too few to move `w` off the prior, v1 degrades gracefully to the elicited
  rubric — but the §eval 4.3 rubric-vs-learned ablation is then the honest report.
- **Within-session adaptation is the hard requirement the real-user eval imposes.** A study
  participant gives one sitting, so the model must improve over ~20–40 labels, not thousands. This
  raises the sample-efficiency bar and is *why* the linear part is elicited-warm-started and
  shrinkage-regularised rather than learned from scratch.
- **Interestingness gap (Astronomaly's central point).** Unexplained ≠ interesting. We no longer defer
  this to a "Stage 2 later" — the learned preference model *is* the answer, conditioning out known
  factors (Stage 1) + cross-match artifact demotion + the per-user ranker (Stage 2). Residual risk:
  the preference model needs enough signal to separate artifact from discovery; the confound and
  localisation controls (§eval 4.4) gate any "saves time" claim.
- **Likelihood-OOD pathology.** A single model can rate a true anomaly as typical; marginalising over
  seeds (`S̄`) partially mitigates but does not fully cure it — cross-match evidence + the learned
  ranker are the Stage-2 backstop. The single-model-vs-`S̄` ablation (§eval 4.3) gates the ensemble's
  cost.
- **Ensemble cost (the one new density expense).** Full `K`-seed committed (deliberately, for
  model-independence); MC-dropout/Hutchinson shortcuts rejected. Budget `K×` training accordingly.
- **User-study logistics.** Within-subject, counterbalanced arms need a usable scoring UI, IRB/consent
  where applicable, and enough astronomer-hours for statistical signal from few participants. This is
  now a real project dependency, not just a metric.
- **Held-out class confounds.** Recovery may ride a trivial correlate (SNR, magnitude); §eval 4.4
  regress-out control is mandatory before any "saves researcher time" claim.
- **pmi_flux dependency.** Needs the conditional `log_prob` estimator; comparative ranking across
  objects tolerates estimator bias (constant offset), so it is a softer dependency than in
  [factor-discovery](../agent_factor_discovery/planning.md).
- **Telescope allocation deferred (no live budget).** Rungs 5–6 as *live actions* are out of reach, so
  the cost-aware allocation return (§method 3.1, objective D) is deployment-time future work; v1–v3 is a
  recommender, and the accessible anchor is retrospective literature confirmation.
- **Literature entity-resolution & hallucination.** Object↔paper matching is the failure point — a
  wrong match poisons a high-`π` label. Mitigate: coordinate+name resolution with confidence scores,
  **retrieve-then-read** against ADS/arXiv (never cite from model memory), verify every bibcode/DOI
  resolves, and spot-check high-`π` labels. Handle absence-of-evidence rigorously ("searched well and
  found nothing" ≠ "gave up") before trusting an `uncatalogued` flag.
- **Citation attribution (papers ≠ objects).** Impact accrues to papers, not objects; attributing it
  needs object-centrality weighting and dropping survey/method/catalog papers (§method 3.3).
- **Sociology confound & Goodhart-toward-hype.** Raw citations encode fashion / prestige / benchmark
  reuse; optimising them selects for hype and rediscovery. Mitigate: predict the **conditional,
  field-normalised residual** impact, keep it **low-`π`**, and **gate by `S̄`**.
- **Matthew / anti-novelty bias of the literature.** The densest credible signal is also the most
  conservative — literature rewards the already-known. Use it to **anchor/validate**, not as the
  primary training target, or the model learns to re-find the canon.
- **Time-censoring / survivorship.** Impact labels exist only for objects old enough to have been
  followed up (already well-studied); age-adjust citations and treat them as distribution-shifted from
  fresh anomalies.

## 6. Timeline (~7 weeks; density/config reuse `spender_I_flow` + factor-discovery wiring)
| Phase | Work | Est. |
|---|---|---|
| 0–1 | `log_prob` scorer available; train `K`-seed ensemble on `spender_I_flow`; `S̄` + `U` flag + per-factor localisation in `flower.outliers` | 6–9 d |
| 2 | Cross-match client (SIMBAD/Gaia/SDSS) + agent evidence-record extractor + cluster/object dossiers | 5–7 d |
| 3 | Preference model: elicited-prior linear + residual + hierarchical pooling; online update from labels; persist per-astronomer state | 6–8 d |
| 4 | Scoring UI + within-session loop wiring (evidence scores, holistic verdicts, mid-loop query injection) | 4–6 d |
| 5 | Within-subject real-user study: run arms A/B, adaptation curves, baselines & ablations | 6–9 d |
| 6 | Confound / shuffle / localisation / disagreement controls + write-up | 4–5 d |

## Appendix: Mapping to existing Flower code
| New piece | Reuses / modeled on |
|---|---|
| Conditional density + ensemble | `spender_I_flow` (K seeds); `log_prob` from factor-discovery / pmi_flux |
| `S̄`/`U` scorer + localisation | new `flower.outliers` module (`drop_variables` decomposition) |
| Cross-match enrichment + evidence record | new astroquery client + LLM extractor; post-hoc only |
| Literature agent (labels / features / prior) | `astroquery.nasa_ads` + SIMBAD/NED bibcodes; ADS/arXiv retrieve-then-read; grounded extraction |
| Conditional-impact estimator | attribution → field/year normalisation → residual-on-known-factors; low-`π`, `S̄`-gated |
| Reward fusion + LCB ranking | precision-weighted rung fusion (A); rung-weighted loss (B); lower-confidence-bound (C); allocation (D) deferred |
| Preference model (linear+residual, pooling) | new module; elicited-prior + shrinkage; persists per-astronomer state |
| Agent triage + dossiers (cluster & object) | `flower.discovery.agent` (shared with factor-discovery) |
| Scoring UI + loop | new; drives evidence scores / holistic verdicts / query injection |
| Integration test | add `TestAnomalyTriage` to `tests/test_integration.py` |

### Sources
[Astronomaly](https://ui.adsabs.harvard.edu/abs/2021A&C....3600481L) ·
[WAIC generative ensembles](https://arxiv.org/pdf/1810.01392) ·
[Nalisnick typicality](https://arxiv.org/pdf/1906.02994) ·
[Likelihood Regret](https://arxiv.org/pdf/2003.02977) ·
[NormalcyScore contextual AD](https://github.com/lucabindini/NormalcyScore)
