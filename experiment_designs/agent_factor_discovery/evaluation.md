↑ [Back to README](README.md)

# Evaluation

The agreed headline is a **novel-finding case study** on real SDSS spectra. Since residuals have no
clean ground truth, it is credible only if the loop first works where truth *is* known — so a
lightweight **calibration control** gates trust.

## 4.1 Calibration control (run first)
**Planted / held-out factor recovery** — withhold a *known* catalog variable from the training bank
(redshift, `[Fe/H]`, `σ`); check the agent (a) detects a residual direction correlated with it,
(b) names/engineers a proxy, (c) the gate confirms it. Report **precision/recall** over several
withheld variables. Optionally inject a **synthetic** latent factor of known strength and verify
MI-gain is *monotone in strength*. **Gate:** if planted factors are not recovered, no case study.

## 4.2 Headline: novel-finding case study
With the bank *complete* (all known factors in), run on held-out spectra and report whatever residual
factor the agent proposes. Reportable only if it clears all of:
- **MI-gain** positive and bootstrap-resolved on the test set;
- **shuffle control** collapses the gain to ≈0;
- **not redundant** with any bank factor (partial-correlation / conditional-MI check);
- **physically legible** — the named hypothesis survives post-hoc astrophysical scrutiny.

## 4.3 Baselines / sanity checks
- **CAAFE-style baseline** — LLM features scored by a downstream *classifier/regressor*, to show the
  residual gate finds factors a task-loss misses.
- **No-agent ablation** — random / greedy-correlation proposal vs the LLM naming step.
- **Encoder-noise floor** — residual directions on shuffled spectra must yield no nameable factor.

## 4.4 What makes this a "go"
Calibration recovers planted factors (4.1) **and** the case study yields ≥1 finding clearing every
gate in 4.2. A recovered-but-uninteresting result is still a positive methods outcome; 4.1 failure stops.
