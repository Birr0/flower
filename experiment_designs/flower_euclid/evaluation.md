↑ [Back to README](README.md)

# Evaluation

## 6. Evaluation & validation

An unusual embedding is not self-evidently *interesting*. Validation must rule out two null
explanations for a high anomaly score before it counts as beyond-morphology science: **(i)** it is an
imaging artifact (§7's contamination risk), and **(ii)** it is merely a rare morphology the score
S1 over-weights (§5.1). The battery below combines expert inspection with the four independent
automated signals selected earlier, and evaluates **both S1 and S2**.

### 6.1 Expert inspection (primary human validation)

For each score (S1, S2) and each ablation run (A0, A1), draw the top-`N` anomalies and have an expert
blind-label each cutout (`euclid_q1` imaging) into: `artifact` · `known-rare-type`
(merger/lens/interacting) · `genuinely-novel` · `uninteresting/mis-segmented`. This is the arbiter of
"scientifically interesting" the user identified as an expert step. The automated signals below exist
to *corroborate and scale* this, not replace it.

### 6.2 Automated verification battery (ordered by Q1 availability)

All four signals selected; ordered most- to least-immediately-available for the Q1 footprint.

1. **Physical-property enrichment (MER join, immediate).** Via the `object_id`+`tile_index` MER join
   (§4), attach photometry and derived quantities (colours, photo-z, stellar mass, SFR where
   released). Two tests: **(a) enrichment** — are anomalies over-represented in physically extreme
   regions (unusual colour, high/low sSFR)? **(b) residual-predictivity probe** — can a simple probe
   predict a physical property from the *residual* better than from morphology alone? A positive gap
   is direct evidence the residual carries non-morphological physics. **[TO VERIFY]** which derived
   properties (mass/SFR/photo-z) are in the Q1 MER release vs. external.
2. **Rare-object recall (catalogue-based).** Recall@K of the anomaly ranking against
   [`euclid_strong_lens_expert_judges`](https://hf.co/datasets/mwalmsley/euclid_strong_lens_expert_judges)
   (strong lenses) and published Q1 merger catalogues (Euclid Collab.: La Marca et al. 2025, cited by
   the morphology paper). Lenses and mergers are morphologically distinctive, so this partly probes
   whether the residual retains *known* rare classes — a sanity floor, not the novelty claim.
3. **Multi-wavelength counterparts.** Cross-match the anomaly set with X-ray / radio / optical-spectra
   catalogues overlapping the Q1 fields; test enrichment in independently-flagged AGN, starbursts,
   etc. **[TO VERIFY]** overlap footprint and depth for Q1 (may be thin).
4. **Spectroscopic outliers.** Where spectra exist for anomalies, test whether image-embedding
   anomalies are *also* spectroscopic/emission-line outliers — the strongest independent-modality
   confirmation, but likely the smallest sample in Q1. **[TO VERIFY]** spectroscopic coverage.

### 6.3 Positive controls

- **Artifact recovery (A1, from §5.4).** Dropping `artifact`/`problem` from the conditioning **must**
  push saturated stars / ghosts into the anomaly tail. This is a pass/fail check that the score
  tracks structure the conditioning omits; failure invalidates the isolation mechanism.
- **Beyond-GZ feature recovery.** A non-deliverable sanity check (§7): does the A0 tail recover the
  *kinds* of beyond-decision-tree features Wu & Walmsley list (dust lanes in edge-on disks, ellipticals
  with bluer companions)? Recovering them supports the mechanism; going *past* them (novel classes
  they did not name) is where the marginal contribution over the SAE paper would actually be
  demonstrated.

### 6.4 Success criteria

1. **Mechanism holds:** A1 recovers artifacts (6.3); S1/S2 tails are not explained purely by
   morphology class (S2 in particular surfaces within-class outliers).
2. **Signal above chance:** anomaly ranking shows statistically significant enrichment / recall over a
   random-ranking baseline in ≥2 of the four automated signals, for both S1 and S2.
3. **Beyond morphology, concretely:** the residual-predictivity probe (6.2.1b) beats
   morphology-alone for ≥1 physical property, and expert inspection finds a non-trivial
   `genuinely-novel` fraction in the A0 tail after artifacts are conditioned away.
4. **Beyond the SAE paper (stretch):** at least one anomaly class not enumerated by Wu & Walmsley,
   with a plausible physical interpretation.
