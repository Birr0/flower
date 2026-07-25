↑ [Back to README](README.md)

# Method

The system is **two-stage**: a density model supplies *recall* (what is unusual given known
factors), and a learned preference model supplies *precision* (what is worth a researcher's time).
Keeping them separate preserves the model-independence of the anomaly signal while letting the
subjective "interesting" signal be learned and inspected.

## Stage 1 — recall: model-marginalised surprisal

Rank on a **K-seed ensemble** so an anomaly is a property of the data given `y`, not of one model's fit:
- **Score** `S̄(x) = mean_k[ −log p_k(z | y) ]` — mean conditional surprisal across `K` independently
  seeded CFMs (each via pmi_flux `log_prob`). An object enters the pool only if *all* seeds find it
  unlikely given known factors. **This is the sole recall/candidate-selection signal** (no
  expected-information-gain term; the earlier `S + EIG` acquisition is dropped — interestingness now
  lives entirely in Stage 2). Take the top-`k` as the candidate pool.
- **Confidence flag** `U(x) = Var_k[ −log p_k(z | y) ]` — seed disagreement. **Not** a ranking term;
  reported per candidate as an agreement tier ("all seeds agree" vs "seeds disagree — caution"). Full
  `K`-seed is deliberate to keep anomalies model-independent; MC-dropout / Hutchinson shortcuts are
  rejected for this reason.

### 1.1 Interpretable localisation
Reuse `drop_variables`: for each candidate, compare `−log p(z | y_full)` against `−log p(z | y_{−f})`
per factor `f`. The factor whose presence least explains the object (largest residual tail) is the
**violated factor** — a human-readable reason ("anomalous *given* its redshift and `[Fe/H]`") and a
feature the Stage-2 model consumes.

## Stage 2 — precision: evidence + a learned preference model

### 2.1 Agent evidence-gathering
For each candidate the agent **autonomously decides and runs** cross-match queries
(SIMBAD/Gaia/SDSS); the astronomer may **inject a query mid-loop** ("also pull the Gaia parallax").
The agent then **extracts the heterogeneous raw responses into a fixed structured evidence record** —
the LLM absorbs all the catalog-format ugliness so the downstream model sees a stable, low-dimensional
vector. Illustrative schema (each field chosen to be informative about *interestingness*, with
mutually-exclusive pairs respected — e.g. `catalog_conflict` requires a match, so it cannot co-occur
with `uncatalogued`):

| Field | Type | Signal |
|---|---|---|
| `match_status` | categorical: uncatalogued / known-artifact / known-class / ambiguous | uncatalogued or ambiguous ↑; known-artifact ↓ |
| `primary_otype` | SIMBAD type over a controlled vocab (one-hot / small learned embedding) | "this *kind* of object interests me" |
| `catalog_conflict` | scalar: spectrum's implied class vs. catalogued class | mismatch is a strong genuine-discovery signal |
| `n_matches`, `nearest_sep_arcsec` | scalars | isolation / crowding context |
| `corroborating_flags` | bits: variability, multiwavelength counterpart, PM/parallax SNR | positive evidence of real interest |
| `violated_factor` | vector (from §1.1) | *which* known factor it breaks, and how hard |
| `S̄`, `U` | scalars | carry the density signal + confidence tier forward |

A free-text embedding of the agent's dossier can be appended as an extra feature block later; the
structured record is what makes the model trainable from few labels, so it is the starting point.

### 2.2 The preference model — interpretable linear part + learned residual
Re-rank the pool by an interestingness score `r(evidence)`:

```
r(evidence) = wᵀ·evidence  +  f_residual(evidence)
              └── linear ──┘   └──── residual ────┘
```

- **Linear part `wᵀ·evidence`.** The astronomer's **elicited per-field scores are the prior mean of
  `w`** (e.g. "catalog_conflict ≈ +2, known_artifact ≈ −3"). `w` is then **fit to holistic verdicts
  but regularised toward that prior**, with the pull strong when data is scarce and weakening as
  labels accumulate. Consequence: with 0 labels `r` reduces to the astronomer's stated rubric (cold
  start / day-one behaviour); with many labels `w` is data-driven and the rubric was just the warm
  start. This is the standard prior-plus-shrinkage move — it is also what removes the drift risk of
  learning weights from scratch on a handful of labels.
- **Residual `f_residual`.** A small nonlinear model that predicts *only the gap* between the linear
  base and the astronomer's true judgment — i.e. the **interaction/conjunction effects** a weighted
  sum cannot represent ("catalogued-ordinary **and** spectrum-conflict **and** violated-redshift is
  worth far more than the sum of its parts"; interactions can also be *suppressive* — a known artifact
  should crush a high `S̄`). It is the only component that genuinely needs training data.

**Labels (two kinds, mapped to the two parts):** *per-field evidence scores* — the astronomer scores
the individual evidence fields for a candidate (dense, higher-information, solves credit assignment,
less subjective than a single verdict) → shape the **linear** part; *holistic verdicts* — occasional
"this whole object/cluster is a yes/no" → train the **residual**. Evidence-scoring is the primary
interaction; holistic verdicts supplement.

### 2.3 Per-astronomer policies, pooled to consensus
Each astronomer has their **own** `w` (and residual): their taste. Pool via **partial pooling
(hierarchical)** — each person's weights are drawn from a shared population prior, so users with few
labels borrow strength from the group instead of overfitting (same machinery as §2.2, one level up).
**Inter-astronomer variance is a first-class output**: low-variance objects are *objectively* odd;
high-variance objects are taste-dependent — the human-level twin of the model's seed disagreement `U`.

### 2.4 Cluster and object reasoning
The agent reasons at **both** levels. **Clusters** give strength-in-numbers evidence for a *new class*
(and let an astronomer label a whole cluster at once — cheap, high-yield supervision). **Objects**
are not neglected: singletons may carry the most novel signal. Cross-match is used **symmetrically** —
demote known artifacts, promote corroborated genuine oddities — and the agent emits a short **dossier**
at both granularities.

### 2.5 The loop
Rank by `S̄` → agent gathers evidence + builds dossiers → astronomer scores evidence / gives holistic
verdicts / injects queries → preference model updates → pool re-ranks the remaining candidates. The
astronomer's feedback both **steers what is investigated** and **trains the ranker** within a session.

## 3. Reward hierarchy: from surprisal to published impact

The signals that tell us "this was worth it" form a **ladder** of escalating cost, credibility, and
latency — cheap/abundant/weak at the bottom, expensive/rare/authoritative at the top. We **train on
the cheap-dense rungs and anchor/calibrate to the expensive-sparse ones**; each higher rung is a check
against Goodharting the one below.

| # | Signal (commitment) | Quality | Role |
|---|---|---|---|
| 1 | conditional surprisal `S̄` | weakest (unusual ≠ interesting) | generate |
| 2 | agent cross-match corroboration | weak self-supervised | enrich |
| 3 | astronomer evidence scores (per-field) | *stated*, dense | train (linear) |
| 4 | astronomer holistic verdict | stated, considered | train (residual) |
| 5 | telescope follow-up allocation | *revealed* | **deferred — no access** |
| 6 | follow-up outcome / **literature confirmation** | near-ground-truth | anchor |
| 7 | paper written / **citation impact** | revealed, graded | validate |
| 8–9 | peer review / citations / new class named | community / field | external validation |

**We do not hold telescope time** (rungs 5–6 as *live actions*), so the live budgeted-allocation
objective is **deferred to deployment** — the system is a recommender, not an allocator. The accessible
anchor is **retrospective literature confirmation** (§3.2), which is also the *densest* credible rung
available to us.

### 3.1 Reward label, model, conservatism
Put every rung on a common **expected-discovery-value** scale `V` (log-odds of a genuine, publishable
discovery). For object `x` with a sparse set of observed rung signals `s_i(x)`, fuse by **precision
(credibility) weight** `π_i ∝ 1/σ_i²` — tiny for `S̄`, large for confirmation, so the highest observed
rung dominates and cheap rungs only fill in when nothing better exists:

```
y(x) = Σ_{i∈obs} π_i·s_i(x) / Σ_{i∈obs} π_i                        (A) label fusion
```

The **reward model** (the ranker) is the two-part preference model of §2.2, fit to `y` with a
rung-weighted loss and the elicited prior:

```
r̂(x) = wᵀ·φ(x) + f_resid(φ(x))                                    (B) reward model
L(θ) = Σ_x π_best(x)·(r̂(x) − y(x))² + γ‖w − w_prior‖² + (pool §2.3)
```

Rank on a **lower-confidence bound** so we distrust `r̂` away from any credible label — the
δ / KL / prior-shrinkage analog, here derived from the hierarchy:

```
r̃(x) = r̂(x) − β·σ(x)                                             (C) conservatism
```

The **cost-aware return** — allocate scarce follow-up by `priority(x) = r̃(x)/ĉ(x)` under a budget — is
written down but **deferred (D)**; it activates only once a real follow-up budget exists.

### 3.2 Literature grounding
The evidence agent (§2.1) extends from catalogs to prose: SIMBAD returns **bibcodes per object** — the
bridge into the literature (`astroquery.simbad`/`ned` → **NASA ADS** `astroquery.nasa_ads` → arXiv/OA
full-text, tiered to control cost). It serves the hierarchy in **three roles**:
- **Label** (rung-6 anchor) — a grounded per-object verdict
  `confirmed-discovery / known-artifact / mundane-known / uncatalogued-silence`, every claim tied to a
  real bibcode.
- **Feature** — the same reading distilled into `φ(x)` fields (prior explanation, paper count/recency/tone).
- **Prior** — a population "publishability" model of *what kinds* of objects the field writes up
  (rung-7 mined at scale), informing `w_prior`.

**Discipline:** literature rewards the already-known (Matthew bias), so it is an **anchor/validation
signal, not a naive training target** — optimising "gets cited" directly selects for rediscovery and
fashion, the opposite of the goal.

### 3.3 Conditional literature impact (a graded rung-7 label)
Citation impact densifies the binary confirmation into a *continuous* reward — but **raw** counts are
contaminated: papers ≠ objects, and counts encode fashion / prestige / benchmark-reuse and
time-censoring, not object-interest. We therefore predict **conditional, field-relative** impact,
mirroring Flower's own conditional-density thesis:
- **Attribute** — weight each object by its centrality to a paper (primary subject vs. one table row).
- **Normalise** — field- and year-relative citations (RCR-style percentile), drop survey/method/catalog
  papers, deweight self-citation, age-adjust the censoring.
- **Condition** — regress out the object's *known catalog factors* and predict the **residual**: "cited
  more than a typical object with these properties would be" — impact as *surprise*, not popularity.

Used as a **low-`π` auxiliary label** (regularises / breaks ties; never dominates rungs 3–4 or 6) and
**gated by surprisal**: the target frontier is **high conditional-impact `AND` high `S̄`** — evidence
signatures resembling past high-impact discoveries that are *novel in the current data*. That
conjunction is how past impact finds the *new* rather than re-finding the canon.

## 4. Wiring into Flower
- **Density** — reuse the amortized conditional CFM and `log_prob` from the
  [factor-discovery design](../agent_factor_discovery/method.md); frozen `spender` encoder,
  `spender_I_flow` config. Ensemble = `K` seeds of the same training (the one genuinely new density
  compute cost).
- **Stage-1 scorer** — new `flower.outliers` module computing `S̄`, the `U` flag, and per-factor
  localisation (`drop_variables` decomposition).
- **Stage-2** — a cross-match client (astroquery) for autonomous enrichment; a **literature agent**
  (`astroquery.nasa_ads` + SIMBAD/NED bibcodes → grounded extraction) supplying rung-6 labels,
  `φ(x)` features, and the publishability prior; an evidence-record extractor in
  `flower.discovery.agent`; a small preference-model module (elicited-prior linear + residual,
  hierarchical pooling) that persists per-astronomer state; a **conditional-impact estimator**
  (attribution → field/year normalisation → residual-on-known-factors), used as a low-`π`,
  `S̄`-gated auxiliary reward.
- **No retrain per object** — density scoring is forward-pass only over the trained `K`-seed ensemble;
  only the lightweight preference model updates online from astronomer labels.
