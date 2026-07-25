↑ [Back to README](README.md)

# Planning

## 8. Implementation plan

**Principle: reuse-first.** This experiment should land as *a new experiment config + a data
materialisation step + a `y_catalog`*, reusing the existing frozen-encoder + CFM + CFG machinery.
The one capability it genuinely needs but does not yet have — per-question conditioning masking — is
deferred to an existing GitHub issue (below), and Phase 1 ships without it.

### 8.1 New pieces (minimal)

- **Experiment config tree** `src/conf/experiment/euclid_Flow/` mirroring `spender_I_flow`:
  `meta.yaml` (paths/identity, `data_name`), `model.yaml` (`lightning_loader` + callbacks),
  `train.yaml` / `embed.yaml` defaults lists, `sweeps.yaml`. `_target_` for the flow reuses
  `flower.models.modules.LightningFlowMatching` (or a thin `flower.models.euclid` subclass mirroring
  `flower.models.spectra`'s one-line inheritance).
- **Data materialisation script** (not a new `Dataset` class): cross-match `euclid_q1_embeddings`
  (`pooled_features_block_11` → `z`, §3) with `gz_euclid` (morphology `_fraction` columns, §4) on
  `id_str`, write one on-disk HF dataset of rows `{z: <384-d>, <question_answer columns...>}`. Loaded
  at train time by the existing `flower.data.embeddings.FlowerEmbeddingDataset` (existing).
- **`y_catalog`** listing each morphology `question-euclid_answer` as `size=1, continuous=True` (§4),
  consumed by `get_conditional_len` / `get_no_of_continuous_variables` (existing).

Because embeddings are **precomputed** (§3 Option A), no base-encoder forward pass runs at train
time — `z` is a column — so this is even lighter than `spender_*_flow` (which runs the frozen encoder
live). The frozen-MAE wrapper is only needed if we take §3 Option B.

### 8.2 GitHub dependencies

| Issue | What it gives | This project's reliance |
| ----- | ------------- | ----------------------- |
| [#8](https://github.com/Birr0/flower/issues/8) *More expressive condition embedder* | item 1: **variable-level masking (null variables per conditioning part)**; item 2: per-variable projections + summation | **Blocking for the clean design.** §5.3 feature-set MIs and correct §4 tree-NaN handling both need per-question null-masking. This project *relies on that PR*. Item 2 also better suits the ~40-way continuous conditioning than raw concatenation. |
| [#10](https://github.com/Birr0/flower/issues/10) *Standardized base-model wrapper (freeze+embed)* | uniform frozen-encoder `encode(X)->{"z":...}` | Needed only for §3 **Option B** (embed with `euclid-dr1-mae` ourselves). Not blocking if Option A (precomputed) is used. |
| [#18](https://github.com/Birr0/flower/issues/18) *HF download interface* | easy pull of datasets/backbones from HF | Convenience for fetching `euclid_q1_embeddings` / `gz_euclid` / MAE; not blocking (can download manually first). |
| [`pmi_flux/README.md`](../pmi_flux/README.md) *(companion design, unimplemented)* | CFM log-likelihood scorer: `log p(z\|y)` via probability-flow-ODE + Hutchinson divergence, and the PMI/MI readouts | **Blocking for scoring.** Flower can *train* the conditional CFM today, but the anomaly score (§5.1) and MIs (§5.2) require this likelihood machinery, which is itself only a design. Phase 1 depends on a minimal implementation of it (conditional density + S1/S2); the full pmi flux/decomposition is not needed for the headline. |

### 8.3 Staging

- **Phase 0 — data + fact verification.** Materialise the cross-matched dataset; resolve every
  `[TO VERIFY]` (embedding block/dim/normalisation, the 3 missing answer-sets, tree dependencies,
  MER-derived properties, cross-match counts).
- **Phase 1 — headline anomaly result, no masking (ships before #8).** Train A0 and A1 (§5.4) with
  **binary** CFG (full morphology vector vs. unconditional). Interim NaN handling: sentinel/zero-fill
  or the §4 "always-asked core," explicitly flagged as a stopgap until #8. Deliver S1/S2 anomaly
  scores (§5.1) and the §6 validation battery. This is a complete, publishable result on its own.
- **Phase 2 — feature-set MIs (needs #8).** Once #8's variable-level masking lands, add per-subset
  conditioning masks for §5.3 conditional MIs and unify tree-NaN with MI-masking through the one
  "not provided" token.

### 8.4 Testing

Per CLAUDE.md testing strategy, **extend the integration layer** (`tests/test_integration.py`):
load the `euclid_Flow` config via `initialize_config_dir`/`compose`, override to CPU/tiny batch,
instantiate the `lightning_loader` graph, and run a mini training step — catching config/code drift
(e.g. `_target_` or `y_catalog` shape errors). No new isolated unit test unless a genuinely new
function appears.

### 8.5 Compute

Small: `z` is 384-d, the sample is ≤ the GZ Euclid footprint (100K–1M), the flow is a modest
`VelocityField`, and no image encoder runs at train time (Option A). Single-GPU (or CPU for
smoke-tests); the expensive-but-one-off cost is the change-of-variables likelihood integration over
the sample at scoring time (pmi §3 — ODE solver + Hutchinson divergence), not training.

## 9. Risks & open questions

**Risks (ordered by threat to the thesis).**

1. **Artifact-dominated tail (highest).** The MAE encodes artifacts (§7); if A0's artifact
   conditioning fails to suppress them, or A1 fails to recover them, the "clean isolation" claim
   collapses. Mitigated by the §5.4 ablation as an explicit control, but this is the make-or-break
   risk.
2. **Novelty over Wu & Walmsley is asserted, not proven.** Same MAE, same "beyond morphology" goal.
   The edge (per-object scoring, exhaustive conditioning-out, feature-set MIs) must show up as a
   concrete result (§6.4 criteria 3–4), or this reads as a method reskin. Feature-set MIs (Phase 2 /
   #8) are the most defensible differentiator, and they are gated on a dependency.
3. **RR2-vs-DR1 representation mismatch.** If we use precomputed RR2 embeddings but the SAE paper
   used a different MAE, the head-to-head in §6.3 is not apples-to-apples. Resolved by the §3
   `[TO VERIFY]`.
4. **S1 just ranks rare morphologies.** Addressed by reporting S2 alongside, but if S2's
   morphology-local standardisation is poorly defined the within-class claim weakens.
5. **Masked conditioning instability (Phase 2).** Per-subset CFG masking over a tree-structured,
   partially-NaN conditioning vector may train poorly; the §4 "always-asked core" is the fallback.

**Open questions (consolidated `[TO VERIFY]` list).**
- Exact quotes/section locations in arXiv:2510.23749 (§7).
- MAE embedding: which `pooled_features_block_*` is the `global_pool: avg` output; normalisation;
  which MAE the SAE paper used (§3).
- `clumps`/`problem`/`artifact` answer sets and the exact tree dependency edges (§4).
- Whether pmi's CFG samples partial masks or is binary-only (§5.3).
- MER-released derived properties (mass/SFR/photo-z) and multi-wavelength/spectroscopic Q1 coverage
  (§6.2).
- Cross-match join guarantees and overlap counts across the three tables (§2).

---

## 10. Timeline

Estimates assume **one researcher at ~0.6 FTE** on this project, with the flower-side feature work
(#8, and a minimal pmi likelihood scorer) done by the same person or a collaborator. Ranges, not
promises; durations are working-time, and the two **external dependencies** (#8 PR and the pmi
likelihood implementation) are the critical path, not the science.

### Critical path

```
Phase 0 (data + verify) ──► Phase 1a (train A0/A1)
                        └──► [pmi scorer impl] ──► Phase 1b (score S1/S2) ──► Phase 1c (validate) ──► WRITE-UP (headline)
                                                                                                          │
                        [#8 masking PR] ─────────────────────────► Phase 2 (feature-set MIs) ───────────►┘ (extended paper)
```

The pmi likelihood scorer gates the *result*; #8 gates the *differentiating* Phase-2 analysis.
Phase 0 and both dependency implementations can proceed in parallel from day one.

### Phased schedule

| Phase | Work | Depends on | Est. |
| ----- | ---- | ---------- | ---- |
| **0. Data + fact verification** | Materialise the cross-matched `{z, morphology}` dataset; resolve the §9 `[TO VERIFY]` list (embedding block/dim/norm, 3 missing answer-sets, tree edges, MER properties, overlap counts); build `y_catalog`; write the integration test. | data access (have it) | **1–2 wk** |
| **Dep A. Minimal pmi scorer** | Implement `log p(z\|y)` via probability-flow ODE + Hutchinson divergence and the S1/S2 + normalised-MI readouts, per `pmi_flux/README.md`. Reusable across both designs. | — (parallel with 0) | **2–4 wk** |
| **Dep B. #8 variable-level masking** | Land the condition-embedder upgrade (null-masking per question + per-variable projections). Enables clean tree-NaN handling and Phase 2. | — (parallel) | **2–3 wk** |
| **1a. Train A0/A1** | Train the conditional CFM over MAE `z` with binary CFG, both conditioning sets (§5.4). NaN stopgap (sentinel/zero-fill or always-asked core) until #8. | Phase 0 | **1 wk** (+ tuning) |
| **1b. Score** | Run S1/S2 scoring over the sample; optional vector residual. | 1a, Dep A | **1 wk** |
| **1c. Validation battery** | Expert inspection (top-N, blind labels); MER enrichment + residual-predictivity probe; strong-lens/merger recall; multi-λ / spectroscopic where available; positive controls (A1 artifact recovery). | 1b | **2–4 wk** (expert time is the variable) |
| **Headline write-up** | Paper/report on the Phase-1 anomaly result — complete and publishable on its own. | 1c | **2–3 wk** |
| **2. Feature-set MIs** | Per-subset conditioning masks; `I(z;y_S)` and incremental MIs; unify tree-NaN with MI-masking. Retrain with subset-sampled masks. | Dep B, 1a | **2–3 wk** |
| **Extended write-up** | Fold Phase-2 decomposition into the paper — this is the primary differentiator over the SAE work. | 2, headline | **1–2 wk** |

### Rough calendar

- **Headline result (Phase 0 → 1c):** ~**6–11 weeks** wall-clock, bounded below by `max(Dep A, Phase 0→1a)` and by expert-inspection turnaround in 1c.
- **Submittable headline paper:** ~**8–14 weeks**.
- **Full paper incl. feature-set MIs (Phase 2):** add ~**3–5 weeks** on top, gated on the #8 PR
  landing.

### Schedule risks

- **Expert-inspection latency (1c)** is the least controllable — it is human, iterative, and the
  arbiter of "interesting." Front-load a small pilot inspection right after 1b to de-risk.
- **Dependency slip.** If Dep A slips, the headline slips; if Dep B (#8) slips, only Phase 2 slips —
  so the plan deliberately puts the *publishable* result behind Dep A only, and the *differentiator*
  behind #8.
- **Artifact contamination (§9 risk 1)** could force an extra cleaning/iteration loop in 1c; the
  A0/A1 ablation is designed to catch it early rather than at write-up.
