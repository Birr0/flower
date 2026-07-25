↑ [Back to README](README.md)

# Method

## 1. Design overview & architecture mapping

**One-line statement.** Flow a CFM model over frozen MAE embeddings of Euclid cutouts, conditioned
on Galaxy Zoo Euclid morphology vote fractions; score each object by how improbable its embedding
is under the morphology-conditioned flow; the low-likelihood tail is the "beyond morphology"
anomaly set.

**Mapping onto existing Flower machinery.** Flower's canonical pipeline is *VAE encoder → latent
`z` → CFM over `z` conditioned on `y_catalog`*. The `spender_*_flow` experiments (existing) already
specialise this to the case where the encoder is a **frozen pretrained model** rather than a
trained VAE: `flower.models.spectra.PretrainedSpender` wraps a frozen `spender` encoder whose
`encode(X)` returns `{"z": ...}`, and `LightningFlowMatching` flows over that `z`. This experiment
is the same shape with two substitutions:

| Flower role            | Canonical (rgbmnist)      | This experiment                                        |
| ---------------------- | ------------------------- | ------------------------------------------------------ |
| Base encoder → `z`     | trained `VAE`             | **frozen `euclid-dr1-mae`** (spender-style wrapper)    |
| Base representation    | VAE latent                | **MAE pooled embedding** of an RGB-ified Euclid cutout |
| `y_catalog` conditioner| dataset factors (labels)  | **GZ Euclid morphology vote fractions** (continuous)   |
| Flow model             | `LightningFlowMatching`   | same, `_target_` at a new/reused module                |
| Residual of interest   | latent \ known factors    | **MAE embedding \ morphology** = "beyond morphology"   |

Because the decision "Frozen MAE, flow direct" was taken (no VAE on top of the MAE embedding), the
base representation is *deterministic and fixed* — there is no `mu`/`logvar`, `encode` returns a
bare `{"z": embedding}`, exactly the frozen-encoder contract the encoder-output standardisation of
issue #9 already accommodates. This means the *only genuinely new pieces* are (a) a data module that
serves aligned `(MAE embedding, morphology vote fractions)` pairs and (b) a `y_catalog` describing
the GZ Euclid morphology answers as continuous conditioning variables. The flow model, CFG
conditioning, and training loop are reused.

**Why conditioning yields "beyond morphology."** Conditioning the flow on morphology forces the
model to explain away all embedding structure that co-varies with morphology; what remains — the
conditional density's shape, and in particular its low-probability regions — is structure the
morphology vote fractions do not predict. An object with an ordinary morphology but an unusual MAE
embedding *for that morphology* receives a low conditional likelihood. That is the operational
definition of a "beyond-morphology" outlier used throughout this design.

**Non-goals (this iteration).** No VAE is trained on the embeddings; no baseline comparison against
raw-MAE (unconditioned) outliers is included as a deliverable (per design decision), though §5 notes
where the unconditional pathway still appears for scoring reasons; morphology enters as
*probabilities / vote fractions*, not as a Zoobot embedding vector.

---

## 2. Data

All inputs are published by Walmsley on the Hugging Face Hub and available to the user with
permissions (several are 🔒 gated — access already granted). The experiment needs, per Euclid
object, an aligned triple: **(MAE embedding, GZ Euclid morphology vote fractions, identifier +
imaging for inspection)**. The relevant repos:

| Role                                   | HF repo                                             | Notes |
| -------------------------------------- | --------------------------------------------------- | ----- |
| Base representation (precomputed)      | [`mwalmsley/euclid_q1_embeddings`](https://hf.co/datasets/mwalmsley/euclid_q1_embeddings) | 1M–10M rows; parquet. **[TO VERIFY]** that it contains the `euclid-dr1-mae` vector (vs. Zoobot embeddings) and the embedding dimension. |
| Base encoder (if we embed ourselves)   | [`mwalmsley/euclid-dr1-mae`](https://hf.co/mwalmsley/euclid-dr1-mae) | `timm` MAE, trained on `euclid_dr1`, cited by arXiv:2510.23749. Fallback if precomputed embeddings are unsuitable. |
| Conditioning (morphology)              | [`mwalmsley/gz_euclid`](https://hf.co/datasets/mwalmsley/gz_euclid) | 100K–1M rows; GZ Euclid vote fractions + imaging. Defines the `y_catalog` (§4). |
| Imaging / cutouts (inspection)         | [`mwalmsley/euclid_q1`](https://hf.co/datasets/mwalmsley/euclid_q1) | RGB-ified Euclid Q1 cutouts for expert inspection of anomalies. |
| Validation — strong lenses (§6)        | [`mwalmsley/euclid_strong_lens_expert_judges`](https://hf.co/datasets/mwalmsley/euclid_strong_lens_expert_judges) | 10K–100K expert-judged lens candidates; ready-made rare-object recall set. |

**Sample definition.** The modelled sample is the **cross-match** of `euclid_q1_embeddings` (MAE
vector present) with `gz_euclid` (morphology present). The morphology catalog (100K–1M) is the
binding constraint, so the usable sample is at most the GZ Euclid footprint. **[TO VERIFY]** the
join key (Euclid object ID / `id_str`) and the resulting overlap count.

**Provenance / RGB-ification.** The MAE consumes RGB-ified cutouts; per the design decision we take
the **precomputed** embedding where available rather than re-deriving the RGB mapping, so the exact
RGB construction is inherited from Walmsley's embedding pipeline and does not need re-implementing.
**[TO VERIFY]** that `euclid_q1_embeddings` was produced with `euclid-dr1-mae` (not `euclid-rr2-mae`
or a Zoobot encoder) so base and paper-reference agree.

**What flower needs on disk.** Flower's existing embedding path
(`flower.data.embeddings.FlowerEmbeddingDataset`, existing) loads a HF `Dataset` from disk and
returns each row as tensors, attaching a `y_catalog`. So the concrete data task is: build one
on-disk HF dataset whose rows are `{z: <MAE embedding>, <morphology answer columns...>}`, plus a
`y_catalog` (§4) describing the morphology columns. No new Dataset class is strictly required if the
cross-matched table is materialised into that shape — see §8.

**Open data questions (for the verification pass).**
- Exact MAE embedding dimensionality and whether it is L2-normalised / standardised in the release.
- Whether to model on the full GZ Euclid overlap or a cleaned subset (e.g. drop low-`smooth` vote
  counts, magnitude/size cuts) — affects both the flow fit and the artifact contamination (§7).
- Join keys and per-object 1:1 guarantee across the three tables.

---

## 3. The base representation: the Euclid galaxy MAE

**Architecture (confirmed from `euclid-dr1-mae/config.yaml`).** A `timm`
`VisionTransformer` encoder with `embed_dim: 384`, `patch_size: 8`, `depth: 12`, `num_heads: 6`,
`mlp_ratio: 4`, `global_pool: avg`, `num_classes: 0`, paired with a lightly `MAEDecoder`
(`embed_dim: 512`, depth 3) and `mask_ratio: 0.9`. So the per-object base representation `z` is the
**average-pooled 384-dimensional** encoder output of a (224px, patch-8) RGB-ified cutout. This is the
`z` the CFM flows over; there is no `mu`/`logvar` (deterministic frozen encoder, §1).

**Which MAE, and where the embeddings come from — a decision.** There are two sources, and they are
not the same model:

| Option | Source | Dim | Compute | Caveat |
| ------ | ------ | --- | ------- | ------ |
| **A — precomputed** | `euclid_q1_embeddings`, `pooled_features_block_*` | 384/block (×12 blocks) | none | Produced by the **RR2** MAE (`rr2-mae-encoder`), *not* `euclid-dr1-mae`; dataset predates the DR1 model. |
| **B — embed ourselves** | run `euclid-dr1-mae` over `euclid_q1` cutouts | 384 (avg-pool) | one forward pass over the sample | Matches the DR1 model card exactly; must reproduce the RGB-ification. |

**[TO VERIFY — consequential]** which MAE the Wu & Walmsley SAE paper (arXiv:2510.23749, Oct 2025)
actually used. `euclid-dr1-mae` was published **Nov 2025**, *after* that paper, so the SAE analysis
we position against in §7 was almost certainly run on the **RR2** MAE (or the
`euclid_encoder_mae_zoobot_vit_small_patch8_224` variant), not DR1. If so, **Option A is not the
inferior shortcut — it is the choice that keeps us on the same representation as the work we compare
to.** Recommendation: **Option A, using the final block** (`pooled_features_block_11`) as the 384-dim
`z`, pending confirmation of which block corresponds to the `global_pool: avg` output; keep Option B
as the fallback if we deliberately want the newer DR1 representation.

**Open choices (verification pass).**
- Which of the 12 `pooled_features_block_*` is *the* embedding (final block vs. a concatenation vs.
  a mid-block the SAE paper favoured).
- Whether the released vectors are already L2-normalised / standardised; if not, decide the
  standardisation applied before flow training (flows are scale-sensitive).

---

## 4. The conditioning: Galaxy Zoo Euclid morphology

Source: Euclid Collaboration: Walmsley et al. (2025), *"Euclid Q1: First visual morphology
catalogue."* The conditioning values are **Zoobot predictions of volunteer vote fractions**, not raw
volunteer answers — "Zoobot predicts every answer to every question." They are released on HF
(`gz_euclid`) as `question_answer_fraction` columns, with parallel `_dirichlet` columns encoding
uncertainty. **Naming correction (confirmed from the `gz_euclid` schema):** the question slugs carry
a `-euclid` suffix, so the real column names are e.g. `smooth-or-featured-euclid_smooth_fraction`,
`disk-edge-on-euclid_yes_fraction`, `bar-euclid_strong_fraction` — the tree table below omits the
`-euclid` infix for readability.

### The decision tree (13 questions)

From Table A.1 and the Fig. 13 legend. Answer lists confirmed for the first 10 questions;
`clumps` / `problem` / `artifact` answer names are **[TO VERIFY]** (Fig. 14 omits them and the
answer legend did not enumerate them).

| # | Question (`slug`)       | Answers (`_fraction` suffixes)                                  | Asked when |
| - | ----------------------- | -------------------------------------------------------------- | ---------- |
| 1 | `smooth-or-featured`    | smooth, featured-or-disk, problem                              | always |
| 2 | `disk-edge-on`          | yes, no                                                        | featured-or-disk |
| 3 | `has-spiral-arms`       | yes, no                                                        | disk, not edge-on |
| 4 | `bar`                   | strong, weak, no                                              | disk, not edge-on |
| 5 | `bulge-size`            | dominant, large, moderate, small, none                        | disk, not edge-on |
| 6 | `how-rounded`           | round, in-between, cigar-shaped                               | smooth |
| 7 | `edge-on-bulge`         | boxy, none, rounded                                           | edge-on disk |
| 8 | `spiral-winding`        | tight, medium, loose                                          | has spiral arms |
| 9 | `spiral-arm-count`      | 1, 2, 3, 4, more-than-4, cant-tell                            | has spiral arms |
| 10| `merging`               | none, minor-disturbance, major-disturbance, merger            | always |
| 11| `clumps`                | **[TO VERIFY]**                                               | featured |
| 12| `problem`               | **[TO VERIFY]**                                               | (rare) |
| 13| `artifact`              | **[TO VERIFY]**                                               | (rare) |

The confirmed 10 questions contribute **34 vote-fraction answers**; the full 13 total is
`34 + |clumps| + |problem| + |artifact|` — resolve during verification. (The `gz_euclid` card was
truncated past `bar-euclid` on fetch, so the last three answer-sets must be read from the parquet
schema directly, e.g. `datasets.load_dataset("mwalmsley/gz_euclid").column_names`.) The "Asked when" column is
reconstructed from the GZ decision-tree convention and the paper's example
(`smooth-or-featured_featured-or-disk_fraction > 0.5 AND disk-edge-on_no > 0.5` ⇒ featured face-on);
**[TO VERIFY]** the exact dependency edges against the released tree.

### Structural fact that drives the design: tree-shaped missingness

The catalogue sets a question's vote fractions to **NaN where that question is "not relevant"**,
defined as *leaf probability < 0.5* (the product of the vote fractions leading to it). So a smooth
face-on galaxy has NaN for `spiral-arm-count`, `edge-on-bulge`, etc. **This means the conditioning
vector is natively per-object masked by the decision tree.** Two consequences:

1. **The masking extension (§1 edge #3, §5) is not bolted on — it is intrinsic.** The tree already
   defines, per object, which morphology questions carry information. The flow must accept
   partial/absent conditioning by construction, which is exactly what enables per-feature-set
   conditional MIs. We should therefore adopt CFG-style conditioning-dropout that treats NaN answers
   as "not provided" rather than imputing them, so a leaf question genuinely absent for an object and
   a leaf question *dropped for MI estimation* use the same mechanism.
2. **Two candidate conditioning designs (decision for §5):**
   - *(a) Full tree, masked.* Condition on all 34+ answers, NaN → mask token. Maximal morphology
     information; requires masked conditioning to be robust.
   - *(b) Always-asked core.* Condition only on the two always-asked questions
     (`smooth-or-featured`, `merging`) plus, optionally, `how-rounded`/`disk-edge-on` — no missingness,
     simplest, but explains away far less morphology (weaker "beyond morphology" claim).
   Recommendation: **(a)**, because the whole novelty thesis is *exhaustive* isolation of
   non-morphological structure; (b) is a fallback if masked conditioning proves unstable.

### `y_catalog` contract

Each answer column becomes a **continuous** conditioning variable (vote fraction ∈ [0,1]). The
`y_catalog` (per `flower.models.modules.get_conditional_len` /
`get_no_of_continuous_variables`, existing) lists each `question_answer` name, `size=1`,
`continuous=True`. Open choices for the verification pass:
- Whether to also feed the `_dirichlet` uncertainty columns (richer, but doubles conditioning dim and
  changes the semantics from "the morphology" to "the morphology *and its confidence*").
- Whether to include `artifact` (and `problem`) as conditioning — doing so lets the flow *explain
  away artifact-driven embedding structure*, directly attacking the artifact-contamination risk of
  §7; excluding them keeps conditioning purely astrophysical. **This is a substantive modelling
  choice, flagged for §5/§6.**
- NaN encoding: mask token vs. sentinel value vs. zero-fill — must be consistent with the CFG
  dropout mechanism.

**Cross-match keys (feeds §2).** `id_str` (formatted `{release}_{tile_index}_{object_id}`) joins the
catalogue to the embeddings table; `object_id` + `tile_index` join to the MER catalogue — which
carries photometry, and (per the paper) correlates with "mass, star formation rate, location in the
cosmic web," i.e. the §6 physical-property validation signals come from the same MER join.

---

## 5. The residual / anomaly score

This section defines "beyond morphology" operationally and reuses, rather than re-derives, the
likelihood/MI machinery specified in the companion design
[`pmi_flux/README.md`](../pmi_flux/README.md). **This experiment is an application of
that estimator**: train one CFG conditional CFM over the MAE embedding `z` conditioned on morphology
`y`, then read anomalies and MIs off its conditional/unconditional pathways. All references below of
the form "(pmi §X)" point into that document.

### 5.1 Per-object anomaly score (headline deliverable)

The flow gives the conditional log-density `log p(z | y)` via the instantaneous
change-of-variables integral along the probability-flow ODE (pmi §2.2, with the ODE-solver /
Hutchinson-divergence details in pmi §3). We report **two distinct scores**, and the distinction is load-bearing — they answer different
questions and will surface different objects, so both are deliverables and §6 validates each:

- **(S1) Absolute score** — `a_abs = − log p(z_MAE | y_morphology)`. High when the embedding is
  improbable given morphology *in absolute terms*. Because conditional entropy varies across the
  morphology manifold (smooth ellipticals occupy a tighter embedding region than disturbed mergers),
  `a_abs` mixes two effects: genuine per-object surprise **and** the intrinsic rareness/looseness of
  the object's morphology class. Its tail therefore over-represents rare or high-entropy morphologies.
- **(S2) Morphology-local score** — `a_loc = a_abs` standardised within a morphology neighbourhood
  (e.g. minus the conditional-neighbourhood mean over comparable `y`, or a rank/z-score within
  morphology bins). High only when an object is atypical **relative to others of its own
  morphology**, factoring out class-level rareness.

**The distinction stated plainly:** `a_abs` asks *"is this embedding unusual, full stop, once
morphology is accounted for?"* and will be dominated by objects that are both morphologically and
representationally rare; `a_loc` asks *"is this embedding unusual for a galaxy that looks like
this?"* and isolates within-class outliers a class-level score would bury. Neither is "correct" —
`a_abs` is the right lens for global novelty/artifact discovery, `a_loc` for finding the odd object
inside an otherwise ordinary morphology bin. §6 reports enrichment for both.

- **Vector residual, optional.** Beyond the scalars, the conditional-vs-unconditional displacement of
  the flow (pmi §2.4's displacement/work identity) gives a *per-object residual direction* in
  embedding space, not just a magnitude — useful for clustering anomalies by *how* they exceed
  morphology. Optional, not required for the headline.

### 5.2 Normalised mutual information (how much morphology explains)

Averaging the pointwise `PMI(z, y) = log p(z|y) − log p(z)` over the sample gives
`I(z; y) = E[PMI]` (pmi §2.2). To make it a *fraction-explained* reading rather than a nats value,
report a **normalised** quantity — the design decision from earlier — e.g. `I(z; y) / H(z)` (share of
embedding entropy accounted for by morphology) with `H(z)` from the unconditional flow, or an
`R²`-like `1 − H(z|y)/H(z)`. **[TO VERIFY]** the exact normaliser against pmi §2.2's conventions so
the two docs agree.

### 5.3 Feature-set–resolved MI via conditioning masks

The masking extension. Train the CFM with **classifier-free-guidance conditioning dropout that
operates per morphology question**, so the model can evaluate `p(z | y_S)` for any subset `S` of the
13 questions — not only the full vector or the empty set. This is the *same* mechanism that must
already handle the tree-shaped NaN missingness of §4 (a question absent for an object = a dropped
question), so it is architecturally free. It yields:
- **Per-question / per-feature-set MI:** `I(z; y_S)` for chosen `S` (e.g. `S = {smooth-or-featured}`
  vs `S = {spiral-arm-count, spiral-winding}`) — which morphology axes explain which embedding
  structure.
- **Incremental / conditional MI:** `I(z; y_j | y_{S})` — the information a question adds beyond a
  set already conditioned on, a decomposition a static SAE dictionary cannot produce (§7 edge #3).

Design consequences for training: masks must be sampled over subsets during training (not just
all-or-nothing CFG), and NaN answers must map to the *same* "not provided" token as an
MI-dropped answer, so absence-by-tree and absence-by-design are indistinguishable to the model.
**[TO VERIFY]** whether pmi's CFG setup already samples partial masks or assumes binary
conditional/unconditional — if binary, this is the one genuine training-side extension this
experiment adds over pmi.

### 5.4 Artifact conditioning + ablation (decision: include artifacts)

Per design decision, the conditioning set **includes** the `artifact` (and `problem`) vote fractions.
Rationale: the MAE demonstrably encodes imaging artifacts (§7, Wu & Walmsley), so conditioning on the
artifact answers lets the flow **explain those away**, cleaning the astrophysical anomaly tail. We
formalise this as an **ablation over the conditioning set**:

| Run | Conditioning `y` | Expected effect on the anomaly tail |
| --- | ---------------- | ----------------------------------- |
| **A0** (primary) | all 13 questions incl. `artifact`/`problem` | artifacts suppressed; tail enriched in genuine astrophysical novelty |
| **A1** (ablation) | astrophysical questions only (drop `artifact`/`problem`) | **artifacts should re-appear in the tail** — recovering known artifacts as anomalies is a positive control that the score tracks embedding structure the conditioning omits |

A1 doubles as the sharpest mechanism check in the whole design: if dropping the artifact
conditioning does *not* push saturated stars / ghosts into the anomaly tail, the claim that
conditioning isolates "the structure `y` omits" is not actually holding. Conversely, A0's tail —
with artifacts conditioned away — is where the beyond-morphology-*and*-beyond-artifacts science
lives, and is the set §6 validates. (This is naturally expressed through the §5.3 masking machinery:
A0 and A1 are just two conditioning masks, no separate model required.)
