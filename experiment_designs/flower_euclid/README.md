# Residual Representations of Euclid Beyond Morphology via Conditional Flow Matching

Status: design draft, not yet implemented. No code referenced here exists yet unless explicitly
marked "(existing)". This document is being built and verified section by section; sections marked
**[TO VERIFY]** contain claims or numbers not yet confirmed against data/checkpoints.

This design is split across four files; this README is the entry point. See the outline below for
where each section lives.

## Abstract

Galaxy morphology is a rich but *incomplete* description of a galaxy: two objects that a
morphology model reports as identical spirals can differ in redshift, environment, colour,
star-formation state, or in ways nobody has named yet. We ask what a general-purpose
self-supervised representation of Euclid imaging encodes *over and above* morphology, and whether
that residual is a productive place to look for scientifically interesting outliers.

Flower trains a conditional flow-matching (CFM) model over a base representation, conditioned on
known factors of variation, so that the fitted flow captures exactly the structure the conditioning
factors leave unexplained. We instantiate this on Euclid as follows. The **base representation** is
the fixed per-object embedding produced by a frozen, pretrained galaxy **masked autoencoder** — the
Euclid galaxy MAE of Walmsley & Wu (the `euclid-*-mae` family, trained to reconstruct Euclid imaging
from 90% masking; §3 selects the **precomputed RR2-MAE** embeddings, both to avoid an encoder pass
and to match the representation the related SAE work most likely used) — applied to RGB-ified Euclid
cutouts. The **conditioning** is the **Galaxy Zoo Euclid morphology** description of the same
object — the decision-tree vote fractions — entering through Flower's existing continuous
`y_catalog` pathway. A CFM model flows over the MAE embedding conditioned on morphology; the
conditional flow's own likelihood then defines a **residual / anomaly score**: objects the flow
finds improbable *given their morphology* are those whose MAE embedding carries structure morphology
does not account for. The headline deliverable is **anomaly / outlier discovery** — surfacing Euclid
objects that are unusual beyond what their morphology would predict — validated by expert inspection
plus a battery of four independent automated signals (physical-property enrichment, rare-object
recall, multi-wavelength counterparts, spectroscopic outliers; §6).

This is deliberately close to, and must be positioned honestly against, Wu & Walmsley (2025,
[arXiv:2510.23749](https://arxiv.org/abs/2510.23749)), who use the *same* Euclid galaxy MAE (modulo
the RR2/DR1 checkpoint question resolved in §3) with sparse autoencoders to identify interpretable
features *outside the Galaxy Zoo decision tree*. The scientific question ("what does the MAE encode
beyond morphology?") is shared; the mechanism (a conditional generative model of the embedding, with
a likelihood-based residual and anomaly score) is different. §7 addresses where — if anywhere — that
difference buys something the SAE approach does not.

## Section outline

All sections are drafted. Remaining unresolved specifics are consolidated in §9's `[TO VERIFY]`
checklist and are the input to the step-by-step verification pass.

**[`method.md`](method.md)**

1. [**Design overview & architecture mapping**](method.md#1-design-overview--architecture-mapping) —
   base/condition/residual mapped onto Flower's `spender_*_flow` frozen-encoder + CFM machinery.
2. [**Data**](method.md#2-data) — Euclid Q1 sample, the aligned per-object triple (embedding /
   morphology / imaging), HF provenance, cross-match and sample definition.
3. [**The base representation**](method.md#3-the-base-representation-the-euclid-galaxy-mae) — the
   Euclid galaxy MAE (384-d avg-pooled ViT); RR2-precomputed vs. DR1-embed-ourselves decision.
4. [**The conditioning**](method.md#4-the-conditioning-galaxy-zoo-euclid-morphology) — the GZ Euclid
   13-question decision tree, tree-shaped missingness, and the `y_catalog` contract.
5. [**The residual / anomaly score**](method.md#5-the-residual--anomaly-score) — two scores (S1
   absolute, S2 morphology-local), normalised MI, feature-set MIs via conditioning masks, and the
   artifact-conditioning ablation (A0/A1).

**[`evaluation.md`](evaluation.md)**

6. [**Evaluation & validation**](evaluation.md#6-evaluation--validation) — expert inspection + four
   automated verification signals + positive controls + success criteria.

**[`related-work.md`](related-work.md)**

7. [**Related work & novelty positioning**](related-work.md#7-related-work--novelty-positioning) —
   Wu & Walmsley SAE paper and the claimed edge.

**[`planning.md`](planning.md)**

8. [**Implementation plan**](planning.md#8-implementation-plan) — reuse-first config/data work,
   GitHub dependencies, staging, tests, compute.
9. [**Risks & open questions**](planning.md#9-risks--open-questions) — ranked risks and the
   consolidated `[TO VERIFY]` list.
10. [**Timeline**](planning.md#10-timeline) — phased schedule, dependencies, and the critical path.
