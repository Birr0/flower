# Agent-Mediated Anomaly Triage over a Conditional Generative Density

*Experiment design — working system on Flower. Draft for discussion.*
*Scope agreed: **two-stage** — Stage 1 recall = model-marginalised conditional surprisal `S̄` over a
K-seed ensemble (model-independent "unusual"); Stage 2 precision = an agent gathers cross-match
evidence per candidate and a **learned, interpretable preference model** re-ranks for
"interesting." Human-in-the-loop is the **spine, not a deferred stage** · per-astronomer policies
partially pooled to a consensus, with inter-astronomer disagreement surfaced as a signal ·
evaluation = **within-subject real-user study** (discoveries-per-inspection) with a withheld rare
class as objective anchor · grounded in spectra (`spender_I_flow`).*

## Abstract

> *Astronomy is what humans do when they do astronomy.*

Researcher attention is the scarce resource in survey astronomy, and *unusual* is not the same as
*interesting*. We turn Flower's conditional density `p(z | y)` — a flow over a frozen encoder latent
conditioned on a catalog `y` of known astrophysical factors — into a **two-stage discovery loop**.

**Stage 1 (recall) ranks by the model-marginalised conditional surprisal**
`S̄ = mean_k[−log p_k(z | y)]` over a **K-seed ensemble**: an object enters the candidate pool only
if *every* seed finds it unlikely given known factors, so anomalies are properties of the data — not
of one model's fit. Seed disagreement `U = Var_k[…]` is reported as a *confidence tier*, not a
ranking term. A `drop_variables` decomposition localises *which* known factor each candidate violates.

**Stage 2 (precision) turns "unusual" into "worth your time."** An LLM agent **autonomously
cross-matches** each candidate (SIMBAD/Gaia/SDSS), extracts the heterogeneous responses into a
**fixed structured evidence record**, and a **learned preference model** re-ranks the pool. That
model is deliberately interpretable: a **linear part warm-started from an astronomer's elicited
per-factor scoring rubric** (so it is useful from the first label), **plus a small nonlinear
residual** that captures the *conjunctions* — e.g. "catalogued as ordinary **and** spectrum conflicts
**and** violates redshift" — that a weighted sum cannot express. It is trained from **per-field
evidence scores** (dense) and occasional **holistic verdicts** (for the residual). Policies are
**per-astronomer, partially pooled** to a consensus; **inter-astronomer disagreement** becomes a
"objectively odd vs. taste-dependent" signal — the human-level twin of the model's seed disagreement
`U`. Cross-match works **both ways**: demote known artifacts, promote corroborated genuine oddities.
Reasoning is kept at **both cluster level** (evidence for a new *class*) and **object level**
(singletons carry the most novel signal).

## Section outline
- **[`related-work.md`](related-work.md)** — Astronomaly / active astro triage, contextual AD,
  generative ensembles, preference learning & elicited priors.
- **[`method.md`](method.md)** — Stage 1 surprisal ensemble + localisation; Stage 2 evidence record,
  the two-part preference model, per-astronomer pooling; Flower wiring.
- **[`evaluation.md`](evaluation.md)** — within-subject real-user study, two-layer ground truth,
  adaptation curve, baselines, controls.
- **[`planning.md`](planning.md)** — dependencies, risks, timeline, code mapping.
