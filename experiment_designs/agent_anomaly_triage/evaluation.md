↑ [Back to README](README.md)

# Evaluation

The headline claim — *the loop saves researcher time by surfacing what this astronomer finds
interesting* — is a claim about a learned, partly **subjective** preference layer, so a simulated
labeler cannot carry it. Primary validation is a **within-subject real-user study**, layered on top
of an **objective** rare-class anchor that keeps the recall claim honest. Two axes, two kinds of
ground truth.

## 4.1 Ground truth — two layers
- **Objective anchor.** Withhold a rare spectral class (BAL quasars, a rare emission-line type,
  DR-flagged oddities) from training and seed it into the test pool. Recovery of the withheld class
  is class-agnostic ground truth that no participant can bias, and it defends against "they just liked
  what the model happened to show them."
- **Subjective interest.** Each astronomer's own evidence scores and holistic verdicts define
  *their* interestingness target — what the Stage-2 preference model must learn.

## 4.2 Primary — within-subject real-user study
Few astronomers will ever be recruited, so a between-subject design has no power. Use **within-subject,
counterbalanced arms**:
- **Arm A — pure `S̄`** (Stage 1 only, no learned re-ranking).
- **Arm B — full loop** (Stage 1 recall → agent evidence → learned preference model).

Each participant works both arms (order counterbalanced to control learning/fatigue). Report:
- **discoveries-per-inspection** (`recall@budget`) on the objective anchor — the "researcher-time"
  curve — Arm B vs Arm A.
- **within-session adaptation curve** — because a participant supplies only one sitting's labels, the
  measured quantity is *how fast the ranking improves over their first ~20–40 scores*. This is the
  real test of the elicited-prior warm start: the model must be useful at label #1 and visibly better
  by label #40.
- **subjective yield** — objects the participant themselves later rates "interesting," recovered per
  inspection.

## 4.3 Baselines
- **Random** and **embedding-distance** (kNN / isolation-forest in raw `spender` space — the
  Astronomaly-style feature-anomaly baseline).
- **Reconstruction error** of the encoder.
- **Single-model surprisal** vs **ensemble `S̄`** — Stage-1 ablation: `S̄` should improve rare-class
  recall and, critically, cut the seed-to-seed variance of the *ranking itself*. Report ranking
  stability across seeds, not just recall.
- **Elicited rubric only** (frozen linear, no residual, no learning) vs **full preference model** —
  the Stage-2 ablation: does learning off the prior + the residual beat the hand-set rubric within a
  session? This is the gate that the learned layer earns its labels.

## 4.4 Controls
- **Confound check** — verify recovery is not driven by a trivial correlate (SNR, magnitude, sector);
  regress it out and confirm the lift survives before any "saves researcher time" claim.
- **Shuffle-`y` control** — permute the conditioning; the *conditional* lift over the unconditional
  score must collapse, proving the Stage-1 gain comes from conditioning on known factors.
- **Localisation check** — for recovered objects, the agent's named "violated factor" (§method 1.1)
  should match the class's known physics for a majority of cases.
- **Confidence-flag check** — low-`U` (all-seeds-agree) candidates should show higher precision than
  high-`U` ones, validating `U` as a usable trust tier.
- **Inter-astronomer disagreement check** — objects with low human-ensemble variance should be
  recovered consistently across participants (objectively odd); high-variance objects should split by
  participant, validating disagreement as a "objective vs. taste" tier (§method 2.3).

## 4.5 What makes this a "go"
On the objective anchor, the full loop (Arm B) beats every baseline on `recall@budget` for ≥2 of 3
withheld classes; the within-session adaptation curve shows the preference model improving over a
single participant's labels and beating the frozen-rubric ablation; and the Stage-1 controls
(seed-`S̄` stability, shuffle-`y`) confirm the conditional signal. A within-subject win on
discoveries-per-inspection is the evidence that the worklist saves researcher time.
