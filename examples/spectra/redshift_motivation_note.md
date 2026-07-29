# Motivation: why a conditional flow beats fixed-basis ICA for redshift removal

Companion to `ivae_sweep_results/redshift_motivation.png` (produced by
`redshift_motivation.py`). Where the sweep notes report *that* Flower removes
redshift better, this explains *why the fixed-basis baselines cannot* — the
structural motivation for a conditional flow.

## The problem: redshift is distributed across the basis

ICA/iVAE hand you a **fixed independent basis** and remove a factor by deleting
the axes that carry it. But redshift is not axis-localised. Measuring each
component's |correlation| with `z` and taking the **participation ratio**
(effective number of contributing axes, `(Σw)²/Σw²`):

| basis (spender_I, 10-dim) | effective # z-carrying axes |
|---|---|
| FastICA | **3.8** |
| iVAE | **4.4** |

Redshift lives in ~4 of 10 axes (and, being nonlinearly encoded, is
MLP-recoverable from the remainder until ~7 are gone). Crucially, the components
with the most redshift dependence are also the ones with the most *physics*
dependence (Panel 1/2: the physics curve rides on top of the z bars) — the two
share the same axes.

## The consequence: axis-deletion cannot separate them

Because `z` and the physics occupy the same axes, deleting enough components to
remove `z` necessarily removes the physics. From the residual-A sweep (Panel 3):
`z` R² only reaches ~0 after ~7 of 10 components are dropped, by which point the
physical targets have collapsed.

**At matched redshift removal (z R² ≈ 0.03):**

| method | mean physics R² (logM*, logSFR, A_v) | axes dropped |
|---|---|---|
| FastICA (drop) | 0.16 | 7 |
| iVAE (drop) | 0.31 | 6 |
| **Flower cond** | **0.45** | **0** |

## Why Flower is different

Flower never commits to an axis-aligned basis. Its conditional flow re-expresses
the *whole* representation given `z`, removing redshift's influence as a
coordinated transformation rather than by deleting coordinates. So it removes
`z` (R² 0.03) while retaining the physics (0.45) at **zero** axis-deletions — a
point off the entire ICA/iVAE axis-deletion frontier. (Being invertible, the same
model can also re-inject a chosen `z` — counterfactual redshifting — though that
is not needed for the removal argument here.)

The strongly z-entangled stellar mass (logM*) is the exception that proves the
rule: it is genuinely redshift-correlated (flux-limited selection), so it is
partly removed by *every* method, Flower included — its loss is a property of the
data, not of the remover.

## One line

Redshift is smeared across ~4 of the basis axes together with the physics, so no
fixed basis can excise it by axis-deletion without collateral damage; Flower
removes it as a whole-representation conditional transform and keeps ~3x more of
the physics at matched removal.
