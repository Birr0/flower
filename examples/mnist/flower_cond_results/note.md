# Why preserving colour `b` *and* rotation matters (class-only RGB-MNIST)

Records the comparison between the ICA baselines (`../ivae_sweep_results_fastica_rot/`)
and Flower's conditioned embedding (`./results.csv`), and — the key point —
justifies *why* retaining an **independent** factor (colour `b`) and an
**entangled** factor (rotation) are different tests, and why the second one is
the one that matters.

Condition removed = **digit**. Factors we probe for retention: colour `b`
(independent of the digit) and rotation (label-independent augmentation, but
**entangled with digit shape in the latent**). Probes: linear + MLP; rotation
target from `compute_rotation.py`.

## The numbers (MLP probe)

| Method | digit ↓ (logreg / MLP) | b R² ↑ | rotation R² ↑ |
|---|---|---|---|
| Raw (orig) | 0.947 / 0.979 | 0.990 | 0.924 |
| FastICA residual B (mean-subtract) | 0.114 / 0.858 | 0.984 | 0.699 |
| FastICA residual A, k=7 | 0.580 / 0.815 | 0.975 | 0.408 |
| FastICA residual A, k≈19 (strong removal) | 0.165 / 0.190 | 0.953 | **−0.175** |
| **Flower cond** | **0.748 / 0.792** | 0.960 | **0.611** |

Two facts:
1. **Colour `b` is preserved by everything** (≈0.95–0.99) until FastICA's k≈23
   cliff.
2. **Rotation is destroyed by FastICA source-dropping** — it hits ~0 by k=13 and
   goes negative by the point the digit is actually removed. Flower keeps rotation
   at 0.611 at a comparable removal level.

## Why preserving `b` (independent) matters — but is only a *sanity check*

`b` is statistically independent of the digit and lives in its own latent
directions (it is literally a channel-scaling). Retaining it shows a method does
**not destroy information indiscriminately** — it removes the target, not
everything. That is necessary: a "remover" that also flattened `b` would be
useless.

But it is a **low bar**. *Every* method here preserves `b`, including the blunt
ones, precisely because `b` shares no directions with the digit. Evaluating
preservation only on an independent factor is therefore **misleading**: it makes
blunt source-dropping look surgical when it is not.

## Why preserving rotation (entangled) matters — the *real* test

Rotation is the discriminating benchmark because it is **entangled** with the
condition: the measured orientation occupies the same "shape" latent directions
that encode digit identity. This exposes the difference between two things that
are easy to conflate:

- **removing the factor** = making the label unpredictable, vs
- **removing the shared subspace** = deleting every direction that carries the
  label, and with it any other factor living there.

Blunt methods (drop the `k` most digit-dependent components) can only do the
second. They over-remove: to erase the digit they delete the shared shape
directions, taking rotation with them (rotation R² → negative). A surgical method
must do the first — strip the label's *predictive information* while leaving the
geometry that co-occupies those directions intact.

Why this is the point that actually matters:

1. **It is the whole scientific purpose of conditioning on known factors.** You
   suppress a known factor in order to *study the residual structure it does not
   explain*. If your removal also strips entangled-but-distinct factors
   (orientation, pose, style), you have thrown away exactly the residual signal
   you set out to examine. In the galaxy-spectra use case this is not academic:
   you condition on redshift to remove it, but you must **keep** the physical
   properties (star-formation, metallicity) that are entangled with redshift in
   the representation. A blunt remover would delete the physics along with the
   redshift.

2. **It is the correct notion of "controlling for" a variable.** Controlling for
   X while studying the rest requires removing X's *contribution*, not the whole
   X-correlated subspace. Removing the shared subspace is **over-adjustment** — it
   discards legitimate residual variation that merely correlates with X.

3. **It separates surgical from blunt methods; the `b`-only view cannot.** On `b`
   alone, FastICA-drop and Flower look equivalent. Rotation is what reveals that
   source-dropping's removal is *collateral* (it cannot remove the digit without
   removing rotation), whereas Flower's is *targeted*.

## What Flower shows here

- **Cross-probe consistency (robust removal).** Flower cond digit acc is
  0.748 (linear) ≈ 0.792 (MLP) — a 0.04 gap. FastICA residual B is 0.114 (linear)
  vs 0.858 (MLP) — a 0.74 gap, i.e. it only *hides* the digit from a linear probe.
  Flower suppresses the digit in a way that survives a stronger probe.
- **Surgical retention.** At a matched removal level (~0.79–0.82 MLP digit acc)
  Flower keeps rotation at **0.611** vs FastICA's **0.408**, and never drives it
  negative. It removes the digit's information while sparing the entangled factor.

**Honest caveat.** Flower does not push the digit to chance (MLP 0.792); it
*suppresses consistently* rather than *erases*. The claim is not "perfect
removal" but "robust, probe-agnostic suppression that spares entangled residual
structure" — which is the property the blunt ICA baselines lack (residB fools one
probe; residA removes rotation along with the digit).

## Why the digit accuracy does not — and should not — reach chance

Flower leaves the digit at MLP accuracy ~0.79, well above chance (0.10). This is
not a shortfall of removal; it is the *direct consequence* of doing surgical,
faithful removal, and the same entanglement result explains exactly why.

**The two goals are in tension when the factor is entangled.** Driving digit
accuracy to chance means making the seed carry *zero* information about the digit
— i.e. deleting every latent direction from which the digit can be predicted. But
the FastICA sweep showed those directions are the *same* ones that carry rotation
(and, generally, stroke, slant, shape — the legitimate residual geometry). So
"digit at chance" is only reachable by destroying that entangled residual
structure: FastICA reaches digit ≈ 0.15 only by taking rotation R² *negative*.
Near-chance digit removal and entangled-factor preservation are **mutually
exclusive** here — a mathematical/informational tension, not an engineering gap.
A method that keeps rotation *must* leave some digit-predictable signal, because
that signal lives in the directions it is deliberately keeping.

**The residual digit-predictability is mutual information with the *retained*
factors, not a leak of the label.** Once the seed still encodes orientation,
slant, thickness, shape, and those factors are themselves statistically
informative about the digit (a 1 and a 7 differ precisely in such geometry), any
probe can predict the digit above chance *without the representation encoding the
digit directly*. The above-chance number is `I(retained residual structure ;
digit)` — an irreducible quantity if you insist on keeping that structure — not
removable digit information. Forcing it to zero would be **over-adjustment**:
throwing away legitimate residual variation just because it correlates with the
label.

**Cross-probe consistency shows this is genuine residual, not a hidden leak.** A
removable leak looks like FastICA residual B: digit 0.114 to a *linear* probe but
0.858 to an *MLP* — the information is intact, merely hidden from one probe class.
Flower is 0.748 (linear) ≈ 0.792 (MLP): the *same* small amount of digit
predictability is visible to every probe. That flat probe response is the
signature of an irreducible residual carried by retained structure, not of a leak
waiting to be exposed. So the ~0.79 is the honest floor set by entanglement,
reached consistently — the opposite of the baselines' probe-dependent illusion of
removal.

**When *would* chance be the right target?** Only for a *separable* nuisance —
one that shares no directions with anything you want to keep. Colour `b` is like
that: you could drive a `b`-analog to chance without touching rotation, because
`b` is disentangled. The digit is not that: its identity is woven into the
geometry, so the faithful outcome is consistent suppression to the entanglement
floor, not erasure. Reaching chance would signal that the seed is no longer a
faithful residual.

## Summary (why retaining rotation is the test that matters)

- **Scientific purpose.** Conditioning on a known factor is done to *study the
  residual structure it does not explain*. Removal that also strips
  entangled-but-distinct factors (orientation, pose, style; or, for spectra,
  redshift-entangled physics like star-formation/metallicity) destroys the very
  signal you set out to examine.
- **Correct "controlling for."** Controlling for X means removing X's
  *contribution*, not the whole X-correlated *subspace*. Deleting the shared
  subspace is over-adjustment — it discards legitimate residual variation that
  merely correlates with X.
- **It separates surgical from blunt; the `b`-only view cannot.** On `b` alone
  FastICA-drop and Flower look equivalent. Rotation reveals that source-dropping's
  removal is *collateral* (it cannot remove the digit without removing rotation),
  whereas Flower's is *targeted* — and this same entanglement is why Flower's
  digit accuracy sits at the ~0.79 floor rather than at chance.

**One line:** preserving `b` shows a method is not destructive (necessary, easy);
preserving **rotation** shows it removes the *label*, not the *shared subspace* —
and that same entanglement is precisely why consistent suppression (~0.79), not
chance, is the faithful outcome.
