# Per-component redshift dependence, and what conditioning does to it

Companion to `component_table.csv` / `.png` (produced by `component_table.py`).
Reproduction and verification checks: `REPRODUCE.md` beside this file.

The question: our removal/preservation comparison reports Flower at 10 dimensions against
FastICA and iVAE at 2 and 1, and the selection rule behind those numbers was never written
down. This measures both — what each individual coordinate carries, and what the deletion
rule actually ranks by.

## The selection rule

`drop_top_k_dependent(..., dependence="continuous")` ranks by **absolute Pearson
correlation** with `z` (`ica.py:159` → `_abs_correlation` → `dependence.abs_pearson`),
computed once on the training split and reused for every `k`, so retained sets are nested.
The categorical branch (correlation ratio, η²) is used only for cMNIST's discrete digit;
no spectra row goes through it.

**This handicaps the baselines and we should say so.** The ranking is linear, so a
component carrying `z` nonlinearly sorts low and is deleted last. The `eta_z` column
quantifies the gap: raw rank 8 reads |ρ| 0.023 vs η 0.196, FastICA rank 8 reads 0.004 vs
0.118 (30×) — against an η null of ~0.025. A nonlinear ranking could only improve the
baselines.

## Raw and Flower coordinates correspond — measured, not assumed

Hungarian matching on the |ρ| matrix between the raw Spender embedding and Flower's seed
returns the **identity permutation**: raw *j* → seed *j* for all ten, mean matched
|ρ| = 0.691 (range 0.583 at idx 8 to 0.867 at idx 7). Structurally this is expected — for
the CondOT path the flow map is `x₀ = x₁ − ∫v dt`, identity minus a displacement in data
space, so coordinates keep their labels unless the displacement rotates them.

It is a strong correspondence, **not an identity**: each seed coordinate is a blend of
~2–3 raw ones (participation ratio 2.58 over raw coordinates). Quote the mean |ρ| whenever
the paired table is shown, so the pairing reads as measured rather than assumed.

FastICA and iVAE indices have **no** such correspondence — ICA component order and sign are
arbitrary, and permutation is precisely what iVAE's identifiability is "up to". Their
indices are meaningful only within one fit.

## Result: conditioning flattens the redshift profile

Paired by coordinate index, |ρ| with `z`:

| idx | Raw | Flower | |
|---|---|---|---|
| 6 | 0.488 | 0.014 | 35× down |
| 8 | 0.460 | 0.036 | 13× |
| 0 | 0.339 | 0.044 | 7.7× |
| 2 | 0.306 | 0.011 | 27× |
| 4 | 0.272 | 0.047 | 5.7× |
| 1 | 0.191 | 0.026 | 7.2× |
| 5 | 0.182 | 0.039 | 4.6× |
| 3 | 0.065 | 0.002 | 35× |
| 9 | 0.023 | 0.062 | **up** |
| 7 | 0.001 | 0.032 | **up** |

Flower drives every coordinate to a **common floor** rather than shrinking proportionally:
0.488 and 0.182 both land around 0.01–0.04. The peak falls 0.488 → 0.062 and the profile
flattens from a structured head-and-tail into a near-uniform band.

**The two coordinates that carried no redshift go up** — idx 7 (0.001 → 0.032) and idx 9
(0.023 → 0.062), against a max-null of 0.010. Real but tiny; nothing approaches the 0.488
that was removed. The honest reading is that the flow removes the bulk and smears a small
residual, rather than relocating the information. Report this rather than only the
reductions: a flat near-null profile in arbitrary order is the signature of removal, and
the small rises are part of it.

## Which physics sits where

`z` is **co-located with stellar mass and anti-located with SFR and dust** in the raw
basis. The top-`z` coordinates (6, 8, 0, 2, 4) are also the top-logM\* coordinates
(0.663, 0.785, 0.585, 0.598, 0.571), while the two lowest-`z` coordinates (9, 7) carry
logSFR 0.485/0.522 and A_v 0.444/0.554 and almost no mass.

Two consequences:

1. **Flower's logM\* loss (0.819 → 0.270) is forced, not a weakness.** Mass occupies the
   redshift axes, so any method that removes `z` removes it. This is the per-component
   version of the Malmquist argument.
2. **The preservation claim should rest on logSFR and A_v**, which do not share axes with
   `z`. At matched removal FastICA retains 0.109 / 0.065 against Flower's 0.594 / 0.532.

Note the corollary, which the earlier `redshift_motivation_note.md` phrasing ("the physics
curve rides on top of the z bars") overstates: a z-free physics-carrying coordinate *does*
exist in every basis. The defensible claim is not that deletion destroys the physics, but
that no ordering by z-dependence reaches low `z` without taking the physics-heavy
deletions — `z` is spread over ~3.8 axes and no small subset suffices.

## Caveats

- Run at `--n-train 40000`; the paper-protocol sweep (`ivae_sweep_paper_eval_results/`)
  uses 200k and its own fits. **Ranks here do not index drops there.**
- The iVAE arm is the `condition_encoder=True` model, whose encoder takes `z` as input and
  can re-encode it into every source. Its rows are an upper bound on iVAE's z-dependence,
  not a property of nonlinear ICA. FastICA is the clean baseline.
