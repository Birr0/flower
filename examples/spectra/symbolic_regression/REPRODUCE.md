# Reproducing the symbolic-regression results

**Only the scripts are in version control.** The fronts, figures, CSVs and notebooks they
produce are not — they are large, they are regenerable, and committing them would put binary
data in the history. This file is how you get them back.

Everything below writes into `job_results_*/` and `*_results/` directories beside the scripts,
none of which are tracked. Run from *this* directory: the scripts import each other by module
name (`from val_rescore import build_merged`), so the working directory matters.

## Environment

```bash
cd /path/to/flower
uv sync
uv run --with astroml --with astropy --with pyoperon --with scipy python <script>
```

`astroml`, `astropy`, `pyoperon` and `scipy` are **not** in `pyproject.toml` and `uv sync`
removes them, hence the `--with` flags. Versions used for the recorded results:

| package | version |
|---|---|
| pyoperon | **0.4.0** (PyPI) |
| scikit-learn | 1.8.0 |
| numpy | 2.4.4 |
| pandas | 3.0.2 |
| sympy | 1.14.0 |
| scipy | 1.17.1 |
| astropy | 7.2.0 |
| astroML | 1.0.2 |

**The pyoperon version matters.** The original `LGM_FIB_P50` fronts were produced on the cluster
with an *unrecorded* build. Anything regenerated here uses 0.4.0. Search behaviour and constant
optimisation can differ between builds, so a regenerated front is not guaranteed to reproduce a
published number exactly — compare within one build, not across.

`.env` must set `DATA_ROOT`.

## Data

| what | where |
|---|---|
| full-sample embeddings | `$DATA_ROOT/sdss_II/spender_I_flow_v2/spender_I_flow_v2/embeddings/7655991_0` (note the **doubled** directory) |
| volume-limited embeddings, all splits | built by `../make_volume_limited.py` |
| volume-limited embeddings, test only | `$DATA_ROOT/vol_limited_embeddings_7655991_0/z=*` — pre-existing, the validation reference |
| MPA-JHU | `$DATA_ROOT/sdss/SDSSspecgalsDR8_{1,2}.fit.gz`, fetched by astroML on first run (~105 MB) |
| galSpecExtra | `$DATA_ROOT/galSpecExtra-dr8.fits` |
| VAC_ID ↔ object_id crossmatch | HF `Birr001/VACG_raw_cross_match` |
| NYU VAGC k-correct | HF `Birr001/kcorrect_VAC` |

Model run is **`7655991_0`** — the frozen-base spectra model. In the Flower/NeurIPS line the same
id is the E1 frozen-base *ablation*; same id, opposite meaning.

## Order

Steps 1–3 fit models and take time. Everything after re-reads stored fronts and is fast.

### 1. Volume-limited embeddings with train/val/test splits

The pre-existing volume-limited cuts have only a `test` split, which the symbolic regression
cannot use: `utils.fit_sym_fn` fits its scaler on **train** and `val_rescore.py` selects on
**val**.

```bash
cd ..                       # examples/spectra
uv run --with astropy python make_volume_limited.py
cd symbolic_regression
```

Writes `$DATA_ROOT/vol_limited_embeddings_7655991_0_allsplits/z={cut}/{train,val,test}/`.
**Check the gate before continuing:** it must report recall 1.0000 and precision 1.0000 at all
five cuts against the stored test-only cuts. Anything less means an input has moved.

### 2. The fronts

```bash
# original target, on the cluster (paths hardcoded in train_logm.sh)
sbatch train_logm.sh

# aperture-corrected target, locally -- 15 cells, ~5 min each
uv run --with astroml --with astropy --with pyoperon python run_total_mass.py

# volume-limited, all five arms -- 15 cells, ~80 min
uv run --with astroml --with astropy --with pyoperon python run_volume_limited.py

# the orig+z control arm, which fit_sym_fn does not otherwise provide
uv run --with astroml --with astropy --with pyoperon python run_orig_z.py
uv run --with astroml --with astropy --with pyoperon python run_orig_z.py \
  --feature lgm_tot_p50 --cut 0.150
```

`run_total_mass.py`, `run_volume_limited.py` and `run_orig_z.py` all call `utils.fit_sym_fn`
**unmodified** — same search, same 10,000-row subsample per seed, same scaling, same MLP/LR
references. Only the inputs differ. Do not edit `utils.py` or `train_logm.py` to add an arm;
that would invalidate the existing fronts.

### 3. Selection on val, and the constant-precision repair

```bash
uv run --with astroml --with astropy python val_rescore.py
uv run --with astroml --with astropy python val_rescore.py --feature lgm_tot_p50
uv run --with astroml --with astropy python val_rescore.py --refit    # least-squares refit
```

The refit's `Refit_Kind` column separates *recovers* / *improves* / *degrades*; **do not pool
them** — only *recovers* is a faithful repair.

### 4. Analyses over the stored fronts

```bash
# X=0 reductions
uv run --with sympy python latent_zero_limit.py
uv run --with sympy python latent_zero_limit.py \
  --feature vollim_z0.150_lgm_tot_p50 --embed-type cond+z \
  --zmax 0.150 --outdir latent_zero_limit_vollim_results

# variable-addition waterfall
uv run python variable_additions.py --feature lgm_tot_p50 --form waterfall \
  --embed-type cond+z --outdir variable_additions_lgmtot_results
uv run python variable_additions.py --feature vollim_z0.150_lgm_tot_p50 --form waterfall \
  --embed-type cond+z --outdir variable_additions_vollim_results

# frontier comparison figure
uv run python plot_vollim_frontier.py

# additive separability, treatment and control
uv run --with astroml --with astropy python separability.py
uv run --with astroml --with astropy python separability.py --feature lgm_tot_p50
uv run --with astroml --with astropy python separability.py \
  --feature vollim_z0.150_lgm_tot_p50 --embeddings-cut 0.150
uv run --with astroml --with astropy python separability.py \
  --feature origz_LGM_FIB_P50 --target LGM_FIB_P50 --swap-cond-for-orig
uv run --with astroml --with astropy python separability.py \
  --feature origz_vollim_z0.150_lgm_tot_p50 --target lgm_tot_p50 \
  --embeddings-cut 0.150 --swap-cond-for-orig

# what the latent coordinates are
uv run --with astroml --with astropy --with scipy python identify_x9.py
uv run --with astroml --with astropy --with scipy python identify_x9.py --embeddings-cut 0.150

# completeness-limit overlay
uv run --with astropy --with sympy python mass_limit_overlay.py
```

## Things that silently give wrong answers

- **`latent_zero_limit.py --zmax`** defaults to 0.3, the full sample's fit range. On a
  volume-limited cut that extrapolates well past the data. Pass the cut.
- **`--outdir`** must differ per run of `latent_zero_limit.py` and `variable_additions.py`, or a
  later run overwrites an earlier figure.
- **Means vs medians over seeds.** `plot_vollim_frontier.py` uses medians and only lengths where
  ≥2 seeds contributed, deliberately: volume-limited `orig` seed 42 at L=22 has Train_R² 0.573
  but Test_R² −27.48, and L=22 is the only length that seed alone reached, so a mean-based
  frontier reports the broken value and invents a result.
- **Variable indexing.** pyoperon prints 1-indexed, and `fit_sym_fn` builds
  `[scaled cond (10 dims), raw z]`, so `X9` is `cond` dimension 8 and `X11` is redshift.
  `identify_x9.py` asserts this rather than assuming it.
- **The `orig+z` control** is built by overwriting the `cond` column with `orig`
  (`run_orig_z.swap_cond_for_orig`), because `fit_sym_fn` branches only on `cond+z` and `z`.
  `separability.py --swap-cond-for-orig` applies the same swap when rebuilding the matrix — pass
  it, or the control is scored against the wrong latent.
- **`utils.py` sets `max_length=25`** yet fronts contain lengths 26–29. This is pyoperon's
  length accounting, not a misconfiguration.

## Interpretation

Findings, retractions and the numbers themselves live in `papers/wwdc_symbolic_regression/`
(`FINDINGS.md`, `TODO.md`, `PROVENANCE.md`) — a local working directory, not tracked here.
Several results that look clean on the flux-limited sample **reverse** under volume-limiting;
read the retractions there before citing anything regenerated from these scripts.
