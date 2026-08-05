# Reproducing the two per-component tables (R1 round-2 response)

Table 1 = per-component `|Pearson|` with redshift, the statistic the drop sweep actually
ranks by. Table 2 = the full drop sweep, every `k`, physical targets excluding stellar mass.

All commands run from `examples/spectra/` with the venv active
(`source ../../.venv/bin/activate`, or prefix each with `uv run`). CPU only — these probe
*existing* embeddings and train one small iVAE; no GPU, no re-embedding.

## Prerequisites

- `.env` with `DATA_ROOT` set. The embeddings read are
  `$DATA_ROOT/sdss_II/spender_I_flow_v2/spender_I_flow_v2/embeddings/7526202_0`
  (the `SPENDER_MAP["spender_I"]` entry in `ivae_sweep.py`).
- Physical targets come from the HF dataset `Birr001/spectra_catalog`, index-aligned.
  Unauthenticated access works but warns; set `HF_TOKEN` to silence it.
- **`src/flower/evaluation/dependence.py` is currently untracked.** `component_table.py`
  imports `abs_pearson` and `correlation_ratio` from it, so on a fresh clone this script
  fails at import. Check with `git ls-files src/flower/evaluation/` before assuming a
  checkout can run it.
- `component_table.py` itself is new and untracked at time of writing.

## Table 1 — per-component `|rho|` with z

```bash
python component_table.py --spender spender_I --outdir component_table_results
```

Defaults, all recorded in `params.json`: `--epochs 30`, `--n-train 40000`,
`--n-filter 300000`, `--seed 42`.

Wall time **~65 s**, almost all of it data loading — the per-component panels fit no
probes, only closed-form correlations. (Contrast `ivae_sweep_paper_eval.py` at ~20-25 min,
which is dominated by MLP probes.)

Writes into `component_table_results/`: `component_table.csv` (the table, one row per
component per representation), `component_table.png`, `params.json` (args, bin count,
per-representation permutation nulls), `component_table.log` (captured stdout).

### Verifying the output

| Check | Expected |
|---|---|
| stdout header | `train (40000, 10); z range [0.010, 0.300]` |
| Raw, largest `rho_z` | 0.488 |
| FastICA, largest `rho_z` | 0.363 |
| iVAE, largest `rho_z` | 0.502 |
| **Flower, largest `rho_z`** | **0.062** (every component below the raw basis's 8th-ranked one) |
| Permutation nulls (`params.json`) | `rho_null` 0.007-0.015, `eta_null` 0.025-0.029 |

Independent cross-check against a different script computing from the same fit config:

```bash
python redshift_motivation.py --spender spender_I \
    --outdir /tmp/rm_check --results-csv ivae_sweep_results/results.csv
```

should print `FastICA 3.84, iVAE 4.41` for the participation ratio, matching the 3.8 / 4.4
recorded in `ivae_sweep_results/redshift_motivation_note.md`. Use a scratch `--outdir` so it
does not overwrite the committed-to-disk figure in `ivae_sweep_results/`.

Determinism: FastICA takes `random_state=seed` and the iVAE `torch.manual_seed(seed)`, so
repeat runs on the same machine and library versions reproduce exactly. Expect small
last-digit drift across a different BLAS or torch build.

## Table 2 — the drop sweep

**No compute.** These values are already on disk in
`ivae_sweep_paper_eval_results/results.csv`, produced by the run documented in
`ivae_sweep_paper_eval_results/REPRODUCE.md` (paper protocol: `(64, 64)` probe,
`max_iter=1000`, 200k train rows). To re-emit the table:

```bash
python - <<'PY'
import csv
rows = list(csv.DictReader(open("ivae_sweep_paper_eval_results/results.csv")))
cols = ["z_r2_mlp", "logSFR_r2_mlp", "A_v_r2_mlp"]

def get(src, meth, k):
    return next(r for r in rows
                if r["source"] == src and r["method"] == meth and int(r["k"]) == k)

def line(label, r):
    print(f"| {label} | {r['n_dims']} | "
          + " | ".join(f"{float(r[c]):.3f}" for c in cols) + " |")

print("| Method | Dims | z | logSFR | A_v |")
print("|---|---|---|---|---|")
line("Raw representation (Spender I)", get("Raw", "none", 0))
line("Flower (seed)", get("Flower-cond", "embedding", 0))
for src in ("FastICA", "iVAE"):
    for k in range(1, 10):
        line(f"{src}, drop {k}", get(src, "residA", k))
PY
```

To regenerate the CSV itself rather than re-read it, see
`ivae_sweep_paper_eval_results/REPRODUCE.md` (`python ivae_sweep_paper_eval.py --n-k 9`,
~20-25 min).

### Verifying the output

| Check | Expected |
|---|---|
| Raw row | `z` 0.711, logSFR 0.669, A_v 0.591 |
| Flower row | `z` 0.108, logSFR 0.594, A_v 0.532, 10 dims |
| FastICA row matched to Flower's removal | drop 8, `z` 0.101, logSFR 0.109, A_v 0.065, 2 dims |
| iVAE closest approach to Flower's removal | drop 9, `z` 0.186 — never reaches 0.108, at 1 dim |
| Monotonicity | `z_r2_mlp` non-increasing in `k` within each method (nested sets) |

That last check currently **fails marginally for iVAE**: `z` rises 0.600 → 0.619 from
`k=2` to `k=3`, and there are a few `+0.002`-scale rises in the physical targets. The sets
are nested, so an ideal probe cannot improve when features are removed. The magnitudes are
at MLP fit-noise scale and this is the standardised run, but per `CLAUDE.md` a non-monotone
drop curve should be treated as a scaling bug until shown otherwise — it has not been run
down.

## Caveats that matter when quoting these together

1. **Different fits.** Table 1 is 40k train rows, Table 2 is the 200k paper protocol, and
   each fits its own FastICA/iVAE. ICA component order and sign are arbitrary, so
   **rank *j* in Table 1 is not drop *j* in Table 2** — do not invite a reader to index one
   by the other. To put both on one fit, re-run Table 1 with `--n-train 200000`.
2. **The iVAE arm is confounded.** `ivae_sweep.train_ivae` hardcodes
   `condition_encoder=True` (`src/flower/models/ivae.py:137` concatenates `u` onto the
   encoder input), so its encoder takes `z` as an input and can re-encode it into every
   source. Read the iVAE rows as an upper bound on its z-dependence, not as a property of
   nonlinear ICA. FastICA has no such conditioning and is the clean baseline.
3. **The ranking is linear.** `drop_top_k_dependent(..., dependence="continuous")` ranks by
   `|Pearson|` only, so components carrying `z` nonlinearly sort low and are deleted last.
   The `eta_z` column of `component_table.csv` quantifies the gap (e.g. raw rank 8:
   `|rho|` 0.023 vs `eta` 0.196, against an `eta` null of ~0.025). This handicaps the
   baselines; a nonlinear ranking could only improve them.
4. **Stellar mass is omitted from Table 2 deliberately**, not for convenience: it is
   strongly redshift-entangled through the flux limit and is therefore removed by every
   method including Flower (0.819 → 0.270). Its full column is in
   `ivae_sweep_paper_eval_results/results.csv` as `logM*_r2_mlp` if a reader asks.
