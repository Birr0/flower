# Reproducing R2 Table 1 (ICA / iVAE / Flower at matched redshift removal)

All commands run from `examples/spectra/` with the venv active
(`source ../../.venv/bin/activate`). CPU only — these probe *existing* embeddings
and train one small iVAE; no GPU, no re-embedding.

## The run behind the table

```bash
python ivae_sweep_paper_eval.py --n-k 9
```

Defaults, all recorded in `params.json`: `--spender spender_I`, `--epochs 30`,
`--n-train 200000`, `--n-filter 300000`, `--seed 42`,
`--outdir ivae_sweep_paper_eval_results`.

Wall time ~20-25 min (24 rows x 4 MLP fits at 200k train rows).

## Why this script and not `ivae_sweep.py`

`ivae_sweep.py` and the paper's spectra benchmark (`embedding_benchmark.py`) measure
the same quantity — redshift recoverable from the raw `spender_I` latents — and
disagree, because they differ in three ways at once:

| | `ivae_sweep.py` | `embedding_benchmark.py` (the paper) |
|---|---|---|
| MLP probe | `(64, 32)`, `max_iter=300` | `(64, 64)`, `max_iter=1000` |
| Train rows | 40k, random subsample | 200k, first rows after masking |
| Row mask | every physical target valid (43.1k test rows) | redshift validity only; per-target mask applied at probe time (52.8k test rows for `z`) |
| Raw `z` R² | 0.555 | 0.711 [0.703, 0.718] |

A weaker probe understates recoverable redshift in *every* row, not just the raw
one, so quoting ICA and Flower on the first while the paper quotes the second is not
a like-for-like comparison. `ivae_sweep_paper_eval.py` imports `load_run` from
`embedding_benchmark` and its `(64, 64)`/1000 probe, so the data path, masking, row
truncation and read-out are the paper's. The method machinery (FastICA, iVAE, the
residual constructions, the source-dropping sweep) is imported unchanged from
`ivae_sweep`.

Alignment note: `load_run` masks on `isfinite(x).all(axis=1)`, computed per split, so
`orig`/`cond`/`uncond` need not keep the same rows. The condition and targets are
re-aligned to each split's own mask — sharing one split's `z` across all three would
silently misalign the Flower rows.

Standardisation note: every representation is standardised (scaler fit on train)
inside `evaluate`, immediately before probing. iVAE sources are raw encoder `mu`
values with per-dimension std spanning 2.9-98.7 and means as far as -37; probing those
unscaled makes `MLPRegressor` fail to converge erratically depending on which columns
survive the drop sweep. `superseded_unstandardised/` holds that broken run as a record.
`ivae_sweep.py` and `ivae_sweep_matched_probe.py` still probe iVAE sources unscaled, so
their iVAE rows carry the same defect.

Monotonicity gate: `drop_top_k_dependent` ranks columns once and drops the top `k`, so
the kept sets are nested and R² must be non-increasing in `k`. `check_monotonic` flags
any rise beyond 0.02 to stdout *and* into `summary.txt`. A run whose summary says
FAILED must not be quoted.

## Reference points to check against

| Quantity | Source | Value |
|---|---|---|
| Raw `z`, 2-Layer | `flower_vs_frozen_base_results/results.csv` (`flower,orig,z,2-Layer`) | 0.7108 [0.7025, 0.7184] |
| Flower seed `z`, 2-Layer | same file (`flower,cond,z,2-Layer`) | 0.0846 [0.0797, 0.0899] |

The `Raw` and `Flower-cond` rows of this run should land inside those intervals. If
they do not, stop — the protocol has drifted from the benchmark and the table cannot
be quoted against the paper.

## Building the table

`z` removal is `z_r2_linreg` / `z_r2_mlp`; preservation is `logSFR_r2_mlp` and
`A_v_r2_mlp` (`logM*_r2_mlp` is reported in the response's prose, not the table).

Rows: `source=Raw, method=none`; `source=Flower-cond, method=embedding`; and for each
baseline the `method=residA` row whose `z_r2_mlp` is closest to Flower's, which is
also the `k` (components dropped) quoted in the row label. The `--n-k 9` grid sweeps
`k = 1..9` so the matched point is not interpolated across a gap.

```bash
python - <<'EOF'
import pandas as pd
d = pd.read_csv("ivae_sweep_paper_eval_results/results.csv")
flower = d[d.source == "Flower-cond"].iloc[0].z_r2_mlp
cols = ["source", "k", "z_r2_linreg", "z_r2_mlp", "logSFR_r2_mlp", "A_v_r2_mlp", "logM*_r2_mlp"]
print(d[d.source.isin(["Raw", "Flower-cond"])][cols].to_string(index=False))
for src in ["FastICA", "iVAE"]:
    a = d[(d.source == src) & (d.method == "residA")].copy()
    a["gap"] = (a.z_r2_mlp - flower).abs()
    print(a.sort_values("gap").head(3)[cols + ["gap"]].to_string(index=False))
EOF
```

## Related runs (do not mix into one table)

- `ivae_sweep_results/` — the original sweep, weak probe, 40k rows, common-rows mask.
  Kept so previously reported numbers stay reproducible.
- `ivae_sweep_matched_probe_results/` — paper's probe but the *common-rows* mask
  (43,098 test rows). Raw `z` = 0.607, matching `matched_flower_vs_frozen_base_results`
  (0.6090 [0.6024, 0.6154]). Use this if quoting `z` and the physical targets on
  identical rows; use `ivae_sweep_paper_eval_results/` to compare against the paper.

## Outputs

`results.csv`, `summary.txt`, `tradeoff.png`, `params.json`,
`ivae_sweep_paper_eval.log` (full stdout, including per-split row counts).
