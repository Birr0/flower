"""Rebuild the volume-limited cuts over embeddings, with train/val/test splits.

The stored cuts in ``$DATA_ROOT/vol_limited_embeddings_7655991_0/z=*`` carry only a
``test`` split, which is enough for probing (``vol_limited_probe.py``) but not for the
symbolic regression: ``utils.fit_sym_fn`` fits its ``StandardScaler`` on the full
post-filter *train* split before drawing the 10k subsample, and ``val_rescore.py``
selects expression length on *val*. Re-running the SR on volume-limited data -- the
paper-native test of whether the recovered mass-redshift relation is the selection
function -- therefore needs all three.

This is the author's original volume-limiting script with one substitution: it joins
the **flow embeddings** (``orig``/``cond``/``uncond``) rather than the raw spectra,
since the symbolic regression reads embeddings. The selection itself is unchanged, and
the two catalogue inputs come from the Hub because neither is on disk any more:

- ``Birr001/VACG_raw_cross_match`` -- the VAC_ID <-> object_id positional crossmatch,
  already split train/val/test (the local ``VACG_raw_cross_match_dataset`` is gone).
- ``Birr001/kcorrect_VAC`` -- the NYU VAGC k-correct catalogue supplying ``ABSMAG_r``,
  ``KCORRECT_r`` and ``MASS``.

**The cut.** For redshift cut ``zc`` on the grid ``linspace(0.001, 0.3, 300)``:

    z > 0   and   z <= zc   and   ABSMAG_r <= 17.77 - DM(zc)

with ``DM`` from ``FlatLambdaCDM(H0=100, Om0=0.3)`` -- h=1, matching the VAGC's
``ABSMAG`` convention -- and no K-correction term in the threshold, since ``ABSMAG_r``
is already k-corrected. 17.77 is the SDSS main sample limit (Strauss et al. 2002). Cuts
0.050/0.075/0.100/0.125/0.150 are grid indices 49/74/99/124/149.

Rows whose two redshifts disagree by more than ``Z_TOL`` (1e-4) are dropped, as in the
original: ``Z`` from the k-correct VAC against ``z`` carried on the embedding.

**Validation gate.** Test-split membership is compared against the stored cuts, which
were built by the same selection from the same catalogues. Agreement should be
essentially exact; anything below ~0.99 means an input has moved and the output should
not be used.

Writes ``{outroot}/z={cut}/{split}/000.parquet`` -- the layout
``val_rescore.load_embeddings`` reads, so ``build_merged(f"{outroot}/z={cut}", ...)``
works unchanged. Deliberately does *not* write to the stored cuts, and does not push to
the Hub; the original script's ``push_to_hub`` is left out on purpose.

Usage:
    python make_volume_limited.py
    python make_volume_limited.py --cuts 0.100 0.150 --dry-run
"""

from __future__ import annotations

import argparse
import glob
import os

import matplotlib as mpl

mpl.use("Agg")

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
from datasets import load_dataset, load_from_disk

FLUX_LIMIT_R = 17.77  # SDSS main galaxy sample, Strauss et al. 2002
COSMO = FlatLambdaCDM(H0=100, Om0=0.3)  # h=1, the VAGC's ABSMAG convention
Z_GRID = np.linspace(0.001, 0.3, 300)
Z_TOL = 1e-4
SPLITS = ["train", "val", "test"]
Z_CUTS = ["0.050", "0.075", "0.100", "0.125", "0.150"]

CROSS_MATCH_REPO = "Birr001/VACG_raw_cross_match"
KCORRECT_REPO = "Birr001/kcorrect_VAC"

# Columns carried through, mirroring the stored volume-limited embedding files.
VAC_COLS = [
    "VAC_ID",
    "RA",
    "DEC",
    "MASS",
    "INTSFH",
    "METS",
    "B300",
    "B1000",
    "KCORRECT_r",
    "ABSMAG_r",
    "MTOL_r",
]
# ``id`` is required: val_rescore.build_merged re-derives its merge key from it.
EMB_COLS = ["id", "orig", "cond", "uncond", "z", "mask_ratio", "ra", "dec", "BESTOBJID"]

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a897f"
SURFACE = "#fcfcfb"
STORED = "#2a78d6"
RECON = "#eb6834"
THIRD = "#1baf7a"


def m_lim(cut: float) -> float:
    """Faintest ABSMAG_r still above the flux limit at the cut redshift."""
    d_l = COSMO.luminosity_distance(cut).to(u.pc)
    dm = 5 * np.log10(d_l / (10.0 * u.pc))
    return float(FLUX_LIMIT_R - dm)


def load_embeddings(root: str) -> dict[str, pd.DataFrame]:
    """Embedding parquet per split, keyed by object_id for the crossmatch join."""
    frames = {}
    for split in SPLITS:
        files = sorted(glob.glob(f"{root}/{split}/*.parquet"))
        if not files:
            message = f"no parquet for split {split!r} under {root!r}"
            raise SystemExit(message)
        df = pd.concat([pd.read_parquet(f) for f in files]).reset_index(drop=True)
        df["object_id"] = df["id"].astype(str).str.extract(r"(\d+)")[0]
        frames[split] = df
        print(f"  {split}: {len(df)} rows")
    return frames


def load_catalogues() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """VAC_ID <-> object_id crossmatch per split, plus the k-correct VAC."""
    print(f"  crossmatch: {CROSS_MATCH_REPO}")
    xm_ds = load_dataset(CROSS_MATCH_REPO)
    xmatch = {}
    for split in SPLITS:
        df = xm_ds[split].to_pandas()
        df["object_id"] = df["object_id"].astype(str).str.extract(r"(\d+)")[0]
        xmatch[split] = df
        print(f"    {split}: {len(df)} rows")

    print(f"  k-correct VAC: {KCORRECT_REPO}")
    kcorr = load_dataset(KCORRECT_REPO)["full"].to_pandas()
    print(f"    full: {len(kcorr)} rows")
    return xmatch, kcorr


def merge_split(
    emb: pd.DataFrame, xmatch: pd.DataFrame, kcorr: pd.DataFrame
) -> pd.DataFrame:
    """embeddings -> crossmatch (object_id) -> k-correct VAC (VAC_ID)."""
    merged = pd.merge(emb, xmatch, on="object_id", how="inner")
    return pd.merge(kcorr, merged, on="VAC_ID", suffixes=["_kcorr", "_raw"])


def apply_cut(df: pd.DataFrame, cut: float) -> pd.DataFrame:
    """z > 0, z <= cut, ABSMAG_r above the limit, and the two redshifts agreeing.

    The redshift box is applied to the *spectroscopic* ``z`` carried on the embedding,
    not the k-correct VAC's ``Z``. The two agree to ``Z_TOL`` by construction here, so
    the choice only matters for rows straddling the boundary -- exactly one across the
    five cuts: object_id 471876337305413632 at z<=0.125, where Z=0.125035 but
    z=0.124956. The stored cuts keep it, and ``z`` is what the flow conditions on
    (``src/conf/data/sdss.yaml``), so raw is both the reproducing and the consistent
    choice. Cutting on ``Z`` instead reproduces every other row identically.
    """
    z_kcorr = df["Z"].to_numpy()
    z_raw = df["z"].to_numpy()
    absmag_r = np.asarray(df["ABSMAG_r"]).reshape(-1)

    mask = (
        (z_raw > 0)
        & (z_raw <= cut)
        & (absmag_r <= m_lim(cut))
        & np.isclose(z_kcorr, z_raw, atol=Z_TOL, rtol=0)
    )
    return df[mask].reset_index(drop=True)


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    wanted = [*VAC_COLS, *EMB_COLS, "object_id", "separation_arcsec"]
    keep = [c for c in wanted if c in df]
    missing = {*VAC_COLS, *EMB_COLS} - set(df.columns)
    if missing:
        print(f"    [warn] absent from the merge, not written: {sorted(missing)}")
    return df[keep]


def validate(test_df: pd.DataFrame, vol_root: str, cut: str) -> dict:
    """Compare test-split membership against the stored cut."""
    path = f"{vol_root}/z={cut}"
    if not os.path.isdir(path):
        return {}
    stored = set(load_from_disk(path)["test"].to_pandas()["object_id"].astype("int64"))
    got = set(test_df["object_id"].astype("int64"))
    tp = len(stored & got)
    return {
        "z_cut": float(cut),
        "absmag_limit": m_lim(float(cut)),
        "n_stored": len(stored),
        "n_new_test": len(got),
        "n_true_positive": tp,
        "n_missed": len(stored - got),
        "n_extra": len(got - stored),
        "recall": tp / max(len(stored), 1),
        "precision": tp / max(len(got), 1),
    }


def plot(report: pd.DataFrame, counts: pd.DataFrame, stem: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), facecolor=SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(INK_MUTED)
        ax.tick_params(colors=INK_SECONDARY, labelsize=9)

    ax = axes[0]
    if len(report):
        ax.plot(report["z_cut"], report["recall"], "o-", color=STORED, label="recall")
        ax.plot(
            report["z_cut"], report["precision"], "s-", color=RECON, label="precision"
        )
        ax.set_ylim(min(0.9, report[["recall", "precision"]].min().min() - 0.02), 1.005)
    ax.set_xlabel("volume-limited cut  $z \\leq$", fontsize=9.5, color=INK_SECONDARY)
    ax.set_ylabel("agreement with stored test rows", fontsize=9.5, color=INK_SECONDARY)
    ax.set_title(
        "Validation gate against the stored cuts", fontsize=10.5, color=INK, loc="left"
    )
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY)

    ax = axes[1]
    width = 0.006
    for i, split in enumerate(SPLITS):
        ax.bar(
            counts["z_cut"] + (i - 1) * width,
            counts[split],
            width=width,
            label=split,
            color=[STORED, RECON, THIRD][i],
        )
    ax.set_xlabel("volume-limited cut  $z \\leq$", fontsize=9.5, color=INK_SECONDARY)
    ax.set_ylabel("rows written", fontsize=9.5, color=INK_SECONDARY)
    ax.set_title("New splits per cut", fontsize=10.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY)

    fig.text(
        0.01,
        -0.05,
        "Cut: z > 0, z <= zc, ABSMAG_r <= 17.77 - DM(zc),"
        " FlatLambdaCDM(H0=100, Om0=0.3), and |Z_kcorr - z| <= 1e-4."
        "\nEmbeddings joined to Birr001/VACG_raw_cross_match on object_id,"
        " then to Birr001/kcorrect_VAC on VAC_ID.",
        fontsize=7.5,
        color=INK_MUTED,
    )
    fig.tight_layout()
    fig.savefig(f"{stem}.png", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {stem}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    data_root = os.environ.get("DATA_ROOT", "/home/birr0/local_data")
    parser.add_argument(
        "--embeddings",
        default=f"{data_root}/sdss_II/spender_I_flow_v2/spender_I_flow_v2/embeddings/7655991_0",
        help="full-sample embeddings, all three splits (note the doubled directory)",
    )
    parser.add_argument(
        "--vol-root",
        default=f"{data_root}/vol_limited_embeddings_7655991_0",
        help="stored test-only cuts, used only to validate",
    )
    parser.add_argument(
        "--outroot",
        default=f"{data_root}/vol_limited_embeddings_7655991_0_allsplits",
        help="destination; deliberately NOT the stored cuts, which are the reference",
    )
    parser.add_argument("--cuts", nargs="+", default=Z_CUTS)
    parser.add_argument("--outdir", default="make_volume_limited_results")
    parser.add_argument(
        "--dry-run", action="store_true", help="validate and report, write no parquet"
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    stem = f"{args.outdir}/make_volume_limited"

    print("loading embeddings")
    emb = load_embeddings(args.embeddings)

    print("loading catalogues")
    xmatch, kcorr = load_catalogues()

    print("merging")
    merged = {}
    for split in SPLITS:
        merged[split] = merge_split(emb[split], xmatch[split], kcorr)
        pct = 100 * len(merged[split]) / len(emb[split])
        print(f"  {split}: {len(merged[split])} rows after both joins ({pct:.1f}%)")

    rows, counts = [], []
    for cut in args.cuts:
        cut_frames = {s: apply_cut(df, float(cut)) for s, df in merged.items()}
        n = {s: len(df) for s, df in cut_frames.items()}
        print(
            f"\nz<={cut} (ABSMAG_r <= {m_lim(float(cut)):.3f}): "
            + ", ".join(f"{s} {v}" for s, v in n.items())
        )
        counts.append({"z_cut": float(cut), **n})

        report = validate(cut_frames["test"], args.vol_root, cut)
        if report:
            print(
                f"  gate: recall {report['recall']:.4f} precision "
                f"{report['precision']:.4f} ({report['n_missed']} missed, "
                f"{report['n_extra']} extra vs {report['n_stored']} stored)"
            )
            rows.append(report)

        if args.dry_run:
            continue
        for split, df in cut_frames.items():
            dest = f"{args.outroot}/z={cut}/{split}"
            os.makedirs(dest, exist_ok=True)
            select_columns(df).to_parquet(f"{dest}/000.parquet", index=False)
        print(f"  wrote {args.outroot}/z={cut}/{{train,val,test}}/000.parquet")

    report_df = pd.DataFrame(rows)
    counts_df = pd.DataFrame(counts)
    report_df.to_csv(f"{stem}.csv", index=False)
    print(f"\nwrote {stem}.csv")
    if len(report_df):
        print(report_df.to_string(index=False))
    plot(report_df, counts_df, stem)

    if len(report_df) and report_df["recall"].min() < 0.99:
        print(
            "\n[warn] recall below 0.99 against the stored cuts -- an input has moved. "
            "Do not use these files until the difference is understood."
        )


if __name__ == "__main__":
    main()
