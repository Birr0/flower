"""What physical quantity is latent X9? (FINDINGS F2/F2b)

The variable-addition waterfall picks **X9 first** in every arm, on both mass targets
and on both the flux-limited and volume-limited samples -- ahead of redshift in the
latter. It is the dominant mass axis of the representation and we have never known
what it is. This correlates it, and every other latent coordinate, against the
physical quantities in the merged catalogue.

**Indexing.** pyoperon prints variables 1-indexed, and ``fit_sym_fn`` builds the
cond+z matrix as [scaled cond (10 dims), raw z]. So **X9 is ``cond`` dimension 8**
(0-indexed) and X11 is redshift. Getting this wrong silently identifies the wrong
axis, so the mapping is asserted rather than assumed: X11's correlation with z must be
1.

**What to expect, and what would be surprising.** ``spender`` normalises each spectrum
by its median flux in 5300--5850 A rest frame, so the latent is *scale-free* -- it can
encode shape (colours, line ratios, D4000, specific SFR) but has no access to a
luminosity. A strong X9 correlation with an *intensive* quantity is the expected
result; one with an *extensive* quantity (total mass, total SFR, size) would have to
come indirectly through the mass-metallicity or mass-size relations, and is worth
flagging rather than reporting flatly. This is the same extensive/intensive split that
``examples/spectra/extensive_intensive_probe.py`` tests directly.

**Two correlation measures**, because a monotone one alone would miss folded
structure:

- ``spearman`` -- rank correlation, sign preserved. The headline number.
- ``eta`` -- correlation ratio: bin the latent, take the quantity's conditional means,
  and report the variance explained. Catches non-monotone dependence a rank
  correlation misses. Always >= |spearman| in magnitude, so a large gap between them
  means the relation is folded rather than monotone.

Both are computed on the test split only, per quantity, over rows where that quantity
is finite and not the ``-9999`` sentinel.

**Run it on both samples.** Two claims that looked clean on the flux-limited sample
(parsimony, F8c; separability, F9) reversed under volume-limiting. Anything asserted
about X9 should be checked the same way before it is written up -- hence
``--embeddings-cut``.

Usage:
    python identify_x9.py
    python identify_x9.py --embeddings-cut 0.150
    python identify_x9.py --latent X3 --top 15
"""

from __future__ import annotations

import argparse
import os

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from val_rescore import build_matrices, build_merged

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a897f"
SURFACE = "#fcfcfb"

SENTINEL = -9999.0
N_ETA_BINS = 40
N_LATENT = 10


def derived_quantities(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Physical quantities, with the intensive/extensive split made explicit.

    Colours and line ratios are formed here rather than taken raw: a colour is the
    scale-free quantity a median-normalised spectrum could plausibly encode, whereas the
    magnitudes it is built from are not.
    """

    def col(name: str) -> np.ndarray:
        v = df[name].to_numpy(dtype=float) if name in df else np.full(len(df), np.nan)
        return np.where(v == SENTINEL, np.nan, v)

    def logratio(a: str, b: str) -> np.ndarray:
        num, den = col(a), col(b)
        with np.errstate(all="ignore"):
            out = np.log10(num / den)
        return np.where((num > 0) & (den > 0) & np.isfinite(out), out, np.nan)

    q: dict[str, np.ndarray] = {}

    # --- intensive: shape, ratios, no luminosity needed -------------------------
    q["D4000 (age)"] = col("d4000")
    q["u-g colour"] = col("modelMag_u") - col("modelMag_g")
    q["g-r colour"] = col("modelMag_g") - col("modelMag_r")
    q["r-i colour"] = col("modelMag_r") - col("modelMag_i")
    q["i-z colour"] = col("modelMag_i") - col("modelMag_z")
    q["log NII/Ha"] = logratio("nii_6584_flux", "h_alpha_flux")
    q["log OIII/Hb"] = logratio("oiii_5007_flux", "h_beta_flux")
    q["Balmer Ha/Hb"] = logratio("h_alpha_flux", "h_beta_flux")
    q["gas metallicity OH_P50"] = col("OH_P50")
    q["specific SFR (tot)"] = col("SPECSFR_TOT_P50")
    q["specific SFR (fib)"] = col("SPECSFR_FIB_P50")
    q["concentration R90/R50"] = np.where(
        col("petroR50_r") > 0, col("petroR90_r") / col("petroR50_r"), np.nan
    )
    q["BPT class"] = col("bptclass")

    # --- extensive: need a distance, so a scale-free latent should not carry them --
    q["log M* (total)"] = col("lgm_tot_p50")
    q["log M* (fibre)"] = col("LGM_FIB_P50")
    q["log SFR (total)"] = col("SFR_TOT_P50")
    q["velocity dispersion"] = col("velDisp")
    q["size petroR50_r"] = col("petroR50_r")
    q["r magnitude"] = col("petroMag_r")

    # --- the control ------------------------------------------------------------
    q["redshift"] = col("z_x")
    return q


def correlation_ratio(x: np.ndarray, y: np.ndarray, n_bins: int = N_ETA_BINS) -> float:
    """Variance of y explained by its conditional mean given binned x."""
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return float("nan")
    idx = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
    fitted = np.empty_like(y)
    for b in np.unique(idx):
        m = idx == b
        fitted[m] = y[m].mean()
    ss_tot = np.sum((y - y.mean()) ** 2)
    if ss_tot <= 0:
        return float("nan")
    return float(np.sqrt(max(0.0, 1.0 - np.sum((y - fitted) ** 2) / ss_tot)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    data_root = os.environ.get("DATA_ROOT", "/home/birr0/local_data")
    parser.add_argument("--feature", default="lgm_tot_p50")
    parser.add_argument("--latent", default="X9")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument(
        "--embeddings",
        default=f"{data_root}/sdss_II/spender_I_flow_v2/spender_I_flow_v2/embeddings/7655991_0",
    )
    parser.add_argument("--embeddings-cut", default=None)
    parser.add_argument("--galspec", default=f"{data_root}/galSpecExtra-dr8.fits")
    parser.add_argument("--specgals-home", default=f"{data_root}/sdss")
    parser.add_argument("--outdir", default="identify_x9_results")
    args = parser.parse_args()

    root = args.embeddings
    tag = "full"
    if args.embeddings_cut:
        root = (
            f"{data_root}/vol_limited_embeddings_7655991_0_allsplits"
            f"/z={args.embeddings_cut}"
        )
        tag = f"vollim_z{args.embeddings_cut}"

    os.makedirs(args.outdir, exist_ok=True)
    stem = f"{args.outdir}/identify_{args.latent.lower()}_{tag}"

    merged = build_merged(root, args.galspec, args.specgals_home)
    mats = build_matrices(merged, args.feature, "cond+z")
    X = mats["X_test"]

    # Rows of X_test correspond to the test frame after fit_sym_fn's own filters.
    test = merged["test"].dropna(subset=[args.feature])
    test = test[test[args.feature] != SENTINEL].reset_index(drop=True)
    if len(test) != len(X):
        message = f"row misalignment: {len(test)} frame rows vs {len(X)} matrix rows"
        raise SystemExit(message)

    quantities = derived_quantities(test)

    # Assert the 1-indexed mapping rather than trusting it.
    z_check = abs(spearmanr(X[:, N_LATENT], quantities["redshift"]).statistic)
    print(f"index check: X{N_LATENT + 1} vs redshift, |spearman| = {z_check:.4f}")
    if z_check < 0.999:
        message = "X11 is not redshift -- the column mapping has changed"
        raise SystemExit(message)

    rows = []
    for j in range(N_LATENT + 1):
        name = f"X{j + 1}"
        v = X[:, j]
        for qname, q in quantities.items():
            ok = np.isfinite(v) & np.isfinite(q)
            if ok.sum() < 500:
                continue
            rows.append(
                {
                    "latent": name,
                    "quantity": qname,
                    "n": int(ok.sum()),
                    "spearman": float(spearmanr(v[ok], q[ok]).statistic),
                    "eta": correlation_ratio(v[ok], q[ok]),
                }
            )
    out = pd.DataFrame(rows)
    out["abs_spearman"] = out["spearman"].abs()
    out.to_csv(f"{stem}.csv", index=False)
    print(f"wrote {stem}.csv")

    target = out[out["latent"] == args.latent].sort_values("eta", ascending=False)
    print(f"\n=== {args.latent} ({tag}), top {args.top} by eta ===")
    print(
        target.head(args.top)[["quantity", "spearman", "eta", "n"]].to_string(
            index=False, float_format=lambda v: f"{v:7.4f}"
        )
    )

    print("\n=== which latent each quantity aligns with best (by eta) ===")
    best = out.loc[out.groupby("quantity")["eta"].idxmax()]
    print(
        best.sort_values("eta", ascending=False)[
            ["quantity", "latent", "spearman", "eta"]
        ].to_string(index=False, float_format=lambda v: f"{v:7.4f}")
    )

    top = target.head(args.top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.0, 0.42 * len(top) + 1.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK_MUTED)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    colors = ["#2a78d6" if s >= 0 else "#eb6834" for s in top["spearman"]]
    ax.barh(top["quantity"], top["eta"], color=colors, height=0.62)
    for y, (e, sp) in enumerate(zip(top["eta"], top["spearman"], strict=True)):
        ax.text(
            e + 0.008, y, f"{sp:+.2f}", va="center", fontsize=8.5, color=INK_SECONDARY
        )
    ax.set_xlabel(
        "correlation ratio $\\eta$  (bar), signed Spearman (label)",
        fontsize=9.5,
        color=INK_SECONDARY,
    )
    ax.set_title(
        f"What is {args.latent}?  ({tag.replace('_', ' ')})",
        fontsize=11,
        color=INK,
        loc="left",
    )
    fig.text(
        0.0,
        -0.10,
        "Blue = positive Spearman, orange = negative. $\\eta$ catches non-monotone"
        " dependence a rank correlation misses, so a large gap\nbetween bar and label"
        " means the relation is folded. spender normalises each spectrum by its median"
        " flux, so the latent is\nscale-free: intensive quantities are the expected"
        " correlates.",
        fontsize=7.5,
        color=INK_MUTED,
    )
    fig.tight_layout()
    fig.savefig(f"{stem}.png", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"\nwrote {stem}.png")


if __name__ == "__main__":
    main()
