"""Is the learned redshift relation the selection boundary, or the mean above it?

``latent_zero_limit.py`` shows that zeroing the conditioned latents collapses each
fitted expression to a function of redshift, and that the resulting curves agree with
one another. It does not show *what that curve is*. The draft calls it the selection
function, but the two are different objects and the distinction matters:

- The **selection function** is a boundary: M_lim(z), the least massive galaxy still
  above the survey's flux limit at redshift z. It is the lower edge of the observed
  distribution.
- The model learns a **conditional mean**: E[log M* | z, latent = mean]. Because the
  observed mass distribution at each z is the true mass function *truncated* below
  M_lim(z), this mean rises with z mechanically -- not because galaxies are more massive
  at higher redshift, but because the low-mass ones are missing.

So the learned curve should be a smoothed image of the boundary sitting some way above
it, and that is what this figure tests. It draws, on the observed density:

1. the X=0 reductions of the cond+z fronts (the learned curve);
2. a running mean of observed log M* in redshift bins (the empirical conditional mean);
3. the completeness limit implied by SDSS's flux limit.

The prediction is (1) ~ (2), both a roughly constant offset above (3).

**The limit curve.** For the main-sample limit r < 17.77,

    M_r,lim(z) = 17.77 - DM(z) - K_r(z)
    log M*_lim(z) = log (M/L)_r + 0.4 (M_r,sun - M_r,lim(z))
                  = C + 0.4 (DM(z) + K_r(z))

with everything galaxy-dependent absorbed into ``C`` -- the mass-to-light ratio, which
varies by type and which we decline to model. So the *shape* is pure cosmology and ``C``
is a single constant offset, fit here to the observed 5th-percentile envelope. K_r(z) is
taken from the NYU VAGC's ``KCORRECT_r``, which is small over this range (0.043 at
z=0.05 to 0.153 at z=0.15).

**Caveat that must not be lost.** The symbolic regression targets ``LGM_FIB_P50``, a
*fibre* mass, while a flux limit constrains *total* luminosity. The fixed 3" fibre
samples a larger physical fraction of a galaxy at higher redshift, so the fibre-to-total
ratio is itself z-dependent and the limit curve above is only approximate for this
target. The empirical 5th percentile is drawn alongside precisely so the cosmological
shape can be checked rather than assumed.

Usage:
    python mass_limit_overlay.py
    python mass_limit_overlay.py --nbins 40 --envelope-pct 5
"""

from __future__ import annotations

import argparse
import os

import matplotlib as mpl

mpl.use("Agg")

from itertools import pairwise

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy as sp
from astropy.cosmology import FlatLambdaCDM
from datasets import load_from_disk
from latent_zero_limit import Z, load_fronts, reduce_to_redshift, to_sympy
from val_rescore import build_merged

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a897f"
SURFACE = "#fcfcfb"
LEARNED = "#2a78d6"
MEAN = "#0d366b"
LIMIT = "#eb6834"
ENVELOPE = "#8a897f"

FLUX_LIMIT_R = 17.77  # SDSS main galaxy sample, Strauss et al. 2002


def kcorrection_model(vol_root: str) -> np.poly1d:
    """Median K_r(z) from the NYU VAGC, as a quadratic in z.

    Taken from the volume-limited files because they are the only ones carrying
    ``KCORRECT_r``. Those are magnitude-selected and so skew red, which biases the
    k-correction slightly; over z < 0.3 the whole term is < 0.2 mag, so its effect on
    the limit curve is small next to the constant offset we are fitting anyway.
    """
    frames = []
    for cut in ("0.050", "0.075", "0.100", "0.125", "0.150"):
        path = f"{vol_root}/z={cut}"
        if os.path.isdir(path):
            frames.append(load_from_disk(path)["test"].to_pandas()[["z", "KCORRECT_r"]])
    if not frames:
        print("[warn] no KCORRECT_r available; using K_r(z) = 0")
        return np.poly1d([0.0])

    df = pd.concat(frames).drop_duplicates()
    bins = np.linspace(df["z"].min(), df["z"].max(), 12)
    mid = 0.5 * (bins[1:] + bins[:-1])
    med = [
        df.loc[(df["z"] >= lo) & (df["z"] < hi), "KCORRECT_r"].median()
        for lo, hi in pairwise(bins)
    ]
    ok = np.isfinite(med)
    return np.poly1d(np.polyfit(mid[ok], np.asarray(med)[ok], 2))


def limit_curve(z: np.ndarray, kcorr: np.poly1d, cosmo: FlatLambdaCDM) -> np.ndarray:
    """Shape of log M*_lim(z), up to the additive constant C."""
    return 0.4 * (cosmo.distmod(z).value + kcorr(z))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    data_root = os.environ.get("DATA_ROOT", "/home/birr0/local_data")
    parser.add_argument("--feature", default="LGM_FIB_P50")
    parser.add_argument(
        "--embeddings",
        default=f"{data_root}/sdss_II/spender_I_flow_v2/spender_I_flow_v2/embeddings/7655991_0",
    )
    parser.add_argument("--galspec", default=f"{data_root}/galSpecExtra-dr8.fits")
    parser.add_argument("--specgals-home", default=f"{data_root}/sdss")
    parser.add_argument(
        "--vol-root", default=f"{data_root}/vol_limited_embeddings_7655991_0"
    )
    parser.add_argument(
        "--nbins", type=int, default=30, help="redshift bins for the summaries"
    )
    parser.add_argument("--envelope-pct", type=float, default=5.0)
    parser.add_argument("--zmin", type=float, default=0.02)
    parser.add_argument("--zmax", type=float, default=0.3)
    parser.add_argument("--outdir", default="mass_limit_overlay_results")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    stem = f"{args.outdir}/mass_limit_overlay"

    merged = build_merged(args.embeddings, args.galspec, args.specgals_home)
    test = merged["test"]
    test = test[np.isfinite(test[args.feature]) & (test[args.feature] != -9999.0)]
    z_obs = test["z_x"].to_numpy()
    m_obs = test[args.feature].to_numpy()
    keep = (z_obs >= args.zmin) & (z_obs <= args.zmax)
    z_obs, m_obs = z_obs[keep], m_obs[keep]
    print(f"{len(z_obs)} test-split galaxies with valid {args.feature}")

    # --- binned summaries of the observed distribution ---------------------------
    edges = np.linspace(args.zmin, args.zmax, args.nbins + 1)
    centres = 0.5 * (edges[1:] + edges[:-1])
    idx = np.digitize(z_obs, edges) - 1
    running_mean, envelope, counts = [], [], []
    for b in range(args.nbins):
        sel = idx == b
        counts.append(int(sel.sum()))
        if sel.sum() < 20:
            running_mean.append(np.nan)
            envelope.append(np.nan)
            continue
        running_mean.append(float(np.mean(m_obs[sel])))
        envelope.append(float(np.percentile(m_obs[sel], args.envelope_pct)))
    running_mean = np.asarray(running_mean)
    envelope = np.asarray(envelope)

    # --- completeness limit, constant fit to the envelope ------------------------
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    kcorr = kcorrection_model(args.vol_root)
    shape = limit_curve(centres, kcorr, cosmo)
    good = np.isfinite(envelope)
    offset = float(np.mean(envelope[good] - shape[good]))
    limit = shape + offset
    residual = envelope[good] - limit[good]
    print(f"fitted offset C = {offset:.3f} dex")
    print(
        f"envelope vs cosmological shape: rms {np.sqrt(np.mean(residual**2)):.3f} dex,"
        f" range {residual.min():+.3f}..{residual.max():+.3f}"
    )
    gap = running_mean[good] - limit[good]
    print(f"mean-minus-limit offset: {gap.mean():.3f} dex (sd {gap.std():.3f})")

    # --- learned curves: the X=0 reductions --------------------------------------
    fronts = load_fronts(args.feature, "cond+z")
    z_grid = np.linspace(args.zmin, args.zmax, 200)
    curves = []
    for row in fronts.itertuples():
        try:
            reduced = reduce_to_redshift(to_sympy(row.Equation), 6)
            if (
                reduced.has(sp.zoo, sp.oo, sp.nan, sp.I)
                or Z not in reduced.free_symbols
            ):
                continue
            fn = sp.lambdify(Z, reduced, "numpy")
            vals = np.asarray(fn(z_grid), dtype=float) * np.ones_like(z_grid)
            if np.all(np.isfinite(vals)):
                curves.append(vals)
        except (TypeError, ValueError, KeyError, ZeroDivisionError, sp.SympifyError):
            continue
    curves = np.asarray(curves)
    print(f"{len(curves)} usable X=0 reductions")

    pd.DataFrame(
        {
            "z": centres,
            "n": counts,
            "running_mean": running_mean,
            f"pct{args.envelope_pct:g}": envelope,
            "limit": limit,
            "limit_shape": shape,
        }
    ).to_csv(f"{stem}.csv", index=False)

    # --- figure -------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.5, 6.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.hist2d(z_obs, m_obs, bins=[120, 120], cmap="Blues", cmin=1, alpha=0.75, zorder=1)

    for i, c in enumerate(curves):
        ax.plot(
            z_grid,
            c,
            color=LEARNED,
            linewidth=1.0,
            alpha=0.30,
            zorder=3,
            label="X=0 reductions (learned)" if i == 0 else None,
        )
    ax.plot(
        z_grid,
        np.median(curves, axis=0),
        color=LEARNED,
        linewidth=2.6,
        zorder=4,
        label="median learned curve",
    )
    ax.plot(
        centres,
        running_mean,
        color=MEAN,
        linewidth=2.4,
        linestyle=(0, (5, 2)),
        zorder=5,
        label="observed conditional mean",
    )
    ax.plot(
        centres,
        envelope,
        color=ENVELOPE,
        linewidth=1.8,
        linestyle=(0, (1, 2)),
        zorder=5,
        label=f"observed {args.envelope_pct:g}th percentile",
    )
    ax.plot(
        centres,
        limit,
        color=LIMIT,
        linewidth=2.4,
        zorder=6,
        label=f"completeness limit, C={offset:.2f}",
    )

    ax.set_xlabel("redshift $z$", fontsize=10.5, color=INK_SECONDARY)
    ax.set_ylabel(
        f"$\\log_{{10}}$ M$_*$  ({args.feature})", fontsize=10.5, color=INK_SECONDARY
    )
    ax.set_title(
        "The learned relation is the conditional mean, not the selection boundary"
        "\nboundary shape is $0.4\\,(DM(z)+K_r(z))$ from the $r<17.77$ limit;"
        " only the offset is fitted",
        fontsize=12,
        color=INK,
        loc="left",
        pad=12,
        linespacing=1.5,
    )
    ax.set_xlim(args.zmin, args.zmax)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, loc="lower right")
    ax.grid(color="#e6e5e0", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(INK_MUTED)
    ax.spines["bottom"].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, length=3, labelsize=9)

    fig.text(
        0.008,
        -0.02,
        "Target is a *fibre* mass while the flux limit constrains total luminosity, so"
        " the limit curve is approximate here;\nthe empirical percentile is drawn so"
        " the cosmological shape can be checked rather than assumed."
        f"  Mean sits {gap.mean():.2f} dex above the limit.",
        fontsize=8,
        color=INK_MUTED,
        va="top",
        linespacing=1.5,
    )
    fig.tight_layout()
    fig.savefig(f"{stem}.png", dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {stem}.csv")
    print(f"wrote {stem}.png")


if __name__ == "__main__":
    main()
