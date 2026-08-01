"""Shared figure style and palette for the paper figures.

Applies the ``scienceplots`` ``science`` style so the figures match the typography
of the journal they are going into, and keeps the categorical palette in one place
rather than duplicated across the three figure scripts.

**LaTeX rendering is attempted, then fallen back on.** The ``science`` style sets
``text.usetex``, which needs a working LaTeX *and* ``dvipng`` for raster output. On
this machine ``latex`` and ``pdflatex`` are present but ``dvipng`` is not, so the
plain style raises at render time. ``use_science()`` therefore tries ``["science"]``
first and drops to ``["science", "no-latex"]`` if it fails, reporting which it used.
Install ``dvipng`` to get true LaTeX typesetting without changing any script.

**Keep figure text LaTeX-safe** for that reason: no bare unicode arrows, minus signs
or inequalities in labels, since those render under ``no-latex`` but break under
``usetex``. Use ``->``, ``-`` and ``$\\leq$``.

The palette is the same fixed assignment used everywhere, so an arm keeps its colour
across figures. Validated as a 5-slot categorical set on a white surface: worst
adjacent CVD separation dE 9.2 (deutan), 27.6 normal vision. Two steps sit below 3:1
contrast against white, which obliges visible relief -- every series in every figure
is direct-labelled as well as carrying a legend entry, so identity never rests on
colour alone.
"""

from __future__ import annotations

import os
import shutil

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import scienceplots  # noqa: F401  -- registers the styles with matplotlib

# Portable TeX Live, as used by the notebooks (equation_explorer.ipynb,
# pareto_fronts.ipynb). The system LaTeX on this machine cannot drive usetex -- it has
# latex and gs but no dvipng, no type1ec.sty and no cm-super -- so this tree is what
# makes real LaTeX typesetting possible. Restore it and the figures pick it up with no
# code change.
TEXLIVE_BIN = "texlive_store/texlive/bin/x86_64-linux"

ARM_COLOUR = {
    "cond+z": "#2a78d6",
    "orig": "#eb6834",
    "uncond": "#1baf7a",
    "z": "#8a4fbf",
    "cond": "#c8a415",
    "orig+z": "#eb6834",
}
ARM_ORDER = ["cond+z", "orig", "uncond", "z", "cond"]

# Human-readable names for figure text. Raw column names carry underscores, which
# render as literal backslashes under no-latex and as subscripts under usetex --
# neither is wanted in a caption.
TARGET_LABEL = {
    "lgm_tot_p50": "MPA-JHU total stellar mass",
    "LGM_FIB_P50": "MPA-JHU fibre stellar mass",
}


def target_label(feature: str) -> str:
    """Caption-safe name for a target column."""
    return TARGET_LABEL.get(feature, feature.replace("_", " "))


INK = "#0b0b0b"
INK_SECONDARY = "#3d3d3d"
INK_MUTED = "#6b6b6b"
SURFACE = "white"


def _add_portable_texlive() -> str | None:
    """Prepend the portable TeX Live to PATH if it is present. Returns the path used."""
    data_root = os.environ.get("DATA_ROOT")
    if not data_root:
        return None
    candidate = os.path.join(data_root, TEXLIVE_BIN)
    if not os.path.isdir(candidate):
        return None
    if candidate not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = candidate + os.pathsep + os.environ.get("PATH", "")
    return candidate


def latex_available() -> tuple[bool, list[str]]:
    """Whether usetex can actually render, and what is missing if not.

    ``latex`` alone is not enough: matplotlib shells out to ``dvipng`` for raster
    output, and the CM fonts need ``type1ec.sty``/``cm-super`` to be scalable.
    """
    missing = [b for b in ("latex", "dvipng") if shutil.which(b) is None]
    return (not missing), missing


def use_science() -> str:
    """Apply the science style, preferring real LaTeX.

    Returns the style actually used.
    """
    tex = _add_portable_texlive()
    if tex:
        print(f"  using portable TeX Live at {tex}")
    ok, missing = latex_available()
    if not ok:
        print(f"  LaTeX unavailable ({', '.join(missing)} missing) -- falling back")

    for style in (["science"], ["science", "no-latex"]):
        try:
            plt.style.use(style)
            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1])
            fig.canvas.draw()
            plt.close(fig)
        except Exception:  # any backend failure means fall through to no-latex
            plt.close("all")
            continue
        else:
            return "+".join(style)
    message = "neither 'science' nor 'science,no-latex' could render"
    raise SystemExit(message)


def tidy(ax, grid_axis: str | None = "both") -> None:
    """Light grid over the science style's box, and muted tick labels."""
    if grid_axis:
        ax.grid(
            True, axis=grid_axis, color=INK_MUTED, alpha=0.18, linewidth=0.5, zorder=0
        )
        ax.set_axisbelow(True)
    ax.tick_params(labelsize=8)
