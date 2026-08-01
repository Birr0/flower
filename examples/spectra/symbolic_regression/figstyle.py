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

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import scienceplots  # noqa: F401  -- registers the styles with matplotlib

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


def use_science() -> str:
    """Apply the science style, preferring real LaTeX.

    Returns the style actually used.
    """
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
