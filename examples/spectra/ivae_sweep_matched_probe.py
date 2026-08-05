"""ICA-baseline sweep re-run with the *paper's* probe capacity (R2 Table 1).

``ivae_sweep.py`` reads out with an MLP of ``(64, 32)`` for ``max_iter=300`` on
40k training rows, but the spectra benchmark the paper reports
(``embedding_benchmark.py``: ``ARCHITECTURES["2-Layer"] = (64, 64)``,
``max_iter=1000``, 200k rows) is strictly stronger. The two disagree on the same
quantity — redshift recoverable from the raw spender_I latents — reading
R2 = 0.555 under the weak probe against 0.711 [0.703, 0.718] under the paper's.

A weaker probe understates recoverable redshift in *every* row, so the ICA-vs-Flower
comparison in the rebuttal table is measured on a different instrument from the one
the paper uses. This script re-runs the identical sweep with the probe matched, so
the rebuttal numbers sit on the paper's scale.

Everything else — the FastICA/iVAE fits, the residual constructions, the k grid,
the direct residualisers — is imported unchanged from ``ivae_sweep`` so that only
the read-out differs. Run from this directory:

    python ivae_sweep_matched_probe.py [--n-train 200000] [--spender spender_I]
"""

import sys

import ivae_sweep

# The paper's spectra probe (embedding_benchmark.py:75, "2-Layer").
ivae_sweep.PROBE_MLP = {"hidden_layer_sizes": (64, 64), "max_iter": 1000}

DEFAULTS = {
    "--n-train": "200000",
    "--outdir": "ivae_sweep_matched_probe_results",
}

if __name__ == "__main__":
    for flag, value in DEFAULTS.items():
        if flag not in sys.argv:
            sys.argv += [flag, value]
    ivae_sweep.main()
