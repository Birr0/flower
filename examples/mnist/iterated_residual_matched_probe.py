"""Iterated residualisation re-run with the *notebook's* probe budget (R2 Q3).

``iterated_residual.py`` reads out with ``MLPClassifier``/``MLPRegressor`` at
``hidden_layer_sizes=(64, 32)``, ``max_iter=300`` (inherited from
``ivae_sweep._digit_acc`` / ``_b_r2``), but the cMNIST benchmark the paper reports
(``benchmark.ipynb``) uses the same architecture at ``max_iter=1000``. The
architectures match, so any gap is early stopping rather than capacity — but it
runs in the direction that *understates* recoverable digit, i.e. the direction
that flatters the residualisation baseline the reviewer asked about.

This script re-runs the identical experiment with the probe budget matched to the
notebook, so the rebuttal numbers sit on the paper's scale. The residualisers,
the iteration loop, the data loading and the standardisation are imported
unchanged from ``iterated_residual`` so that *only* the read-out differs.

Note the MLP *residualiser* arm still fits at ``max_iter=300`` — that is the
baseline under test, not the measurement instrument, and changing it would
confound the comparison with the run this one is diffed against.

Run from this directory (needs ``DATA_ROOT`` and the rotation CSVs):

    python iterated_residual_matched_probe.py [--n-iter 5]
"""

import sys

import iterated_residual
from ivae_sweep import RANDOM_STATE
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.neural_network import MLPClassifier, MLPRegressor

# The paper's cMNIST probe (benchmark.ipynb: (64, 32) at max_iter=1000).
PROBE_MLP = {"hidden_layer_sizes": (64, 32), "max_iter": 1000}


def _digit_acc(x_tr, x_te, y_tr, y_te, kind):
    if kind == "logreg":
        clf = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    else:
        clf = MLPClassifier(random_state=RANDOM_STATE, **PROBE_MLP)
    clf.fit(x_tr, y_tr)
    return accuracy_score(y_te, clf.predict(x_te))


def _b_r2(x_tr, x_te, y_tr, y_te, kind):
    if kind == "linreg":
        reg = LinearRegression()
    else:
        reg = MLPRegressor(random_state=RANDOM_STATE, **PROBE_MLP)
    reg.fit(x_tr, y_tr)
    return r2_score(y_te, reg.predict(x_te))


# ``iterated_residual`` binds these at import time, so patch them there.
iterated_residual._digit_acc = _digit_acc
iterated_residual._b_r2 = _b_r2

DEFAULTS = {"--outdir": "iterated_residual_matched_probe_results"}

if __name__ == "__main__":
    for flag, value in DEFAULTS.items():
        if flag not in sys.argv:
            sys.argv += [flag, value]
    iterated_residual.main()
