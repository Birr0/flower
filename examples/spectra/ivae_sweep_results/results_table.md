# spender_I redshift experiment — results table

z (redshift) = condition removed (lower = better). logM*/logSFR/A_v = physics preserved (higher).
Effective # z-carrying axes (participation ratio): FastICA 3.8, iVAE 4.4 (of 10).

| Method | z linreg ↓ | z MLP ↓ | logM* ↑ | logSFR ↑ | A_v ↑ | mean phys ↑ | axes dropped |
|---|---|---|---|---|---|---|---|
| Raw (orig) | 0.404 | 0.555 | 0.809 | 0.659 | 0.573 | 0.680 | 0 |
| Resid-linear | -0.000 | 0.607 | 0.730 | 0.663 | 0.566 | 0.653 | 0 |
| Resid-mlp | 0.001 | 0.602 | 0.693 | 0.663 | 0.571 | 0.643 | 0 |
| Resid-rf | -0.000 | -0.405 | -0.214 | 0.022 | -0.040 | -0.077 | 0 |
| FastICA residB | -0.000 | 0.607 | 0.728 | 0.666 | 0.567 | 0.654 | 0 |
| iVAE residB | 0.003 | 0.434 | 0.618 | 0.653 | 0.559 | 0.610 | 0 |
| FastICA residA (matched z, k=7) | 0.002 | 0.029 | 0.146 | 0.106 | 0.212 | 0.155 | 7 |
| iVAE residA (matched z, k=6) | 0.018 | 0.027 | 0.135 | 0.335 | 0.451 | 0.307 | 6 |
| Flower cond | 0.011 | 0.028 | 0.250 | 0.577 | 0.512 | 0.447 | 0 |
| Flower uncond | 0.404 | 0.534 | 0.797 | 0.648 | 0.567 | 0.671 | 0 |
