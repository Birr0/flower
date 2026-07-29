# spender_II redshift experiment — results table

z (redshift) = condition removed (lower = better). logM*/logSFR/A_v = physics preserved (higher). Embedding dim = 6.

| Method | z linreg ↓ | z MLP ↓ | logM* ↑ | logSFR ↑ | A_v ↑ | mean phys ↑ | axes dropped |
|---|---|---|---|---|---|---|---|
| Raw (orig) | 0.283 | 0.514 | 0.804 | 0.632 | 0.535 | 0.657 | 0 |
| Resid-linear | -0.000 | 0.764 | 0.782 | 0.640 | 0.545 | 0.656 | 0 |
| Resid-mlp | 0.013 | 0.720 | 0.736 | 0.643 | 0.537 | 0.639 | 0 |
| Resid-rf | -0.001 | -0.454 | -0.024 | 0.278 | 0.238 | 0.164 | 0 |
| FastICA residB | -0.000 | 0.821 | 0.797 | 0.645 | 0.548 | 0.663 | 0 |
| iVAE residB | 0.010 | 0.602 | 0.764 | 0.655 | 0.486 | 0.635 | 0 |
| FastICA residA (matched z, k=4) | 0.003 | 0.117 | 0.241 | 0.339 | 0.379 | 0.320 | 4 |
| iVAE residA (matched z, k=3) | 0.087 | 0.104 | 0.447 | 0.238 | 0.106 | 0.264 | 3 |
| Flower cond | 0.059 | 0.208 | 0.453 | 0.620 | 0.503 | 0.526 | 0 |
| Flower uncond | 0.357 | 0.563 | 0.808 | 0.648 | 0.561 | 0.672 | 0 |
