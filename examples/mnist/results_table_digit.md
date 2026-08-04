# RGB-MNIST digit experiment — results table

digit = condition removed (lower = better; chance 0.10). colour b (separable) & rotation (entangled) = preserved (higher). Embedding dim = 64.

| Method | digit logreg ↓ | digit MLP ↓ | b R² ↑ | rotation R² ↑ | mean preserv ↑ | axes dropped |
|---|---|---|---|---|---|---|
| Raw (orig) | 0.947 | 0.979 | 0.990 | 0.924 | 0.957 | 0 |
| FastICA residB | 0.114 | 0.858 | 0.984 | 0.699 | 0.841 | 0 |
| FastICA residA (matched digit, k=7) | 0.580 | 0.815 | 0.975 | 0.408 | 0.691 | 7 |
| FastICA residA (strong, k=21) | 0.138 | 0.155 | 0.948 | -0.169 | 0.390 | 21 |
| iVAE-cond residB | 0.522 | 0.999 | 0.990 | 0.904 | 0.947 | 0 |
| iVAE-cond residA (matched digit, k=51) | 0.532 | 0.763 | 0.916 | 0.706 | 0.811 | 51 |
| Flower cond | 0.748 | 0.792 | 0.960 | 0.611 | 0.785 | 0 |
| Flower uncond | 0.803 | 0.923 | 0.970 | 0.791 | 0.880 | 0 |
