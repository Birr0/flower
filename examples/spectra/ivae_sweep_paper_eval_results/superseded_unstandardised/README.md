# Superseded run — unstandardised iVAE sources

Kept as the record of a probe-fitting bug, not as a result. **Do not quote these numbers.**

The iVAE arm was probed on raw encoder `mu` values (per-dim std 2.9-98.7, means to -37)
while every other row was standardised. `MLPRegressor` failed to converge erratically
depending on which columns survived the drop sweep, giving R^2 curves that *rise* as
coordinates are deleted — impossible for nested feature sets, e.g. `iVAE residA`
z_r2_mlp 0.052 at k=5 against 0.441 at k=6.

The `Raw` (0.711) and `Flower-cond` (0.110) rows here are unaffected by the bug and did
match the paper's benchmark; they are reproduced in the corrected run alongside a
monotonicity gate.
