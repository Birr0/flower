↑ [Back to README](README.md)

# Method

## 1. The residual-conditioning loop
Given the frozen `spender` latent `z` and a current factor set `S` (a bank subset):
1. **Condition-out** — evaluate the amortized CFM at `y_S` (bank with all but `S` dropped).
2. **Measure residual** — (a) per-dim / PCA spread of `p(z | y_S)` vs `p(z)` to locate directions
   that stay broad; (b) MI-gain of each candidate not yet in `S`.
3. **Nominate (quantitative)** — correlate broad residual directions against a **held-out bank** of
   physical quantities (line indices, flux ratios, catalog cols) → ranked candidates.
4. **Name + engineer (LLM)** — agent reasons over nominations (optional literature/crossmatch tools),
   hypothesizes a physical factor, and **emits code** computing it.
5. **Fold in** — `S ← S ∪ {factor}`; free inner-loop step if already in the bank, else §3.3.
   Terminate when best candidate MI-gain falls below threshold.

## 2. Scoring gate
Selection signal is **MI-gain** `ΔLL(f | S) = E[log p(z | y_{S∪f}) − log p(z | y_S)]` from the
pmi_flux CFG estimator (hard prerequisite); a held-out ΔLL from the same `log_prob` wiring is the
fallback. Used *comparatively* (nats differences) to cancel bias; a **shuffled-`y` control** must
drive the gain to ≈0.

## 3. Wiring into Flower
- **3.1 Feature bank** (`flower.data.sdss` extension) — pre-compute a wide superset of conditioners
  (spectral indices, ratios, PCA/UMAP coords, catalog cols); split into a *training bank* and a
  *held-out naming bank* (§1.3). Catalog contract (`get_conditional_len`) unchanged.
- **3.2 Amortized CFG** (`spender_I_discovery` config) — reuse `spender_I_flow`'s CFG training,
  trained once; `drop_variables` at eval realizes any `y_S` for free, so the inner loop needs **no
  retrain** (`n_layers` stays in `model.yaml`).
- **3.3 Agent orchestrator** (`flower.discovery.agent`) — proposes `S`, calls the §2 gate and §1.3
  correlation, invokes LLM naming; for a *new* feature it **halts for human approval** before
  computing it, extending the bank, and re-launching training via the Hydra entrypoint.
