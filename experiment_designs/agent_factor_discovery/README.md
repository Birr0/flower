# An Autonomous Agent for Residual-Conditioning Factor Discovery over Flower

*Experiment design — working system on Flower. Draft for discussion.*
*Scope agreed: factor discovery (engineer + name) on spectra (`spender_I_flow`) · MI-gain gate
(pmi_flux, ΔLL fallback) · amortized CFG (no per-candidate retrain) · hybrid/staged autonomy ·
novel-finding case study + calibration control.*

## Abstract

Flower learns a conditional density `p(z | y)` over a frozen encoder latent, conditioned on a
catalog `y` of known factors. The interesting object is the **residual**: latent structure no known
factor explains. We propose an **autonomous agent that closes the loop** around it — condition-out
known factors, measure what remains, propose and *name* a new physical factor to explain it, compute
that feature, fold it back in, repeat. We call this the **residual-conditioning discovery loop**.

The enabling trick is **amortization**: one classifier-free-guidance CFM is trained over a rich
pre-computed *feature bank* (superset `y_catalog`), so the agent's fast inner loop explores factor
*subsets* via `drop_variables` **without retraining**, scored by MI-gain (pmi_flux; held-out ΔLL
fallback). Only a genuinely *new* named feature outside the bank triggers a **human-gated retrain** —
exactly the hybrid/staged autonomy split. We ground it in Flower's existing spectra pathway.

## Section outline
- **[`related-work.md`](related-work.md)** — vs CAAFE, agentic-science, MI-disentanglement.
- **[`method.md`](method.md)** — the loop, amortization, naming stage, Flower wiring.
- **[`evaluation.md`](evaluation.md)** — case study + planted-factor calibration control.
- **[`planning.md`](planning.md)** — dependencies, risks, timeline, code mapping.
