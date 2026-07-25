↑ [Back to README](README.md)

# Related work

## Related Work & Novelty Positioning

**LLM-driven feature engineering.** **CAAFE** ([arXiv:2305.03403](https://arxiv.org/abs/2305.03403),
NeurIPS 2023) has an LLM iteratively propose tabular features, keeping those that raise a
**classifier's ROC-AUC** — the closest analog to our engineer+name step, but scored by a *downstream
task loss on a static predictor*: no latent, no conditional density, no *residual*. Our gate is
**MI-gain over a conditional generative latent**.

**Agentic scientific discovery.** **AI-Scientist**, **Coscientist**, **SciAgents**, **Kosmos** and
the HKUST *From Automation to Autonomy* survey run the loop over **literature, code, and lab
instruments**. *Agentic AI Scientists Are Not Built For Autonomous Discovery*
([arXiv:2605.08956](https://arxiv.org/html/2605.08956v1)) notes these rarely act on a crisp internal
*quantitative* object. Ours does: the search space and reward *are* the conditional-generative residual.

**Discovering factors via MI.** **CMID** and **C-Disentanglement** (NeurIPS 2023) minimize
(conditional) MI among latent dims — *training-time regularizers*, not an agent-driven
*condition-out → name → retrain* loop, and they never name factors against external physics.

**Generative residuals in astronomy.** Astro anomaly detection (**AnomalyMatch**, GAN/embedding
outlier searches) treats discovery as *distance in a static embedding*. Flower's `spender_I_flow`
models `p(z | y)` but with a **fixed, human-authored** catalog and no agent selecting conditioners.

**Where the novelty is.** No prior work composes: (a) a frozen-encoder **conditional generative**
latent, (b) an **autonomous agent** whose operator is *condition-out-and-measure-residual*, (c) an
**MI-gain** gate, (d) an **amortized-CFG** inner loop making subset search retrain-free — wired to a
real pipeline. CAAFE has the agent but not the residual/latent; disentanglement has the MI but not
the agent; agentic-science has the loop but not the crisp generative object.

*Novelty verdict: passes as an applications + systems contribution.* The pieces exist individually;
the **residual-conditioning loop operator** and its amortized wiring do not — and the honest framing
(§evaluation) keeps the claim proportionate.
