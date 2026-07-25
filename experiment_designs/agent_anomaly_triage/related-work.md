↑ [Back to README](README.md)

# Related work

## Related Work & Novelty Positioning

**Astronomaly** ([Lochner & Bassett 2021](https://ui.adsabs.harvard.edu/abs/2021A&C....3600481L);
[Protégé 2024](https://arxiv.org/html/2411.04188); [at-scale, 4M galaxies](https://academic.oup.com/mnras/article/529/1/732/7612998))
is the direct competitor and now the direct predecessor to our human loop: generic features →
isolation-forest score → **active learning** personalises recommendations. Its thesis — *anomalous ≠
interesting* — is the exact premise of our two-stage split. **We adopt its human-in-the-loop rather
than defer it**, but change three things: the underlying score is a **conditional** density (condition
*out* known astrophysics), an agent supplies **cross-match evidence** so the human scores *evidence*
rather than raw cutouts, and the learned ranker is **interpretable and warm-started from an elicited
rubric** (useful at label #1) rather than a black-box forest bootstrapped from scratch.

**Contextual anomaly detection** — anomaly as low `p(x | c)` — is established
([NormalcyScore/TMLR 2026](https://github.com/lucabindini/NormalcyScore); joint-VAE contextual AD
[1904.00548](https://arxiv.org/pdf/1904.00548)). So "conditional-density anomaly" is *not* our novelty;
our context `c = y` being a curated **astrophysical factor catalog** (residual "unexplained by known
astrophysics") plus per-factor localisation are the specific twist that Stage 2 then reasons over.

**Likelihood-OOD pathology & generative ensembles** — single generative models assign high likelihood
to OOD inputs ([Nalisnick 2019 typicality](https://arxiv.org/pdf/1906.02994);
[Likelihood Regret](https://arxiv.org/pdf/2003.02977)); **generative ensembles / WAIC**
([Choi et al. 2018](https://arxiv.org/pdf/1810.01392)) robustify density-based OOD by marginalising
over an ensemble. We use the ensemble for **model-independence of the recall ranking** (mean surprisal
`S̄`), reporting seed disagreement `U` only as a confidence tier. Grounded, not new.

**Preference learning, reward decomposition & elicited priors** — Stage 2 is a **learning-to-rank /
preference** problem, not deep RL: the scarce resource is astronomer labels, so we elicit a scoring
rubric as a **prior** and fit a shrinkage-regularised linear model plus a small residual for
interactions (Bradley-Terry-style preference modelling; reward/return decomposition into
interpretable additive terms plus a learned correction; hierarchical partial pooling to share strength
across users). Applying this stack — elicited-prior warm start + interaction residual + per-user
partial pooling, over agent-gathered cross-match evidence — to astronomical discovery is the specific
combination, and the **human-ensemble disagreement signal** (objective-vs-taste, mirroring the model's
`U`) is, as far as we know, novel to the setting.

## Where the novelty is (stated honestly)
On the **methods** axis each ingredient is incremental: contextual AD, generative-ensemble OOD, active
astro triage, and preference learning with elicited priors all exist. The defensible contribution is a
**systems/applications + interpretability** one: a two-stage discovery loop that (a) recalls on a
*conditional*, seed-marginalised density so "unusual" is model-independent, (b) has an agent turn each
candidate into a **structured cross-match evidence record**, and (c) re-ranks for *interesting* with an
**interpretable, elicited-prior preference model** trained on evidence-level scores, personalised per
astronomer and pooled to a consensus with a disagreement tier — which Astronomaly's black-box global
forest structurally cannot do. *Novelty verdict: passes as an application + interpretability
contribution built from known parts.* We frame it that way and let the **within-subject real-user
study** (§evaluation) carry the empirical claim.

### Sources
[Astronomaly](https://ui.adsabs.harvard.edu/abs/2021A&C....3600481L) ·
[WAIC generative ensembles](https://arxiv.org/pdf/1810.01392) ·
[Nalisnick typicality](https://arxiv.org/pdf/1906.02994) ·
[Likelihood Regret](https://arxiv.org/pdf/2003.02977) ·
[NormalcyScore contextual AD](https://github.com/lucabindini/NormalcyScore)
