↑ [Back to README](README.md)

# Related work

## 7. Related work & novelty positioning

**This section exists to keep the claims honest.** The scientific question — *what does a
self-supervised representation of Euclid encode beyond morphology?* — is not new here. It is the
explicit subject of Wu & Walmsley (2025), *"Re-envisioning Euclid Galaxy Morphology: Identifying
and Interpreting Features with Sparse Autoencoders"*
([arXiv:2510.23749](https://arxiv.org/abs/2510.23749)), using the *same* Euclid galaxy MAE
representation this design builds on (the specific checkpoint — RR2 vs. the later DR1 — is the §3
`[TO VERIFY]`; DR1 postdates their paper, so they most likely used RR2, which is also our
recommended Option A). Any framing that ignores this would not survive review.

### What Wu & Walmsley actually claim (and where)

The quotes below are drawn from arXiv:2510.23749v2 and located by section. **[TO VERIFY: exact
wording]** — confirm each verbatim against the source PDF before this doc is finalised; they were
transcribed via an automated extraction pass.

- **The GZ decision tree is an incomplete description, by construction.** *Introduction:*
  the supervised (Galaxy Zoo) framework "may miss concepts that are too rare to be manually
  detected," and "new, rare concepts are statistically guaranteed to be represented in the billions
  of resolved galaxies soon to be imaged by space telescopes like Euclid and Roman." — This is the
  same premise motivating our residual: morphology is not a complete latent description.
- **PCA on the supervised model mostly recovers GZ-aligned directions; SAEs go beyond.** *Results:*
  "PCA features become unidentifiable after ∼30 features and show no alignment with GZ. In contrast,
  our SAE delivers an order of magnitude more aligned features," and, on the MAE, "we see SAE
  features outside of the GZ decision tree, e.g., dust lanes in edge-on disk galaxies, elliptical
  galaxies with bluer companions, etc." — This is their positive result: a *dictionary* of
  interpretable directions, some of which happen to fall outside the GZ tree.
- **The MAE demonstrably encodes non-morphological structure — including artifacts.** *Results
  (self-supervised):* "the MAE embeddings are intent on reconstructing imaging artifacts such as
  saturated stars or ghosts, which can be diverse and span many pixels." — Directly relevant to us:
  the "beyond morphology" residual **will** surface imaging artifacts, so validation (§6) must
  separate genuine astrophysical novelty from artifact-driven novelty rather than assume the tail is
  interesting.
- **Interpretability of the discovered features is partial.** *Discussion:* "some SAE features
  remain difficult to interpret, despite the benefits of using Matryoshka SAEs to combat feature
  splitting and absorption."

### Where our edge is claimed to be

Wu & Walmsley **discover** a dictionary of directions and then *observe, post hoc*, that some are
not aligned with the GZ tree. They do not *condition morphology out*; separation from morphology is
a property they check for on a per-feature basis, not a constraint they impose. Our claimed edge is
that we **isolate the beyond-morphology structure by construction**:

1. **Explicit conditioning vs. post-hoc alignment checking.** By flowing over the MAE embedding
   *conditioned on* the morphology vote fractions, everything predictable from morphology is
   explained away inside the model; the residual is defined as orthogonal-to-morphology by the
   conditioning, not by inspecting a learned direction and judging whether it "looks like" a GZ
   answer. This should isolate non-morphological structure more cleanly and *exhaustively* than a
   sparse dictionary whose members individually may or may not be GZ-aligned.
2. **Per-object, continuous residual + anomaly score.** Their output is a global feature dictionary;
   ours is a per-object conditional likelihood (§5), i.e. a ranked anomaly score for every Euclid
   object, plus — optionally — a normalised mutual-information reading (§5) quantifying *how much*
   of the embedding morphology fails to explain. This is the granularity SAEs do not provide.
3. **Feature-set–resolved information via conditioning masks.** Training the flow with
   classifier-free-guidance–style masking over *subsets* of the GZ vote-fraction vector lets us ask
   which *specific* morphology questions explain which embedding structure — yielding conditional
   MIs per morphology feature set (e.g. "how much does the smooth-vs-featured axis alone explain,
   vs. spiral-arm structure?"), a decomposition finer than a single full-vector residual and not
   available from a static SAE dictionary. (Design consequence: the flow must be trained to accept
   partial/masked conditioning — see §4/§5.)
4. **Shared machinery, zero bespoke discovery model.** The residual falls out of a CFM Flower is
   already built to train, reusing the `spender_*_flow` frozen-encoder pattern — no separate SAE
   training and interpretation stage.

**Honesty check.** The edge is *isolation and per-object scoring*, not "first to look beyond
morphology" — they got there first with the same MAE. Whether cleaner isolation is real (rather than
asserted) is an empirical claim §6 must test; a fair, non-deliverable sanity check is to see whether
our top anomalies recover the *kinds* of beyond-GZ features they list (dust lanes, blue companions)
**and** whether we can push past the artifact-dominated tail they flag. If our residual only
re-finds saturated stars and ghosts, the "better isolation" claim fails.
