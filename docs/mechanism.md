# Mechanism, provenance, and explicit limitations

## Original equations inspected

Kazerounian & Grossberg (2014), [primary article](https://doi.org/10.3389/fpsyg.2014.01053),
sections 6.1–6.3 and 7.1–7.2. Publisher HTML retrieved 2026-10-16; displayed
MathML inspected from its embedded article data. Equations 12 and 13 give:

`dW_ij/dt = alpha f(c_j) [(1-W_ij)x_i - W_ij sum_(k!=i) x_k]`
`          = alpha f(c_j) [x_i - W_ij sum_k x_k]`.

For constant nonnegative x, S=sum(x)>0, the exact update is
`W(t+h) = x/S + (W(t)-x/S) exp(-alpha f(c_j) S h)`.
`instar` implements this equation, with a binary selected-cell gate in the pilot.
The isolated reference experiment tests the analytic attraction, normalization,
zero gate, separate cells' selectivity and memory after adding a new category.
It is not a reproduction of the coupled paper or figures 6–12. Those figures
use preallocated subset/permutation circuitry; no such candidate sequence
identities are placed in this model. Initial selectivity alone is not learning.

The [BU primary text on fuzzy ART choice](https://open.bu.edu/bitstreams/6089fca2-eb33-4168-9ded-daf4f68bc5b3/download),
algorithm equations 6–11, supplies an independent reference test: complement
coding, fuzzy minimum, Weber choice, vigilance and bounded template learning.
Its front matter must be used for bibliographic identity; it is an analysis of
choice functions, not silently treated as the 2014 circuit. The pilot uses
**cosine** vigilance and instar, not fuzzy ART min updates. Fuzzy ART is a
validated primitive here, not the full implemented learner.

## Explicit simplified architecture

1. Temporally scan every contiguous window of length 1–3. Supplied trial
   boundaries and a three-item span are architectural priors; segmentation is
   not learned. All token channels have identical treatment.
2. A discrete Item-and-Order approximation accumulates the arriving item at
   amplitude `0.65**rank`, then normalizes. Appending a different item preserves
   ratios among earlier entries. Repeats accumulate; this loses some order
   information. It is **not STORE2** or biological timing simulation. Rank
   computation is engineered temporal preprocessing, not an oracle factor code.
3. A generic category population has no assigned token subsets or sequence IDs.
   For each window length, select the best normalized dot-product match.
   Novelty below vigilance recruits a new uniform filter. Local instar changes
   only the selected filter. Initial commitment uses four integration units;
   subsequent updates one. This is an explicit fast-commitment choice.
4. Learned top-down filters independently track observed working memory through
   the same exponential outstar tracking approximation (rate .4 vs .8 bottom-up).
   A provisional winning template mixes 25% into the sensory memory before a
   second match. This is a new two-pass feedback rule, not ART resonance dynamics.
5. Each selected category receives observed outcome counts. The normalized
   counts are a local empirical conditional/outstar association, not a trained
   high-capacity decoder. No raw-input readout bypass is supplied. A category
   with at least five observations and >.8 empirical confidence in the wrong
   outcome is reset during training; search or recruitment follows. The outcome
   is a disclosed teaching signal, not hidden unsupervised learning. No true
   motif IDs, relevant positions, expected templates, or episodic tags enter
   hidden learning. This rule is inspired by predictive reset, not a claim to
   implement full ARTMAP match tracking.
6. At prediction, score each candidate as `max(cosine,0)**8 * confidence *
   log(1+count)`, where confidence is smoothed max outcome probability minus .25.
   Hard competition picks one category across windows. The no-competition
   control aggregates the same evidence. This algorithmic competition does not
   implement size/overlap-dependent Masking Field differential equations.

No 2011 physical-weight scales are mixed with 2014 normalized filters. The
pilot does not implement habituative gates, continuous shunting competition,
full resonance, rank coding, variable binding, or a hierarchy of learned
chunks of chunks. Those are unestablished extensions, contingent on passing
the prerequisite test. Short-list retention is measured, never guaranteed by
projection or protected weights. Fixed capacity and skipped updates are logged.

The task tests reusable **cue-conditioned list associations** across irrelevant
context combinations. It does not test a learned algebraic operator, transfer
to unseen cue/signal relations, or arbitrary relational binding. A later task
would combine two discovered motifs and learn an operator on their outcomes;
that stage is unrun and has no ready model. Do not call this pilot its solution.
