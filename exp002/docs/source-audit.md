# EXP002 source investigation and model boundary

2026-10-16. This document precedes development experiments.

## Primary sources inspected

1. Kazerounian & Grossberg (2014), DOI
   [10.3389/fpsyg.2014.01053](https://doi.org/10.3389/fpsyg.2014.01053).
   Full publisher HTML and PDF, especially pp. 9–12, equations 1–13, inspected.
   STORE2 equations 2–3, habituation equation 7, balanced initialization equation
   8, and competitive instar equation 13 are implemented as named components.
2. Grossberg & Kazerounian (2011), DOI
   [10.1121/1.3589258](https://doi.org/10.1121/1.3589258), author-hosted
   [PDF](https://sites.bu.edu/steveg/files/2016/06/GroKazJASA2011.pdf), pp. 454–455,
   equations 13–20. Cognitive working-memory layers, modulatory chunk feedback,
   size/overlap-dependent shunting competition and habituation are implemented.
   The acoustic feature/item stages are outside this benchmark. This paper
   sets its sequence weights by hand; learning them is our extension.

## Available implementation search

Publisher article, author publication index, web searches for the paper title
plus source/MATLAB/code, and GitHub repository searches for `"masking field"
Grossberg`, `Grossberg masking`, `Bradski STORE`, and `ARTWORD` were inspected.
No verified reference implementation of these temporal circuits was found in
these bounded searches. This is not a claim that none exists. General ART
libraries (e.g. [AP6YC/adaptive_resonance](https://github.com/AP6YC/adaptive_resonance))
implement category-learning variants; their existence does not validate a
STORE2/Masking Field implementation. No third-party code was copied.

## Fidelity decisions that must remain visible

- `store2_rhs` transcribes 2014 equations 2–3. Equation 2 prints an input gain
  of .01 while the adjacent prose says .1. Literal .01 is the reference default;
  .1 is separately testable. Neither is silently declared the resolved value.
- The main temporal model uses the 2011 cognitive circuit (equations 13,15,17),
  rather than guessing the unstated E,F,H,L parameters and ambiguous indexed
  feedforward inhibition in 2014 equation 6. STORE2 is a separate reference and
  nonchunking control, not falsely described as the same circuit.
- Its initial bottom-up filters use 2014 equation 8: each unordered item subset
  has k! candidate cells with balanced random initial weight permutations.
  Subset connectivity and maximum list length are prewired. Temporal ordering
  and filter sharpening are learned; this is not arbitrary structural growth.
- The recurrent field coefficients are size(K)*(1+overlap(K,J)), normalized
  across sources K including self, following the displayed sums in 2011 eq17.
  A low-rank algebraic factorization changes computation, not coefficients.
- Firing outputs passed to sigmoid, learning, and feedback are rectified.
  The displayed square functions do not explicitly rectify negative membrane
  values; applying a square to a negative inhibited state would cause it to
  emit a positive firing signal. Rectification is an explicit modeling choice,
  not a claim of exact numerical replication.
- Main-model bottom-up W learning uses 2014 eq13. Independently stored top-down
  M uses a normalized outstar-style tracking rule with the same active-WM
  target. Unlike 2011's hand-set transpose, M is learned. This pairing is a
  proposed synthesis, not a source result. No correct sequence identity gates
  category assignment, reset, or weights. Full acoustic ART resonance, vigilance,
  mismatch search, rank coding, and repeated items within one list are absent.
- Source cognitive feedback is gated by prior lower activity (H(Y)): it cannot
  activate a never-seen working-memory item from zero. A missing input slot is
  weak nonspecific sensory evidence, distinct from literal silence. We must
  measure false completions and sensitivity, not reward hallucinated answers.
- Shunting states are integrated with exponential Euler at frozen within-step
  conductances, preserving their physiological bounds, not task outputs.
  Step halving is checked before confirmation. Learning uses the corresponding
  stable exponential update toward the sensory activity ratio. It is not a
  memory-preserving projection. No old output targets or update rejection.

## What a positive result could mean

Only conditional evidence for this specified learned temporal synthesis. Both
source equations and model modifications are inspectable. Mechanistic validity
requires chunk selectivity, causal competition/feedback use, learning-related
filter change, and preservation of shorter memories. Failure closes the gate
before stability/drift interpretation; a diagram with a chunk layer is not
sufficient. No outcome of this synthesis validates or falsifies complete ART,
STORE2, cARTWORD, or the general biological hypothesis.
