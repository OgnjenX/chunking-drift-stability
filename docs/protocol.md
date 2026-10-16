# Frozen protocol v1.0

Written before any experiment or outcome inspection, 2026-10-16. The initial
Git commit freezes this document, configuration, implementation and tests.
Any repairs must be documented, with the original run preserved. No tuning
of seeds, noise, duration, learning rates, gates, or outcomes after inspection.
This is an exploratory engineering preregistration, not a registered report.

## Task and model

Alphabet A–H, four ordinal slots, and a PAD output class. Inputs are explicit
item-position one-hots (36 dimensions). The task is sensory reconstruction:
predict each item and padding at each slot. No sequence identities, desired
chunk assignments, or behavior labels are used in training. The input itself
is the training target, so rehearsal supplies a sensory teaching signal. This
is not teacher-free autonomous compensation. Evaluation reports token accuracy
on actual items and exact sequence accuracy; PAD is not included in scores.
Full-cue retention and missing-item completion are different outcomes.
Completion reports both all-real-item token accuracy and accuracy only on
masked target positions; the reuse decision uses the latter.

Phase A: 8 individual items and 4 pairs AB, CD, EF, GH. Phase B: only 8 longer
lists, two recurring pairs each: ABAB, ABCD, CDCD, CDEF, EFEF, EFGH, GHGH,
GHAB. Earlier items/pairs are not rehearsed during B. Evaluate the same 12 old
probes across A and B to measure learning-driven recoding and forgetting.
The other 8 pair combinations are held out from all A/B and drift training.

Lower units u=tanh(xE). In the competitive model, c=softmax(u P^T / T),
v=cB and h=(1-g)u+g v; slot logits=hD. E,D,P,B are trained by exact gradients
of slot cross entropy (no auxiliary chunk labels). P is a learned recognition
filter, B a learned feedback dictionary; competition is divisive softmax.
This is one feedback pass, not a settled recurrent attractor. Competition does
not implement Grossberg's size-dependent Masking Field. Ordinal slots do not
implement his continuous Item-and-Order dynamics or prove LTM invariance.

Flat control: uD, no P or B. Capacity-matched dense control: replace softmax by
tanh(u P^T/T)/sqrt(K), with the same E,D,P,B shapes, feedback gain, gradient
updates, exposures, and initialization. It has a generic extra nonlinear path
but no competitive list code. It is the primary control. Same parameter count
does not equate effective capacity, optimization, or computational cost.

Use full-batch SGD, no momentum, no weight projection, no output-preserving
constraint, no hidden copy of old outputs. At each training epoch, independently
mask one real item in half the presentations; clean targets stay intact. The
mask/permutation streams are shared by seed across architectures. Random
initialization scales and complete update equations live in model.py.

## Postlearning factorial and interventions

Branch each B checkpoint into 3 architectures × 2 noise levels × 2 plasticity
schedules (12 conditions/seed). Noise is unconstrained independent Gaussian
random walk on every trainable array, including the readout and feedback.
Per-step SD=level × that array's RMS at B; the RMS stays fixed thereafter.
Dense and competitive models share the same standardized noise arrays. Flat
shares E,D draws; separate per-array streams prevent unused P,B affecting E,D.
No norm clipping or output-based accept/reject. Report parameter counts.

Frozen means no postlearning updates: flat+noise is the noise-only mechanism,
competitive+noise tests passive tolerance. Rehearsal means two epochs per step
on the fixed union of A and B inputs, using only sensory targets and the same
masking. The trained readout may adapt through this rule; evaluation never
refits it. Also report the output of the original B readout on current hidden
responses (anchored decoder), distinguishing intrinsic preservation from
adaptive readout. There is no new category during this period.

Chronic interventions branch from the SAME competitive B checkpoint under
noise and rehearsal: feedback gain=0; random 25% chunk silencing; random 25%
lower-unit silencing. Masks persist through inference and updates. Also record
acute versions of these interventions at every intact checkpoint. The chunk
intervention zeroes selected activities without renormalizing surviving
activities. Lower silencing has the same fractional unit count but cannot
match wiring or activity removal; no selective-necessity claim from one lesion.
Ablation onset and long-term recovery must be distinguished.

## Measurements

At each checkpoint, full-cue accuracy for old items, old pairs, new lists,
and held-out compositions; completion token/exact accuracy when exactly one
real item is removed (32 fixed repetitions/probe). Robustness averages token
accuracy over exactly 0,1,2 missing real items on learned longer lists, with
equal weight per severity (a discrete robustness score, not a continuous AUC).
Training and probe randomness are separate. No probe results select updates.

Save response arrays for every checkpoint, losses, final weights, and raw
per-seed metrics. Fixed clean probes: u (lower), c (higher), h (feedback-combined).
Measure mean 1-cosine, relative RMS change, and linear centered-kernel alignment
(CKA; 1-CKA is geometric change). Near-zero denominators are handled explicitly.
Compare A→B on OLD probes for recoding and B→post on fixed learned probes for
drift; never pool those two intervals. Old-probe representational geometry
also gets a descriptive structure contrast: cosine similarity of changes for
probes sharing an item minus disjoint probes. This is not causal evidence and
has no registered success threshold. Record chunk response entropy, occupied
winner count, and winner agreement over time. These diagnose code collapse;
units are not assigned to lists by an oracle. A single undifferentiated chunk
winner is not evidence of chunk formation.

Composition test: fork B, then learn only held-out recombinations for 0,1,5,20
full-batch epochs. Compare accuracy averaged over these four budgets and old
memory after 20 epochs. No compositional data enters the drift trajectories.
This is small-sample reuse, not a test of arbitrary grammar generalization.

## Gates and decisions fixed in advance

All seeds 0–7 are retained, including failures. Experimental unit=seed, not
individual trials or time points. Report paired 95% percentile bootstrap
intervals (10,000 seed resamples, fixed analysis seed 20261016) and every
per-seed result. Eight seeds and a single small task do not measure population
uncertainty over tasks, hyperparameters, or biological mechanisms.

Acquisition gate: mean clean token accuracy >=0.90 on old probes after A and
on all learned probes after B, for BOTH competitive and dense models. Report
per-seed failures and exact accuracy even when the mean gate passes. Failure
means the relevant mechanism comparison is inconclusive; do not exclude seeds.

Primary contrasts at step 100/noise .08, separately frozen and rehearsal:
competitive minus dense retention on all learned clean probes and robustness
score on learned longer lists. Null: no practically meaningful improvement
(5 percentage points). A conditional advantage requires point differences
>=.05 AND paired 95% lower bounds >0 on BOTH retention and robustness, plus
competitive final mean clean retention >=.90 and old item/pair mean >=.90.
The full mechanistic claim additionally requires lower drift mean cosine
change >=.05 and relative RMS >=.20, higher cosine change <=.75 × lower change,
more than one chunk winner at B, and >=.05 robustness loss from EACH chronic
feedback/chunk intervention relative to intact, with paired lower bound >0.
Lower lesion outcomes are reported alongside chunk lesions; generic lesion
sensitivity alone does not prove selective chunk causality.

Reuse support is separate: competitive minus dense mean held-out masked-target
accuracy across budgets >=.05, paired lower bound >0, and old-memory
accuracy >=.90 after the final budget. Learning-driven recoding is present if
old-probe A→B cosine change >=.05; structured recoding is only descriptive
unless a future protocol adds a statistical test with a proper null.

If acquisition passes but practical differences fail these gates, this model
provides no support for the corresponding prediction. An interval whose upper
bound is below .05 rules out the chosen effect size under this design. An
interval covering .05 is inconclusive about such an advantage. If baseline or
final accuracy is >=.98 for both models, explicitly flag ceiling limitations.
If representation change misses the drift gates, robustness cannot support
stable-function drift. No tuning or replacement experiment in this run.

## Scope

A toy denoising autoencoder with competitive codes can fail despite biological
chunking being useful. A positive outcome cannot establish biological drift,
chunk necessity, large-scale continual learning, or that drift itself helps.
The code tests learned competitive feedback, not complete Grossberg models,
Natrajan–Fitzgerald geometry, self-healing codes, or drifting assembly dynamics.
