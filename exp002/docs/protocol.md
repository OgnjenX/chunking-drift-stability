# EXP002 v1: development readiness and gated confirmation design

**Freeze point:** after the explicitly exploratory source/circuit development,
before any confirmatory run. This is a no-go specification with a prospective
confirmation design, not a claim that a confirmatory experiment has occurred.
The accompanying commit freezes code, development evidence, thresholds, and
this document. Readiness thresholds are declared after seeing development
results and must NOT be described as preregistered tests of those results.
No seed 1001–1012 was used during development. Configuration difficulty is null
because calibration failed; confirmation is not authorized by this protocol
until readiness passes and an executable successor protocol is frozen.

## Construct, inputs and supervision

Source audit: source-audit.md. The implementation uses the two cognitive
working-memory layers and Masking Field of the 2011 paper, with learned 2014
instar filters and a separately stored outstar-style feedback filter. STORE2
is transcribed as an independent reference/control. This synthesis is new;
known source benchmarks have not been successfully reproduced. Acoustic
layers, complete ART matching/search, endogenous segmentation, anatomical
rewiring and rank-coded repeated items within a list are not implemented.

No item-position tensor enters the network. Four item channels receive
successive pulses separated by gaps. Lists vary from 1 to 4 distinct items.
Recurring pairs appear alone and within longer lists. Weak nonspecific input
can replace an item; delays contain noise and distractor activity. All models
receive the same streams and trial-local seeds. A trial boundary resets state;
segmentation is given, not discovered. A response is requested at the end of
the delay. This is delayed sequence recognition/ordered recall through a
shared task readout, not reconstruction of the current input.

The sensory teaching target at the response time is an ordered list of up to
four items plus an end/padding symbol, shared by all models. No correct list
identity is given to chunk filters or learning gates. The output slots are
response coordinates, not copies of the input in space. Main endpoints are
exact entire-list accuracy including correct length and real-item token
accuracy excluding padding. Report item, short-list, long-list, missing-cue,
delay/interference and novel-composition performance separately.

## Exposure and controls

Development trained 30 presentations of each of 10 old items/lists, then 30
of each of 10 longer lists: 600 trials per seed, no old sensory replay in the
second phase. Source learning and readout updates occur online. Controls receive
exactly the same 600 presentations and sensory response targets.

- Full field: 64 potential list cells; anatomy provides item subsets and k!
  candidate cells per k-item subset, not learned anatomical discovery.
- Nonchunking cognitive WM: identical lower equations, no chunk activity, a
  separately trained task readout. It preserves the direct lower computation.
- STORE2 reference: literal 2014 input gain .01 and complementary gated layers.
  Its unit-input assumption requires normalizing noisy inputs whose sum >1;
  this is reported, not treated as a matched-capacity primary competitor.
- Trainable ordinary GRU: 21 hidden units, raw sequential inputs, recurrent
  memory and the same output targets. Its internal weights learn by full BPTT,
  with long-retention gate initialization and Adam .01. It has 2,078 trainable
  parameters, compared with 1,972 in the full field plus linear readout.

Exact parameter and effective-capacity equality is not claimed. Field/RLS
uses local sensory-ratio filter learning plus least-squares task learning;
GRU uses slot cross entropy and BPTT. Its poor development acquisition is a
fairness warning, not evidence that nonchunking mechanisms are inferior.
A future comparison must independently calibrate both architectures' learning
budgets/optimization before testing the mechanism. No extra optimization was
secretly given to either arm after observing a result.

Acute interventions leave the learned task head fixed: feedback off (without
rescaling bottom-up input), competition off, entire chunk path off, and random
25% chunk silencing. A separate trained nonchunking control distinguishes
inference-time damage from ability to learn without chunks. Stronger-activity
or matched lower-population lesions would be required for selective necessity;
they are not yet run. An ablation effect alone can reflect distribution shift
and cannot establish that selective sequence chunks exist.

## Stage 1: all prerequisites must pass

Independent development must FIRST find a common condition with exact list
accuracy .45–.85 in both full field and GRU, not selected for a field advantage.
The development grid is in development-plan.md and all attempted conditions are
retained. Integration step refinement must change clean responses by <2%.
At least 75% as many distinct chunk winners as learned lists, and >=80% distinct
winners for different orders of the same item set, are minimal construct
checks. Also inspect full code patterns and filter learning: winner counts
alone cannot show chunks emerged. A future confirmation must show learned
filter alignment and cross-trial selectivity exceeding its own untrained
baseline, using independent probe trials and no assigned hidden labels.

Confirmation, if eligible: all reserved seeds, no exclusions; old items >=90%
exact, old short sequences >=85%, new longer sequences >=80%; at most 5-point
old-memory loss from before to after learning longer lists. Feedback removal
must reduce calibrated exact performance by >=10 points; competition removal
by >=5 points; chunk-path removal by >=10 points. Each seed-paired 95%
bootstrap lower bound must exceed zero. Both full and primary recurrent
control must first acquire the task. No apparent chunk advantage may rescue
failed acquisition, missing headroom, or absent learned selectivity.

Seed is the inferential unit. For confirmation use 10,000 paired bootstrap
resamples with analysis seed 20261016, all 12 seeds, and per-seed raw values.
Multiple gates form a conjunction, not independent opportunities to claim
success. Any failed prerequisite closes Stage 2/3 interpretation. Wide intervals
are inconclusive; upper bounds below a threshold reject that practical effect
only for a valid, adequately calibrated model/task.

## Stage 2 design — conditional, not executed or claimed implemented

Compare OLD probe responses before/after longer-chunk learning, matched
exposures in controls, and learning-disabled filter ablations. Existing raw
A/B snapshots are development diagnostics, not a confirmatory recoding result.
A successor protocol must measure systematic change against a permutation
null for shared-subsequence relations (seed-level effect >=.05 and 95% lower
bound >0), plus old-function retention gates above. Track lower and higher
coordinate change, CKA geometry, trained cue-to-response relationships, order
relations and the model's actual readout. Do not require stable higher cells:
functional relationships can persist while both populations change.
A shift in activity during new learning is recoding, not postlearning drift.

## Stage 3 design — conditional, not executed or claimed implemented

Only after Stage 1 and valid Stage 2 characterization: fix categories, response
requirements and probe sets. Separately branch checkpoints into (a) ongoing
source-local W/M learning from familiar sensory streams without external task
labels, (b) externally applied independent stochastic W/M perturbations with
plasticity frozen, (c) both, and (d) neither. Calibrate nontrivial perturbation
levels on separate development tasks; no sigma is selected from confirmation.
No solution projection, old-output oracle, output-based update rejection or
protected high-level weights. Physiological shunting bounds do not preserve
function by construction. Fixed anatomical supports limit claims about turnover.

A future frozen executable protocol must specify perturbation levels, counts,
time points, chronically lesioned and sham controls, matched lower lesions,
per-seed noise streams and the final analysis. Register a >=5-point retention
and robustness advantage over an adequately acquired nonchunking model, paired
lower CI>0, and >=90% old-function retention; also require measurable response
change (1−cosine >=.05 and relative RMS>=.2). Report all four outcomes: structured
recoding, robustness only, both, or neither, and inadequate-model/no-test.
Generic robust performance does not show drift is useful. No high-level
coordinate-stability criterion is imposed.

## Current execution boundary

The readiness checker saves a machine-readable no-go result if development
calibration/selectivity fails. It does not run confirmation or compute a
stability verdict. Confirmation machinery for later stages must be added and
frozen in a successor protocol after a validated Stage 1 model exists. Calling
those stages 'done' would misrepresent the work. This boundary prioritizes
construct validity over producing more trajectories from an invalid model.
