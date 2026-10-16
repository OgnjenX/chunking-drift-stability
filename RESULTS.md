# Registered v1 outcome: no support for a competitive chunk advantage

**Run completed:** October 16, 2026. **Status:** Negative result for the tested
model; broader mechanistic question remains inconclusive.

The eight-seed benchmark ran all registered conditions. The competitive
feedback model did not outperform the equally sized noncompetitive control
on retention or perturbation robustness. It did not preserve all older item
memories, its higher responses were not more stable than lower responses, and
feedback/chunk interventions did not reveal the registered causal benefit.
No model settings or thresholds were changed after seeing results.

This is a deliberately small, Grossberg-inspired competitive autoencoder. It
is not a reproduction of Item-and-Order working memory, a Masking Field, ART,
or any of the cited drift models. Its failure does not falsify those models
or demonstrate that biological chunking cannot help stability.

## What was fixed before the run

[Protocol v1](https://github.com/OgnjenX/chunking-drift-stability/blob/25798a538fc4bb8c75c703d6f2d1cec102c90fda/docs/protocol.md)
and its complete code were committed before execution. Seeds 0–7, data,
learning/noise schedules, thresholds, and paired bootstrap analysis were fixed.
A practically meaningful advantage required at least 5 percentage points on
both retention and robustness, with paired 95% lower confidence bounds above
zero. Other gates required preserved old items/pairs, measurable lower drift,
comparatively stable higher responses, and effects of feedback/chunk disruption.

Three architectures were compared: flat (3,456 parameters), competitive
(5,760), and a dense noncompetitive path with exactly the same 5,760 parameters
and array shapes. Each was crossed with no noise/noise and frozen/rehearsed
postlearning plasticity. Three additional chronic interventions used the same
trained competitive checkpoints. Acute interventions were also evaluated.
There were **120 postlearning trajectories and 600 measured checkpoints**,
plus 48 learning-phase records and 96 composition-budget records.

Noise was an unconstrained random walk on all learned arrays, including
feedback and readout. There was no solution projection, old-output target,
noise rejection, or task-preservation constraint. Rehearsal did have a sensory
reconstruction target; it was not an autonomous self-healing condition.

## Acquisition and learning-driven change

All models achieved 100% clean item accuracy after initial learning. Both
controls retained 100% clean accuracy on all learned probes after longer-list
training. Competitive mean accuracy was 97.92%; seeds 3 and 4 were at 91.67%,
and the other six were at 100%. All passed the registered aggregate 90%
acquisition gate. The aggregate concealed a relevant weakness: competitive
old-item accuracy after longer-list learning was **87.50%**, versus 100% in
both controls; its old-pair accuracy remained 100%.

Lower-response change on identical OLD probes across learning was 0.0144
(mean 1−cosine) for competitive, 0.0102 for dense, and 0.0095 for flat. None
reached the registered 0.05 recoding threshold. The descriptive contrast
between changes in probes with shared versus disjoint items was positive in
all three architectures (0.335, 0.324, 0.299 respectively). That contrast had
no registered statistical decision and is not evidence unique to chunking.
Learning-driven structured recoding is therefore **unsupported by the stated
threshold**, with only a descriptive shared-item pattern.

At the end of learning the competitive codes used 5.125 distinct winning units
on average across 20 clean probes. This passes the minimal noncollapse check,
but does not establish a compositional chunk code. The [seed-0 activity map](results/registered-v1/competitive-codes.png)
shows all single items sharing a winner and some longer lists sharing winners.
Good reconstruction can be carried by the direct lower path even when the
competitive code does not distinguish the lists.

## Postlearning retention and robustness

At step 100 with per-step noise SD 0.08 × each array's fixed baseline RMS:

| Plasticity schedule | Competitive clean retention | Matched control | Difference, percentage points [paired 95% CI] | Robustness difference, percentage points [95% CI] |
|---|---:|---:|---:|---:|
| Frozen | 91.67% | 93.49% | −1.82 [−4.17, 0.00] | −0.68 [−1.72, +0.02] |
| Sensory rehearsal | 96.61% | 97.14% | −0.52 [−1.30, 0.00] | −0.06 [−0.24, +0.11] |

Robustness is the registered mean token accuracy on longer lists with exactly
0, 1, or 2 missing real items. It is separate from the clean retrieval result
and from accuracy restricted to the masked target positions. Its final scores
were 96.37% versus 97.05% (frozen), and 97.27% versus 97.33% (rehearsal).

**Ceiling limitation:** longer-list robustness was already about 97.3% before
drift. The matched control remained above 95%, leaving less than the 5-point
headroom required by the registered criterion. That makes this task a poor
sensitivity test of the chosen robustness effect size. The intervals describe
this toy task; they must not be presented as a general exclusion of a meaningful
chunk benefit on harder tasks. Acquisition and longer-list completion also
had substantial ceiling effects, even though the combined clean score was not
at ceiling in every competitive condition.

Old singleton memories were much less stable than the aggregate scores suggest:
competitive final accuracy was **53.13% frozen and 79.69% rehearsed**. Its old
pairs remained at 100%. Corresponding dense singleton scores were 60.94% and
82.81%; the smaller flat model scored 81.25% and 100%. Competitive learning did
not protect constituent item memories under the registered old-memory gate.
This is a failure of preservation in the instantiated model.

## Responses changed, but a stable higher level did not emerge

Competitive lower-response drift met both registered magnitude gates:

| Schedule | Lower 1−cosine | Lower relative RMS change | Higher 1−cosine |
|---|---:|---:|---:|
| Frozen | 0.0908 | 0.4248 | 0.0815 |
| Rehearsal | 0.0777 | 0.4041 | 0.1150 |

The higher level was not at least 25% more stable in either schedule. Under
rehearsal it changed more, not less. Geometry told a similar story: competitive
1−CKA under rehearsal was 0.0163 lower and 0.2484 higher. Some familiar functions
survived while responses changed, but the full learned function was not
preserved, especially for old singletons. This is not evidence of complete
stable-function drift mediated by chunks.

Actual model readouts were used for all performance decisions. The saved
original-readout diagnostic reached 98.18% clean accuracy under competitive
rehearsal, versus 96.61% with the model's current readout. Readout adaptation
and readout noise cannot automatically be called compensation; both can hurt.
No analyst-trained decoder was used to rescue the claim.

## Interventions and compositional reuse

Chronic feedback removal caused only a 0.045-point mean robustness loss
(95% CI −0.110 to +0.187 points). Chronic 25% chunk disruption had a −0.028-point
loss (a tiny improvement), CI −0.069 to 0.000 points. Neither approached the
registered 5-point effect. Chronic lower-unit disruption likewise had a small
mean loss, 0.077 points. Recovery through rehearsal is a possible confound,
but acute final-checkpoint lesions also had small robustness losses:
0.216 points feedback, 0.057 points chunks, and 1.034 points lower units.
These descriptive interventions indicate little dependence on the feedback
path for this easy longer-list metric; they do not establish selective chunk
necessity. The model may largely bypass the mechanism being proposed.

Held-out compositional completion was poor in both large models. Averaging
masked-target accuracy across budgets 0, 1, 5, 20 epochs gave 11.17% competitive
and 10.06% dense. The paired difference was **+1.11 points [−3.26, +5.76]**.
The interval includes the registered +5-point effect, so reuse is
**inconclusive**, with no demonstrated advantage. The 12.5% uniform alphabet
baseline is only descriptive: the grammar and available cues permit much
better inference than uniform guessing. The smaller flat model performed
considerably better (see the overview plot). Extra pathways/competition can
hinder this optimization problem; parameter count is not effective capacity.

## What is supported, unsupported, and inconclusive

- **Supported observations:** noise changed representations without a preservation
  constraint; short-pair and most longer-list functions survived; old singleton
  memories could fail while aggregate scores remained high. Sensory rehearsal
  improved aggregate retention descriptively in several conditions.
- **Unsupported in this model:** the registered competitive retention/robustness
  advantage, comparatively stable higher responses, preservation of every old
  memory type, a meaningful feedback/chunk lesion effect, and the registered
  magnitude of learning-driven recoding. No full-mechanism gate passed.
- **Inconclusive:** useful compositional reuse, stability under harder perturbation
  tasks with adequate measurement headroom, the biological hypothesis, and the
  causal usefulness of drift itself. The last was not directly isolated.

The experiment shows why longer-list success alone is insufficient. It also
shows a weakness in this first operationalization: generic reconstruction can
be solved through lower features while competitive codes partially collapse.
An explicit item-position input is already an engineered order representation.
There is no learned temporal segmentation or chunks-of-chunks hierarchy.
Backpropagation is not a demonstrated local biological compensation rule.
All seeds share one artificial grammar and one hyperparameter setting; paired
bootstrap intervals cover initialization/noise variability only. There is no
correction or confirmatory claim across multiple exploratory outcomes.

A future **new**, preregistered version should first verify meaningful learned
list selectivity and functional use of feedback, measure perturbations with
headroom on separate development data, and include several grammars and
learning budgets. It should add a faithful Item-and-Order/Masking Field model
if claims about those mechanisms are desired. It must retain this negative
run and must not force outputs to remain stable. No such redesign or additional
experiment was performed here.

## Evidence and verification

- [Raw measurements, states, tables, plots and provenance](results/registered-v1/)
- [Frozen configuration](configs/registered.json) and [summary decisions](results/registered-v1/summary.json)
- [Overview](results/registered-v1/overview.png), [diagnostics](results/registered-v1/diagnostics.png), [competitive code map](results/registered-v1/competitive-codes.png)
- [Post-run audit](results/registered-v1/audit.json): all 600 states independently
  reconstructed from saved weights and masks; every clean token/exact score
  checked; zero-noise frozen branches identical; one full noisy rehearsal branch
  reproduced bit for bit. All original source/configuration hashes still match.
- 21 tests passed in the pinned environment, including full finite-difference
  gradient checks, shared noise streams, held-out-set separation, intervention
  semantics, evaluation without mutation, and the ability of noise to destroy
  function.

The run used commit `25798a538fc4bb8c75c703d6f2d1cec102c90fda`, Python 3.11.9,
NumPy 2.3.4, single-thread settings, and the committed dependency lock. It ran
10:23:44–10:24:20 UTC. The initial status in metadata lists only the just-created
untracked results directory. The audit and supplementary diagnostics were added
after the experiment; they did not change training, registered analysis, or
thresholds. Plots were opened and visually inspected.
