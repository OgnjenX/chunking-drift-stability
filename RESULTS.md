# Findings from the replacement campaign

2026-10-16. **Exploratory no-go at Stage1.** The broader abstraction and
reorganization hypothesis remains untested by a valid full experiment. No
confirmation, Stage2 reorganization experiment, or Stage3 drift experiment ran.

The active suite replaces EXP001/EXP002; their complete code and observations
remain at pre-reset `cd95d2cf8c1e76d6a79d8a692238a2df858c7272` and its ancestors.
They were inadequate operationalizations, not scientific falsifications of the
broader hypothesis. The replacement is also a limited model, with limitations
made explicit in [the mechanism specification](docs/mechanism.md).

## What ran and what was validated

The source-derived **isolated** competitive instar update approached the observed
normalized temporal working-memory pattern. Across four fixed input patterns,
mean filter error fell from .610267 to .001513 over 30 updates. Maximum difference
from the independent analytic trajectory was 1.67e-16. Frozen weights did not
improve. This validates the isolated constant-input law, not the coupled 2014
Masking Field, complete ART, or figures 6–12. The working-memory circuit and
competitive population in the task are explicitly simplified approximations.

Development v1 was frozen at [`2336195`](https://github.com/OgnjenX/chunking-drift-stability/tree/233619509b9ed4e369af7424977c5e9762d62de2).
It ran both vigilance settings and every configured ablation on worlds 211–213.
The follow-up protocol/code was committed at
[`6cd2e8a`](https://github.com/OgnjenX/chunking-drift-stability/tree/6cd2e8a9fe415f5b78acf7c0b5610ee5c8b12314)
before worlds 311–313 were run. It increased capacity fourfold and evaluated
shared degraded probes. Both attempts are development, including their
prospective decision rules; neither is confirmatory evidence.

## Stage0: the limited task is identifiable and learnable

The true rule is a cue-conditioned mapping of the following signal to an
outcome. A cue/signal pair is embedded at variable positions among distractors.
All eight predictive mappings are observed in training. New test sequences
combine familiar tokens and motifs with unseen filler-pair contexts. They
require ignoring irrelevant context, **not discovering an unobserved relation,
variable binding, or a composition operator**.

In v1, all learned motif-ID oracle checks and the best trained sequence-feature
controls reached 100%. The oracle receives true motif IDs solely for this
capacity sanity check. First/last-token majority controls scored 26.25–30.63%,
and scrambled-world controls 23.13–25.63% (chance 25%). Exact training/test
strings and filler-pair contexts were disjoint; all primitive tokens were seen.
Thus this task has a learnable shared rule and defeats fixed-position copying.
The high clean accuracy also exposed inadequate headroom for some ablations.

## Stage1: some useful associations, insufficient mechanism validation

V1 selected vigilance .94 using development outcomes. Full-model novel-context
accuracies were 100%, 100%, 71.875%; frozen-filter controls scored 86.875%, 52.5%,
85%. The three-world mean difference was +15.83 percentage points, with a very
wide descriptive bootstrap interval [-13.13, +47.50]. Two worlds selected eight
pure, context-reused motif categories, while the third did not. Old isolated
cue/signal pairs survived, but one world's novice performance was only 75%.
Every full model filled all 384 category slots, skipping 1,398–1,562 candidate
updates. Clean no-feedback accuracy was identical to full. Inference competition
missed the prespecified three-point benefit. Stage1 readiness failed.

V2 tested whether this was merely a capacity/ceiling problem. All full models
had zero skipped updates and recruited 588/610/602 categories out of 1,536.
Shared input corruption created measurement headroom. All configured models
received the same streams and 1,344 presentations per world. Means below are
across **three fresh development worlds**, not additional confirmation seeds.

| Model | Clean novel context | 10% token replacement | 25% token replacement |
|---|---:|---:|---:|
| Full learned categories | 88.13% | 75.83% | 63.96% |
| Frozen random filters, learned outcome counts | 86.67% | 76.46% | 66.88% |
| Singleton categories only | 42.92% | 41.67% | 37.71% |
| No learned-template feedback | 88.13% | 76.46% | 64.17% |
| No predictive reset/reliability selection | 37.50% | 36.88% | 34.38% |
| Sum evidence instead of inference competition | 88.13% | 78.13% | 67.92% |
| Trained linear sequence-feature control | 100% | 84.58% | 75.21% |

Clean full-minus-frozen was only +1.46 points, interval [-4.38, +10.63]. At the
prespecified 10% corruption it was -.63 points [-3.75, +4.38]. Full-minus-no-feedback
was -.63 points [-1.88, +.63]. The nonchunking sequence-feature control exceeded
full by 8.75 points at 10% corruption. These intervals are descriptive bootstrap
ranges over just three independent worlds; they are not precise exclusion
bounds or biological uncertainty estimates. No practical superiority is established.

All three full models retained old outcome prediction at 100%, but novice
outcome acquisition had been 75%, 87.5%, 87.5%. Novice arbitrary-tag decoding
was only 75%, 75%, 62.5%, falling to 12.5%, 25%, 0% after later training. Exact
lookup can recover 100% of these fixed old tags (a disclosed capacity control,
not a bypass in the learner). The category reader was underacquired initially
and later suffered severe interference. This cannot demonstrate preservation
of individual experiences or distinguish loss of storage from loss of access.

Selected filters matched the true cue/signal working-memory pattern with mean
cosines .680, 1.000, .736 in v2; winner motif purity was .669, 1.000, .725.
Every selected category was reused across multiple contexts. Reuse alone is
therefore insufficient: a repeatedly used category can still choose the wrong
relation. Motif purity is a conservative pilot diagnostic that could also
reject legitimate merging of outcome-equivalent motifs; it is not a general
definition of abstraction. No higher-order hierarchy was implemented or validated.

## Why the proposed mechanism is inadequate here

Local filter tuning occurs. The full pilot also uses predictive selection:
removing reset and confidence weighting together substantially reduces accuracy.
This ablation does not separate their effects or establish biological plausibility.
Random fixed features with learned conditional counts are competitive, so the
utility of learning sensory filters is unresolved. The outcome-count reader has
no raw-input shortcut, but rich generic random representations can still work.

The feedback design has a strong limitation. It mixes the currently best
matching template back into its own cue. In the limiting case where learned top-down and
bottom-up templates are aligned, the initial winner b remains at least as close
after adding a positive multiple of its own template. For normalized template
directions u, the b-versus-j dot-product margin becomes
`(1-f) x·(u_b-u_j) + f ||w_b|| (1-u_b·u_j) >= 0`.
This limiting case shows self-reinforcement rather than contextual correction.
Different bottom-up/outstar learning rates can temporarily break that alignment;
the formula is not a universal invariance proof for all trained states. The
implemented feedback supplies no independent contextual expectation. It can
change scores across spans and recruitment thresholds, but that is much weaker
than recurrent ART hypothesis testing. Its lack of benefit does not establish
that well-designed top-down feedback is useless.

Category recruitment also indiscriminately stores nuisance windows; fourfold
capacity fixed overflow but did not fix hypothesis selection. Exact count heads
forget arbitrary tags as contexts merge. A full-path lesion produces chance
by construction because it removes the only outcome path; that is not a
selective causal demonstration of abstraction. Chronic ablations change learned
representations, and inference-competition ablation leaves training competition
intact. Equal exposures and equal category capacity do not match effective
capacity, arithmetic computation, or optimization. V2 allocates 49,152 numeric
filter/feedback/outcome parameters before tag counts, whereas the sequence-feature
control has a different, observed-feature-dependent size recorded per model.
No parameter-matched superiority claim is made.

## Gates and what did not run

| Stage | Status | Consequence |
|---|---|---|
| 0: limited task sanity | Passed v1; v2 clean oracle/control also 100% | Task is learnable; does not validate general relational abstraction |
| 1: learned predictive mechanism | Failed in both attempts | Source-law tuning is not sufficient system validation |
| Confirmation | Not run | Seeds 12001–12008 remain untouched |
| 2: novice-to-expert recoding/hierarchy | Not run | Saved novice readiness snapshots are not a recoding result |
| 3: postlearning drift/robustness | Not run | Sensory corruption above is not synaptic drift |

[Development v1 decision](results/development-v1/decision.json) and
[follow-up decision](results/development-v2/decision.json) preserve every gate.
Sparsity, compression, representational geometry, and causal drift usefulness
have no experimental conclusion here. The presence of a CKA helper is not an
executed geometry analysis. Prospective requirements are in
[the downstream protocol](docs/confirmation.md).

## Verification and reproducibility

Ten component tests pass. Ruff, formatting, type checks and Pylint error checks
were run. Audits reconstructed all 6,048 v1 predictions and all 10,080 v2
predictions (57 saved models), matched aggregate scores and original source
hashes, and replayed one complete original learning trajectory bit for bit.
The original dependency lock remains valid. All three plots were rendered and
visually inspected. These checks support implementation/data integrity, not
scientific completion. Raw input streams, predictions, weights, tuning curves,
configurations and provenance are preserved in `results/`.

Reproduce v1 using the README commands with a fresh output directory. To
reproduce v2 exactly without overwriting outcomes, use a separate checkout of
`6cd2e8a9fe415f5b78acf7c0b5610ee5c8b12314`, then run
`uv run --locked python -m scripts.capacity_headroom`. Audits and summary:

```sh
uv run --locked python -m srclearner.audit
uv run --locked python -m scripts.audit_followup
uv run --locked python -m scripts.summarize
```

## Next specific research action

Do not start drift or add another capacity setting. Design a new development
model that can reject nuisance hypotheses through conditional predictive value,
uses top-down expectations independent of the provisional sensory winner, and
has a separately tested episodic storage/retrieval route. First demonstrate
novice prediction and arbitrary-tag acquisition, learned-filter utility over a
strong frozen-feature control, and the causal contribution of feedback on
ambiguous probes. Only then implement and validate composition of learned
relations before measuring reorganization of fixed old experiences. The present
bounded campaign stops at its registered no-go rather than tuning toward a
positive conclusion.
