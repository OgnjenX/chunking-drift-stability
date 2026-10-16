# Development v1 — specified before execution

Date 2026-10-16. Exploratory, not confirmatory. Source and protocol are committed
before runs. All attempted configurations and raw observations are retained.

## Identifiability and data boundaries

Each world randomizes 14 raw channels into two cues, four signals and eight
fillers. For each cue, a random bijection maps signals to four outcome tokens.
Exactly one adjacent cue-signal motif appears at a variable sequence position;
an unmarked signal is a distractor. Length is five or six. The outcome cannot
be recovered by copying a fixed position. All eight predictive mappings occur
in training. A motif-ID oracle learns its count table from observed outcomes
and must reach 100%; these IDs never enter the learner or nonoracle control.
Filler ordered pairs have even parity in training and odd parity in testing.
Every primitive token is seen during training; exact train/test strings must
be disjoint. This is context recombination, not new relational binding.

For any held-out context, observing the same cue and signal identifies the
outcome from training. An iid scrambled-outcome world instead has conditional
Bayes accuracy .25, regardless of representation. Scrambling is intentionally
unlearnable; its low held-out accuracy is a negative control, not a calibration
failure. Arbitrary tags are sampled independently of rule outcomes. They are
learnable for rehearsed exact old experiences but cannot be inferred for novel
experiences. Tags only train a separate diagnostic associative count head from
selected category responses; they never update sensory/category filters.

## Budget and selection

World seeds 211,212,213 are development only. Eight novice trials, eight repeats,
then 320 later examples repeated four times (1,344 presentations/model).
Two vigilance settings .94/.985; feedback .25; fixed capacity 384; compare six
modes in the configuration file. Same stream/order/exposure count for all.
The random frozen population has the same capacity and downstream outcome
learning, with no recruitment/tuning. Its generic Dirichlet filters are not
sequence templates. No-chunk restricts candidate windows to singletons.
No-feedback removes template mixing; no-predictive removes outcome-reset and
reliability-based selection together, so it cannot isolate those two operations.
No-competition sums category evidence and updates via the same within-length
winner rule; it ablates inference competition only.

A nonchunking softmax linear sequence-feature model gets the same raw stream
and outcomes, with unigram through trigram occurrence features and one local
prediction-error update per trial. Three learning rates .05/.2/.8 are compared;
report all. A unigram version is a bag control. Feature vocabulary is built
from the supplied training inputs without outcomes; it sees no held-out strings
or labels. Parameter count and elapsed time are reported. Identical exposures
do not match floating-point operations, effective capacity or locality. The
sequence-feature baseline is competent engineering, not a biological control.

Select vigilance by mean development novel-context accuracy (stable tie favors
first setting). No further settings after observing results in this attempt.
If readiness fails, stop; a redesign needs its own dated protocol and attempt.

## Prospective readiness gates

Stage0: all worlds have oracle 1.0, no exact overlap, best sequence-feature
control >=.90, first/last-token majority <=.60, scrambled control <=.40.

Stage1 readiness: every world's novice and retained old accuracy >=.85,
novel-context accuracy >=.80, isolated old subchunk accuracy >=.85, no capacity
skips; filter path length >1 (necessary but not sufficient evidence of tuning).
Mean full-minus-frozen novel accuracy >=.10; selected-category motif purity
and fraction reused across >=2 contexts each >=.80. Full-minus-no-feedback,
full-minus-no-predictive and full-minus-no-competition each >=.03.
These are minimum pilot criteria, not biological effect sizes.

**Headroom caution:** clean prediction can reach a ceiling. A failure to show
an ablation effect at ceiling is inconclusive about that mechanism's utility,
but does not pass this readiness gate. No Stage2/3 follows. Difficulty may be
redesigned only in a subsequent development protocol, never confirmation.

All seeds/configurations are shown, including overflow and failed gates.
Three-world bootstrap intervals are descriptive and imprecise; they quantify
world sampling, not biological variability. Test success is code evidence only.
