# Development v2: capacity and measurement headroom

Written after inspecting v1 and committed before this follow-up runs. This is
exploration on fresh worlds 311–313, never reserved confirmation. V1 remains
unchanged, including code hashes, overflow, ceiling and failed readiness gates.

Keep all model rules, vigilance .94 and exposures unchanged. Increase generic
category capacity from 384 to 1536 in every category model including frozen.
Run full, frozen, no-chunk, no-feedback, no-predictive and no-competition; use
sequence-feature control rate .2 (v1 found this task learnable). No per-mode
parameter tuning. All receive identical 1,344 presentations per world. Capacity
is an upper bound, not an equal compute budget: report actual parameter sizes
and wall time; frozen uses every slot from initialization whereas adaptive
models recruit only as needed. Do not claim matched floating-point computation.

At evaluation, use the same clean probes and two shared degraded copies, with
each raw token independently replaced by a uniformly drawn vocabulary token
with probability .1 or .25. Keep the original task outcome. This destroys
information and may make some outcomes unrecoverable; clean oracle accuracy is
not a degraded-cue upper bound. All models see identical corrupted strings.
Use a corruption seed distinct from the training generator. No corrupted probe
trains a model. Report all noise levels; do not select a winning one.

Ask whether capacity skips disappear and feedback has >=.03 mean benefit at the
specified .1 noise, with every full model between .35 and .90 accuracy (headroom).
Also require novice/retained old >=.85 and clean novel >=.80 per world. These are
exploratory readiness diagnostics, not confirmation or proof of Stage1. If a
check fails, stop this bounded reset campaign and document the next architectural
question; do not add another configuration to rescue it. The prior stronger
Stage1 gates, including learning vs frozen and selective reuse, still apply.
No Stage2/3 or confirmation can be authorized by this abbreviated follow-up.

Measure arbitrary tag decoding separately: after novice learning, fit the local
category-to-tag counts on the same old experiences, then update that head from
new tags during later learning. This reader has no influence on hidden filters
or outcome prediction. Poor novice tag recall blocks later retention claims;
a decline alone cannot establish meaningful abstraction.
