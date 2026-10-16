# Chunk learning, representational change, and memory stability

**Date:** October 16, 2026

**Status:** Provisional hypothesis, written before implementation and experiments.
Archived from cognitive-system-theory commit 49e4894; related local-note
crosslinks are rendered as text in this portable copy.
No new evidence is claimed in this section. The dated results addendum, when
present, must be read separately from this initial proposal.

## The proposal

Learning recurring sequences may form competitive list representations that
feed information back to their constituent features. These learned interactions
might make familiar sequences recoverable even when lower-level responses
change. They might also help reuse familiar subsequences in new combinations.
Neither outcome follows simply from a larger chunk winning a competition.

This extends the cooperative stability hypothesis and drift note in the
originating cognitive-system-theory checkout (research-notes/). It is a computational possibility,
not a claim that chunking is the main cause of biological drift.

## Two different changes

**Learning-driven recoding** is change while the system acquires a new grouping,
category, or task. ABC becoming familiar can change responses to A or AB.
This is learning even if performance on earlier probes remains stable.

**Postlearning representational drift** is change in responses to an identical,
fixed probe set after the categories and required input-output function have
been fixed. Synaptic change alone is insufficient: neural responses must change.
If the required function deteriorates, call the outcome drift with forgetting,
not stable-function drift. Rehearsal of familiar inputs can still induce
learning; a fixed task does not mean plasticity has stopped.

**Retention** means correct output from the model's actual readout, without an
analyst refitting it. **Robustness** means performance under a specified new
perturbation, separately from unperturbed retention. **Reuse** means performance
or sample efficiency on new compositions of familiar parts, not accuracy on
already rehearsed lists. **Hierarchy-level stability** compares responses at
lower and chunk levels on the same probes; it does not follow from stable
labels assigned by an experimenter.

## What comes from Grossberg, and what does not

Kazerounian and Grossberg (2014) model ordered working-memory activity and
learning of list categories in a Masking Field. Competitive interactions favor
well-supported predictive lists and suppress less suitable competing chunks.
Suppression during recognition need not erase the smaller learned memories.
The LTM Invariance Principle concerns learning extensions of lists without
indiscriminately destabilizing previously learned sublists. It is not a theorem
that arbitrary synaptic noise preserves memories. Their article also discusses
ART top-down matching; not every simulation of a Masking Field is a full ART
feedback architecture. [Primary article](https://doi.org/10.3389/fpsyg.2014.01053).

Our extensions are that learned chunk feedback might compensate for changing
lower representations, make chunk responses more stable than constituent
responses, and improve compositional reuse under drift. These require tests.
The minimal benchmark will use explicit item-position inputs, a trainable lower
population, and competitive latent units with feedback. It will not reproduce
the original continuous working-memory equations, size-dependent masking
interactions, habituative gates, ART vigilance/reset, or a proof of LTM
invariance. It must be called Grossberg-inspired, not an implementation or
validation of the complete theory. It has two representational levels, not an
unbounded hierarchy of chunks of chunks. Competitive latent units only deserve
the operational name 'chunks' to the extent that their measured selectivity
shows grouping; architecture alone cannot establish meaningful chunk discovery.

## Adjacent explanations

Natrajan and Fitzgerald (2025) analyze nonlinear memory representations and
solution geometry. Their results connect exploration among solutions with
robustness and learning-related allocation; this is a different explanation
from hierarchical chunk formation. Their model assumptions and preservation
of a specified memory must not be converted into evidence that unconstrained
noise in our model improves memory. We will not project noisy weights onto a
solution manifold or reject updates based on preservation of task outputs.
Our experiment is not a replication or direct adjudication of their mechanism.
[Primary article](https://doi.org/10.1073/pnas.2500077122).

Rule and O'Leary (2022) model how a reader can track changing codes through
activity-dependent regulation, Hebbian learning, and recurrent interactions.
Their conditional computational results establish a compensation route, not
that drift is universally beneficial or that chunks are necessary. An adaptive
reader, homeostasis, and recurrence are alternatives that our simple benchmark
does not exhaust. [Primary article](https://doi.org/10.1073/pnas.2106692119).

Kalle Kossio and colleagues (2021) show persistent associative function with
changing assembly membership through activity-dependent and homeostatic
plasticity, with stable input/output relations. Their simulations begin with
specified assemblies; they do not establish discovery of a compositional list
hierarchy from raw experience. This is a strong alternative to any claim that
chunking is necessary for stable-function drift.
[Primary article](https://doi.org/10.1073/pnas.2023832118).

## Minimal discriminating experiment

First learn individual items and short ordered lists. Next introduce longer
lists made from recurring parts. Record changes on a fixed set of old probes
across this learning boundary. Afterward fix the category set and apply
unconstrained, seeded synaptic fluctuations. Record responses and retention at
several checkpoints. Run both frozen plasticity and sensory reconstruction
rehearsal; neither gets an old-output oracle during drift.

Compare ordinary nonhierarchical learning, competitive list learning, noise
without postlearning learning, and hierarchy plus noise. Add a noncompetitive
network with exactly the same trainable array shapes as the competitive model.
Use shared input streams, exposure budgets, seeds, perturbation realizations,
and reconstruction objectives. Test feedback removal and chunk disruption,
with matched lower-population disruption as an intervention control.

Measure old items, old short lists, newly learned longer lists, and held-out
compositions separately. Measure coordinate-level response change, population
geometry, and competitive-unit selectivity. Use the existing readout for
retention. Few-shot composition tests branch from a checkpoint and must not
alter the postlearning drift experiment.

The benchmark's frozen protocol must state numerical acquisition gates, noise
levels, drift thresholds, meaningful effect sizes, paired uncertainty, seeds,
and evaluation checkpoints before execution. Report all seeds and all primary
comparisons. Do not tune after seeing an outcome. A later redesign needs a new
protocol and must retain the original negative result.

## Predictions and falsification

The null is that competitive chunk organization adds no practically meaningful
benefit over matched noncompetitive learning under the registered noise and
experience schedules. A performance gain without measurable response change
supports robustness, not survival of drift. A benefit only with rehearsal
supports a rehearsed reconstruction mechanism, not autonomous self-healing.

The proposed mechanism gains conditional support only if: the networks first
learn the tasks; hierarchy adds meaningful retention/robustness over matched
controls; lower-level responses measurably change; older item and sublist
memories survive; higher responses are comparatively stable; and removing
feedback or disrupting learned chunk units selectively weakens the advantage.
Reuse requires its own held-out composition effect. These are separable claims.

The tested version fails if acquisition succeeds but competitive organization
provides no meaningful benefit, causes worse forgetting, or loses constituent
memories. Stable representations at both levels would fail the drift claim.
If higher units change as much as lower units, the proposed hierarchy-level
stability prediction is unsupported. No reuse advantage leaves the broader
composition proposal unsupported. Poor acquisition, ceiling effects, inadequate
noise, or wide uncertainty make the relevant claim inconclusive rather than
biologically falsified.

## Confounds and limits

Extra parameters, a second computational path, generic denoising, sparsity,
compression, reconstruction rehearsal, and effective noise sensitivity can
all mimic a chunk advantage. Exact parameter matching does not match effective
capacity, optimization difficulty, or computation. Report these limitations
and initial performance rather than normalizing them away. Larger chunks can
be brittle bottlenecks or suppress useful smaller memories. Feedback can
reinforce a wrong interpretation. Changing rates, attention, task context,
recording instability, and other biological mechanisms can also produce
apparent drift. A tiny artificial sequence task cannot establish a neural
mechanism, full continual learning, causal usefulness of drift itself, or
necessity of chunking. Directly testing usefulness of drift requires a separate
intervention that varies drift while preserving comparable learning ability.
