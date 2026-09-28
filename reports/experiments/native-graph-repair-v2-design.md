# Native image-model graph repair: frozen before cross-embryo results

This is a prospective inference experiment, not an accepted submission. Preserve
the active full-training and queued diagnostic processes; run no competing GPU
probe. The previous goal turn made concrete progress by building/backup-verifying
native data and launching real training; it was not a no-progress turn.

## Component selection rule

Never ensemble merely because multiple checkpoints exist. For each training
embryo, its CNN can be used only if source admission and its frozen opposite-
embryo conditional gate both pass. Include that embryo's transformer only if
the CNN, transformer, and their predeclared equal mixture all pass those same
gates. If the CNN fails, use no model from that embryo; do not fall back to a
transformer based on target scores. Pool any eligible embryo groups with equal
group weights and equal member weights within each group. No target-fitted
weights, threshold sweep, or seed search. If no group qualifies, close this
integration experiment without full-movie scoring or submission.

This rule is fixed while full training is still running and the queued cross-
embryo result does not yet exist. It is a validation admission rule, not a claim
of a pristine untouched target set: selection movies were previously exposed.

## One fixed graph candidate

Keep every original public+trajectory node and every existing division. Use
image-only native appearance embeddings from actual detected node positions.
For each child, consider the nearest sixteen previous-frame graph nodes within
20um, plus a null parent, matching the training neighborhood contract. Cache
each node's encoder embedding once. Do not add truth coordinates or detections.

Only posterior >=.99 can propose an edge to a parent that has **no outgoing
edge in the original graph**. A child with no parent can be linked. A child
with an existing parent can be rewired only if its old parent has exactly one
outgoing edge and the old parent lies inside the evaluated candidate set. Thus
the previous link is explicitly compared, never treated as an unseen negative.
Preserve existing divisions; create no new divisions. When children compete for
one free parent, accept only the largest posterior, breaking ties by node IDs.
Do not reduce .99 if too few repairs occur. No runtime rule depends on labels.

This is not the full top-five objective: it is an inference candidate intended
to test whether the newly trained image experts improve a clean public base.
Do not call the overall goal complete merely because this candidate passes.

## Validation and runtime

First pass a small optimization-only GPU cache equivalence/throughput test and
small native-image crop-to-embedding test. Then create predictions without
labels for all100frames of the four locally available excluded validation movies:
44b6_12dfb391,44b6_267148e4,6bba_062c8d37,6bba_07e24132. Compare with the exact
accepted FP32 public+trajectory parent graphs. Only if official patched complete-
movie micro/per-embryo/worst-movie checks pass is it worth fetching/running the
other four held-out movies. All eight and offline Kaggle runtime acceptance are
required before promotion; four movies or conditional link accuracy cannot
establish submission readiness. No scoring of partial movie fragments.

The node-embedding implementation is not yet end-to-end GPU accepted. CPU head
factorization was bit-identical; changed encoder batch partitions differed by
~1e-6. Explicit GPU tolerance/unchanged-choice checks and official full-movie
outcomes remain required. Do not call this an exact floating-point cache until
scope-appropriate evidence supports that statement.

Pure graph tests passed4/4: no new nodes, isolated confident rewiring, division
protection, parent contention, and rejection of malformed/padded probabilities.

## Executed outcome: neutral full-movie candidate, not submission-ready

The predeclared component rule admitted exactly source6bba CNN+transformer at
.5/.5. Source44b6 models are excluded. GPU optimization-only cache probe passed
with bit-identical logits on four groups/24patches; native-image smoke passed
two four-frame segments in4.201s, projecting160.28s for four complete movies.

The full candidate actually completed in **44.602 seconds**, all100frames each:

| Movie | Child queries | Encoded nodes | Seconds | Added / rewired edges |
| --- | ---: | ---: | ---: | ---: |
| 44b6_12dfb391 | 44,615 | 44,953 | 18.112 | 0 / 0 |
| 6bba_062c8d37 | 5,813 | 5,853 | 3.860 | 0 / 0 |
| 44b6_267148e4 | 22,103 | 22,247 | 8.925 | 0 / 0 |
| 6bba_07e24132 | 28,922 | 29,181 | 11.626 | 0 / 0 |

All prediction files were downloaded and checked against terminal SHA256 and
the frozen parent graphs before any labels. Every node and edge set is identical;
different JSON hashes arise from serialization/edge ordering, not graph changes.
`native-repair-pilot-v2-score.json` therefore records **verified_neutral_complete_graphs**,
gate false, ground truth unopened. Official rescoring cannot reveal a genuine gain
from identical complete graphs and was skipped. No new submission is justified.
Do not lower .99 or run a threshold sweep on these target movies to force edits.

Full terminal SHA`90b4b4dc278aef8dc7482444edf68158f06bd62b1320d78e56162b343c638e6f`;
pilot SHA`70e52675a68f62b75f028164e2a8a31260d6365f275bebd4c115ec7af8e81ee6`.
Outputs `.biohub/cache/native-repair-pilot-v2-full-output`.

Next is a bounded, label-free attribution run using the unchanged model/policy:
top-null rates, maximum posterior distribution, baseline-edge agreement, free
parent/child availability and exact filter rejection counts. It changes neither
models nor graphs and opens no target labels. Determine the actual deployment
gap before choosing another model/data experiment; do not assume the zero edits
prove weak image features or that a looser confidence threshold would help.
