# Conservative labels for the broader FOCUS adaptation cache

Run only after the exact eight-movie cache is COMPLETE and its host verifier
passes every artifact, scope, runtime identity and replay check. Re-verify
actual cache against the saved report; incomplete/missing results are not
replaced with partial movies or downloaded public predictions.

Use the fixed four fitting/four diagnostic movie contract. Unlike the earlier
two-movie audit, all99 transitions of each complete movie have the SAME role;
the roles are disjoint by movie, so no temporal embargo is needed here.
Original checkpoint training included all eight; these are adaptation-only
diagnostics, not an independent model holdout.

Construct an edgeless node carrier solely to run the official patched7um
matching on the preserved raw detections. This is not a tracking prediction
or submission. GT is original training-only. Apply the already-tested unique
target/known adjacent parent/strict geometric parent-absence rule, leaving
unknown cells and ambiguous matches unsupervised. Do not alter nodes or label
unknown targets as negatives. Persist per-transition labels and status counts.

Implementation note from pre-execution testing: the scorer's outer evaluate
function skips node matching on edgeless graphs. Call its exact underlying
DistanceMatching(max_distance=7,scale=(1.625,.40625,.40625)) through graph.match
directly instead, preserving/restoring progress options. A small real-library
test compares the resulting node assignments with normal patched scoring.
Do not add artificial edges just to trigger the outer evaluator.

Before new-cache label generation, require the full existing-training replay
receiptca41210bcad92aeaa42c4bde83d2f9f7075d45c302816f0f40d7d3f11d82af78
and its unchanged implementation hashes. It reproduced all labels and status
decisions for176 transitions across16,700 raw detections from the earlier
fully scored training graphs. This is matcher equivalence, not a model gain.

Report fitting/diagnostic and per-movie positive, safely absent and ignored
counts. Do not choose movies, thresholds or null examples from metric gains.
No optimizer, GPU work, source-eight scoring or target opening in this step.
Training requires a separate bounded functionality design after coverage is
known; this audit does not automatically launch it or authorize submission.
