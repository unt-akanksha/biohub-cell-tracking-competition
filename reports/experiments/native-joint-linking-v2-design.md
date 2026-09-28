# Joint native-image linking, frozen before target scoring

The free-parent-only candidate was exactly neutral. Label-free attribution
found126high-confidence disagreements; all point to already occupied parents.
This is not a weak-confidence failure:78,700queries have a real-parent posterior
>=.99, of which78,574agree with an existing edge. Do not lower the .99 threshold.

Use the same admitted .5/.5 CNN/transformer and cache its unchanged image-only
candidate probabilities once. No target labels or scorer feedback have been
opened in either the neutral candidate or attribution. This new structural
hypothesis is joint identity assignment or missing cell division, not a looser
confidence threshold or new model/seed/weight fit.

One fixed candidate combines two supported graph operations:

1. Permute existing single-child parent links only when every participating
   child chooses the next parent in a closed directed cycle with posterior>=.99.
   Preserve edge count, every cell node, and existing divisions. Never steal a
   parent unilaterally from a child that still prefers it.
2. Add a second daughter only for an unlinked child and a parent currently with
   exactly one child, when BOTH daughters choose that parent with posterior>=.99.
   Require the parent to have three observed consecutive pre-division frames
   and each daughter three observed consecutive post-division frames, without
   intervening branches. Accept at most one new daughter per parent, highest
   posterior then stable node-ID order. No nodes are added, moved or deleted.

These are biological hypotheses, not verified divisions. Complete-movie patched
official scoring must reject added false branches or embryo/worst-movie losses.
Do not relax persistence/confidence if this candidate is neutral or fails.
The four movies remain a pilot; all eight movies plus offline inference/runtime
acceptance are required before a new submission. The top-five goal remains
unproven regardless of the pilot result.

## Completed and rejected

Verified cached posteriors total7,882,005bytes across four complete movies;
their decision counts exactly replay the earlier label-free attribution. All
are downloaded and SHA/size verified locally in `.biohub/cache/native-posterior-v2-cache`.

The fixed CPU graph operation took4.453seconds and added98persistent-division
hypotheses:48,1,20,29 respectively in the four ordered pilot movies. It found
zero supported identity cycles. No original division or cell node was changed.
These were hypotheses, not verified true events.

Patched official complete-movie scoring took28seconds and **rejects the candidate**:

| Pooled four-movie metric | Original | Candidate |
| --- | ---: | ---: |
| Combined score | .9383918673 | .9272848560 |
| Raw edge Jaccard | .9163179916 | .9152046784 |
| Adjusted edge Jaccard | .9183918673 | .9172848560 |
| Division TP / FP / FN | 1 /0 /4 | 1 /5 /4 |

Both44b6movies regress: -.00106641 and-.00506071; both6bbamovies are unchanged.
No true annotated division was recovered. Do not promote, submit, lower the
posterior threshold, shorten persistence, remove the measured false cases, or
fit another variant against these four target outcomes. No additional four-movie
download is justified by this failed pilot. Existing submission is unchanged.
Authoritative result `native-joint-pilot-v2-score.json`; frozen pilot contract
`67531626a2a0305d6d8e9b9adda9a42d9c3e3d509d59d1de8d577ae246a8f159`.

The learned parent-affinity ensemble remains recoverable and passes its frozen
conditional diagnostic, but independent parent affinity plus persistence is not
a calibrated division detector. Next inspect division-specific optimization
coverage and supervision; do not infer biological mitosis from two strong parent
scores alone. V2's uniform16-transition sampling included only2annotated44b6
optimization divisions and17in6bba. A read-only inventory is checking whether
other already-authorized training time points contain more true division events.
