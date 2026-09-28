# Reciprocal adjacent-endpoint reconnection v1

Frozen September 13, 2026 before generating or scoring this candidate.

The ordinary-edge audit of four previously exposed ORIGINAL public baseline
graphs finds 18 missing annotated edges still represented by genuine retained
raw detections; three have source out-degree zero and target in-degree zero.
The conservative both-closed-annotation supervised subset has zero negatives,
so no supervised classifier is justified on that subset. An additional topology
diagnostic inspected the 18 edge probabilities; these are NOT pristine test
movies. Their per-edge values will not be used as thresholds or a lookup table.

Test one ordinary tracking repair, distinct from the closed bridge through
omitted detections: reconnect adjacent existing track endpoints, adding NO
nodes and changing NO existing edge or coordinate. Use only the ORIGINAL public
baseline (not rejected D4, projection, short-track or division variants).

Fixed policy: genuine original detector identities, unique reciprocal maximum
raw learned probability >=0.88 over ALL raw alternatives, physical displacement
<=6 microns, source out-degree zero and target in-degree zero. Require two
unbranched past links before the source and two unbranched future links after
the target. Linear least-squares extrapolation from each three-point track must
agree with the opposite endpoint within3 microns, in both time directions.
The 0.88/6/3 constants reuse the prior independently fixed bridge safety values;
no new threshold/seed/feature sweep is permitted. No time-gap, division creation,
deletion, synthetic point, lineage-degree violation, movie ID or label input.

Freeze all four 100-frame graphs before opening annotations for scoring. Use
the same pinned patched official scorer and complete sparse GEFF inventories.
Require pooled combined AND raw edge improvement, every movie and embryo
nonregression, and a genuine TP increase. A neutral/failing result closes this
recipe without threshold tuning. This is only an exposed training diagnostic;
even a pass requires division-positive complete-movie checks, embryo-held-out
model evidence, full offline two-worker inference/runtime and license checks
before promotion. No automatic live submission is authorized by a diagnostic.

CPU only, 20-minute watchdog, immutable input hashes, focused tests first.
No Antelume/Kaggle GPU, shared project mutation, or remote instance operation.
