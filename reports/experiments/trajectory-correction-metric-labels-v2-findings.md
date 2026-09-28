# Source-only selector-label audit

## Hypothesis and frozen scope, September14 2026

The rejected logistic selector's labels count delta_correct_links on known,
unambiguous child links. Wrong-to-none and none-to-wrong changes have zero TP
change and therefore receive label-1 (ignored), even though FP can change.
The pinned patched official scorer uses edgeTP/(GTedges+edgeFP), with sparse
validity masks and duplicate-matched-edge collapse. Therefore a simple count of
removed edges is NOT sufficient to infer its contribution.

`audit-trajectory-correction-metric-labels-v2.py` compares fresh, complete-movie
official scoring of baseline against one atomic correction at a time. It uses
only the existing60optimization movies, never selection/validation annotations.
Both graph inputs, source dataset, component/old-label files, GT manifest/files
and scorer are verified. Baseline must replay prior official counts/score.

Sampling is frozen before counterfactual scoring: two smallest eligible graph
files per source embryo, first three partial-neutral components with unequal
added/removed counts. For the full audit, movies containing known-child neutral
cases precede wholly unknown cases, and known-child components precede unknown
ones. This stratification uses existing partial labels, never the new metric
outcomes. Smoke uses one movie/one component first. The small sample
is diagnostic, not an estimate of population prevalence. It includes unknown
old-label cases; these must not silently become negatives or exact supervision.

No gate is fitted, no threshold is tuned, no oracle-selected graph is exported,
and no result changes the active event fits, frozen evaluator or submission.
A target mismatch would not by itself prove why the previous gate failed.

The prior turn answered the user's feasibility question and inspected actual
gate records. This continuation adds an executable, source-only test rather
than treating the routing proposal as validated improvement.

## Smoke completed at21:10UTC; full diagnostic now permitted

First smoke failed during serialization: official EvaluationResult is a
NamedTuple, not a dataclass. Second failed because summarise returns edge
Jaccard but not edgeTP/FP/FN totals. Both receipts are preserved. Corrected
serialization uses _asdict; baseline counts are compared with the prior raw
per-movie row and score with its official summary. Two schema/regression tests
pass using the actual pinned scorer, including rejection of a changed count.

Third smoke TERMINAL success,16.422seconds, one complete-source-movie baseline
and one complete-movie counterfactual. It exactly replays original baseline;
the first unknown-child component on44b6_74d0c52e changes no official counts.
This proves functionality only, not the missing-FP hypothesis. Full sampling
explicitly prioritizes known-child neutral cases to test that hypothesis; no
smoke outcome is used to choose the full audit's graphs or thresholds.

## Full diagnostic complete: four missed beneficial corrections

Session94882 TERMINAL success in47.687seconds. Four source movies,12atomic
counterfactuals, all baseline counts and scores exactly replayed. Each of the
four known-child neutral examples removes one official FP without changing TP,
FN, divisions or node count. The eight sampled unknown-child examples change
no official counts. All predictions/geometry remain untouched on disk.

| Source movie | Component | Official single-movie score delta |
| --- | ---: | ---: |
| 44b6_74d0c52e | 27 | +0.00641105 |
| 44b6_587a1e22 | 162 | +0.00226592 |
| 6bba_2819ca14 | 6 | +0.00100452 |
| 6bba_b204cac7 | 16 | +0.00118807 |

These are diagnostic one-correction effects, not candidate gains, a pooled
score, an oracle-routing proposal or evidence of leaderboard improvement.
Full receiptSHA`2cf320dc1846fbf6a881f919f1543ed63a0b656344847a2b5b31a39ad6488099`.

An additional hash-verified scan of all14,561source correction labels finds
only36fully-known neutral unequal-edge-count cases:4on44b6,32on6bba, all net
removals. There are12more mixed-unknown cases on6bba that cannot be relabeled
as exact gains without further checking. These36are audit candidates, not
36verified positive labels. Coverage receiptSHA
`06ff8f58d6c34a0dafc0da1c939f6ddaae7b440ca9144edd269208b62ac0a524`.

Implication: the missing-FP training target is real, but correcting it alone
adds at most four clean examples to the weak44b6training fold (previously28
labeled examples). This is not enough evidence to justify a complex movie
fingerprint router or to attribute the earlier gate failure solely to labels.
Next useful selector work must preserve partial-annotation uncertainty and
test transfer with independent predictions, not fit an ID-specific exception.

## 21:23UTC: all36fully-known cases officially verified

Extended all-known audit61116 is TERMINAL success,223.422seconds. It audited
every36case across19source movies, not only the illustrative four. All36
remove exactly one official FP, preserve TP/FN/divisions/nodes, and produce
positive official single-movie score delta. All baselines replay exactly.
No mixed-unknown examples were included, no graph/policy was exported.

ReceiptSHA`c941afb0fed3bb6b9dc14f19d090e747b424f5dbd43ffef06485c671c2e2c306`;
audit scriptSHA`460068ec27c09e9ce3fcd6632f2b18648e1d9a89ffb07531d0423a8c59fad25d`.
The added source input check also verifies each initial graph against its
original backup manifest before applying any counterfactual mask.

The fixed label-only gate ablation is specified in
trajectory-correction-fp-gate-v2-design.md. At21:23:36UTC it launched as62324,
actualPython17108/launcher12020. Both original control fits replayed to1e-10;
revised held44model uses412examples [91negative,321positive], held6model32
[16negative,16positive]. Both complete-movie small smokes passed, then all60
graphs were frozen before official scoring. No new gate quality result yet.
