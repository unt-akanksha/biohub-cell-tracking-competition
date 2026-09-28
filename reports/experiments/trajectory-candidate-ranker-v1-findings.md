# Fixed trajectory ranking: one qualified source model, not a submission yet

The narrow disagreement corpus has only 12 unambiguous labels; no classifier
was fitted to that 11:1 split. Broadening the source candidate distribution
yielded 224,892 image-derived parent groups / 2,930,303 edges. Candidates are
the nearest 16 observed parents within 20 micrometres, union the two existing
stage choices. All candidate sets were frozen before source labels were attached.

There are 2,871 uniquely matched positive parent groups, 31,979 safe negative
candidates, eight known-absent parents and 22 clear current errors/absences.
Twenty repair parents are occupied and two are free. Unannotated/ambiguous
candidates are not silently negative. 2,828 groups have both a positive and a
safe alternative and enter the fixed regularized ranking loss.

The model is an 18-feature linear softmax ranker, NOT a large new network. It
uses physical distance, predicted predecessor/successor motion residuals, raw
neural probabilities with explicit missingness, initial ILP membership and
localization shifts. No movie/embryo identity or current-edge indicator is a
feature. Fixed L2=.01; no hyperparameter sweep. CPU fit/feature generation22.25s.

The initial v1 evaluation excluded ambiguous candidates using truth. Its
conditional gains must not be cited as full-inference gains. V2 preserves and
exactly reproduces its fit weights, then exposes **all** candidates at inference
and evaluates all 2,871 positive groups, including single-safe-choice cases.

| Model source | Whole-movie source cross-fit | Opposite-embryo predictions | Decision |
|---|---|---|---|
| 44b6 | 472 correct vs470 current /471 initial; two breaks |2390 vs2374/2383; no breaks|Reject: a held movie regressed|
| 6bba |2384 vs2374/2383; no breaks|473 vs470/471; no breaks|Eligible for separate selection only|

These are partial annotated parent-choice counts, not official graph scores.
The public backbone overlaps competition training. No ensemble of the rejected
model, no movie-specific routing and no submission promotion is authorized.

Tested fixed intervention: source6 learned-cost **joint** assignment among existing
ordinary one-child/one-parent links, preserving each node's outgoing/incoming
degree. All division- and gap-incident edges are protected; positions/node count
do not change. This is a quality-first component, not by itself a claim that the
full top-five gap is solved. All eight complete source graphs were scored with
the patched official scorer; source6 movies overlap fitting, so a pass would
only allow the separate selection check.

Full graph test **FAILED**, session30222 terminal. Across133changed links,
patched score0.8772907140 ->0.8766335571, TP3713->3712, FP261->263, FN243->244.
Division counts unchanged0TP/2FP/7FN. Seven movies are score-neutral; source
movie44b6_c50204e0 loses0.00725894. Source44aggregate regresses; source6isneutral.
No independent selection scoring, ensemble, routing, threshold tuning or
submission for this ranker+assignment candidate. Preserve this failed evidence.

Next hypothesis needs a feasibility check first: can the degree-preserving
assignment family actually recover annotated source errors? Measure a source-only
constrained upper bound (counts only, never export oracle predictions). If there
is useful headroom, train a loss on joint feasible assignments with sparse
positive constraints, not another independent ranking loss or different threshold.
No new structured learner has been implemented or launched yet. Independent
selection GT remains unopened by this lane; the ongoing baseline collection is
reusable for that new objective and subsequent legitimate candidates.

Antelume is collecting unchanged baseline graphs for all ten pre-existing
selection movies, none included in these fits. Same-contract8-frame smoke passed
26.82s,1,134,069output bytes backed up. Full GPU session46526/PID55483 launched
approximately11:12UTC,25-35minute estimate/60minute cap. This GPU job does not
apply the learned ranker or open labels. Model/assignment scoring is local CPU.

Remote root `/dev/shm/biohub-ranker-selection-v1.LJ0JSl`; contract
286b0eb2d0e2632e9914e3edfbb8c31f74a5429c9fa04f0a99e396a9e123dfab.
On terminal recover with `scripts/harvest-trajectory-ranker-selection-v1.py --mode full`.
Current public submission remains0.946. No new submission or Kaggle GPU used.
