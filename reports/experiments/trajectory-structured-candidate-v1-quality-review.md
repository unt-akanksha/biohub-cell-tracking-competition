# Structured trajectory candidate: quality evidence and remaining runtime gate

Candidate is **structured-loss v1, source6weights only**, archive SHA
e33fe1b79291ed89697db7a5ee6a839bc2ef34e23f504b27f1f29e44290107ba.
Do not substitute the independent ranker, source44weights, or expanded-label
structured-v2weights. No mixture of rejected members is authorized.

Source prefilter failure is retained: v1 improved eight-source pooled score
.87729->.88123 but regressed44b6_c50204e0. V2 expanded physical source labels
without changing old positive identities, yet failed its own source/cross
consistency gates; no full-movie promotion for v2.

The earlier decision to stop v1 after the source prefilter was explicitly
revised: one **fixed, independent diagnostic**, not a source-gate pass or retune,
was run on the ten reserved selection movies because both source embryo
aggregates improved and the worst source movie was unchanged. Predictions were
frozen before selection GT opened. The original failure was never reclassified.

Independent confirmation:

| Complete cohort | Current candidate | New fixed model | Outcome |
|---|---:|---:|---|
| Ten selection movies |.9050633574|.9119301778|Four improve, six neutral; TP+17, FP-34|
| Eight separate validation movies |.9453907265|.9483971160|All movie/embryo nonregression gates pass|

Both evaluations use patched official scoring, full100-frame predictions,
per-embryo/micro-averaged reporting and unchanged node geometry/degrees/division
and gap incidence. The eight-movie baseline reproduces accepted T4 counts exactly.
The new head was not fitted on either cohort. The public backbone has competition
training overlap; this is not wholly independent validation of every backbone.
No public leaderboard score is predicted or top-five success claimed.

Manual review allows **runtime acceptance testing** based on consistent held
evidence, not a claim that every source training example improved. The retained
source regression remains a risk. Further finalization still requires actual
offline two-T4 acceptance, exact public-run CSV, runtime/reserve checks and review
of those artifacts. No competition submission has been made for this candidate.

Portable package contains only prediction functions and18scalar learned weights;
no training code, source annotations or public prediction tables. Its outputs
exactly reproduce all eight local quality graphs. The source6model JSON SHA is
c8411cf313a62f3a7d0f3550f68ca84d4dc97970cf31eefe3ff24bfe810402d4;
portable inference SHAa07d605e09b375e973e4e56bf26faca4d1d4ce10390f40e5eb50d33077d5a661.
This is not a new large neural ensemble; it is a learned joint-assignment layer
on the licensed three-backbone pipeline and our previous trajectory repair.

Kaggle acceptance `indarkarhana/biohub-structured-trajectory-acceptance/1`, kernel
ID134327465, pushed12:05:15UTC September14. Authenticated statusRUNNING; remote
code cells/inputs/private/GPU/internet-disabled settings exactly match stage.
Fresh quota28.65h;2hconservative reservation,8hreserve preserved. Antelume has no
Biohub GPU job active. Full selection corpus is backed up90,246,284bytes.

Acceptance contract4e13ec7dea134c2da7b13133d8ed5ba92d5bb392954f35e26c29288c446a4695.
Do not push again. On terminal download only `trajectory-complete/`,
`trajectory-smoke/` and logs with `--page-size200`; do not download the installed
dependency tree. Verify with `scripts/verify-trajectory-structured-kaggle-v1.py
--outputs <download-folder>`. That verifier has not yet run against the pending
job. Next adapt production staging/submission verification to this new model,
four workers and the degree-preserving swap policy; the old add-only verifier
must NOT be reused unchanged. Keep all model/quality hashes pinned.

## Acceptance complete and production launched, 12:35 UTC

The earlier RUNNING status above is superseded. Actual two-T4 acceptance passed:
all eight final graphs exactly equal the frozen scored graphs, all eight AR2
intermediates equal the accepted T4 baseline, and CSV reconstruction matches.
Full runtime was 952.0428 seconds; cohort projection is 6.5784 hours, not a
hidden-runtime guarantee. Acceptance receipt SHA:
bc5ed56e529a5f89a603a69ebf2127fc39bc48f5c0db9179f2e8f9bfa171f776.

Production kernel `indarkarhana/biohub-structured-trajectory-candidate/1`,
ID 134330065, launched at 12:35:51 UTC with 28.36 Kaggle hours remaining and
20 hours conservatively reserved. The only code change from acceptance is
RUN_MODE=production. Pulled remote code and offline/private/input settings
match exactly; kernel remains RUNNING at the latest authenticated check.
Do not push another version or launch a duplicate.

New release verifier: `scripts/submit-trajectory-structured-v1.py`. It verifies
the original add-only AR2 stage separately from the degree-preserving learned
swaps, independently replays the frozen head from raw/pre-postprocess outputs,
reconstructs CSV, and checks remote version, quota and submission history.
Seventeen tests passed, including replay of all eight real T4 graphs. After
the explicit runtime-accounting correction, seven focused tests also pass.
See `trajectory-structured-production-v1-runtime-policy.md`; public results
have not yet been opened. No new competition submission has been made.

Next on terminal: download only complete/smoke output trees, root submission.csv
and logs with page size 200. Run the release verifier under graph-analysis-venv
with the actual terminal SHA256, first without --execute. Disclose measured
runtime uncertainty before using --execute --runtime-risk-disclosed. The
historical add-only submission script must not be used for this candidate.

## Submitted at 13:02 UTC

The public production run is COMPLETE. Its four 100-frame movies finished in
705.1496 seconds; all four frozen model replays and graph invariants passed.
Independent CSV reconstruction matches 241,192 rows and SHA256
ce7804333cb1c3172a22bbce480f526ddbced11c718f9ef294faaeea7e864f64.
Structured stage rewired 1,103 edges (50, 8, 818, 227 across the four movies),
and the pre-existing AR2 stage added three edges. No node coordinates, node
counts or division incidence changed in the structured stage.

Runtime policy revision 3 passes: calibrated public workload 7.9280 hours,
pooled twelve-movie makespan projection 7.6338 hours, and public-only makespan
projection 9.7448 hours. The latter is close to the ten-hour watchdog. That
uncertainty was explicitly disclosed before submission; no guarantee of hidden
completion is made. Fresh quota was 28.12 hours, with 20 hours conservatively
reserved and the eight-hour reserve retained. One earlier submission today.

One-shot submission succeeded and was independently confirmed in authenticated
history: **56231458**, date **2026-09-14T13:02:07.997000 UTC**, status **PENDING**,
no public/private score yet. Do not resubmit or push a new version. Previous
submission 56219125 remains COMPLETE at 0.946. The top-five goal remains open.

Next-model source expansion is frozen separately, not included in this
submission: 68 optimization movies containing 112 annotated source events,
eight existing prediction caches reused and 60 new movies in eight small
batches. Two scope tests pass. No image download, GPU collection, model fit or
additional submission has launched for that plan.
