# Event-structured core: functionality and source capacity, not model quality

Implemented an original event-choice solver with continuation, two-child fork,
birth and death options. Each parent chooses one event, and every child has one
explanation. This forbids competing forks sharing a daughter or multiple parents
claiming one child. No detection is added, removed or moved. The label-free
candidate builder preserves existing division and synthetic/gap incidence and
retains the current graph as a feasible fallback. Forks consider at most eight
nearest available daughters per parent; all original single-link candidates
remain available. This is not a new deployed model.

Partial-label structured loss permits two known daughters to share their true
parent. It does not discard them as incompatible one-to-one constraints.
Unannotated children carry no negative label or loss; physically ambiguous
alternatives are not given a negative margin. A missing matched parent is never
relabelled as a biological birth. Training requires an optimal loss-augmented
solve; solver timeout/error is not a training oracle. Inference falls back to the
complete incumbent assignment when exact solving is unavailable.

Sixteen combined tests pass, including finite-difference gradients, competing
forks, unknown labels, source-only download scope, and incumbent preservation.
The real source pilot used eight previously cached optimization movies, all
available matched source forks, and one deterministic ordinary frame per movie.
All four physically matched source forks remain representable under the bounded
options. Twelve local CPU frames solved optimally with no fallback, each taking
0.125–0.672 seconds. The source script originally referenced a nonexistent
receipt filename; that failed before analysis and was repaired to the actual
hash-bearing feature receipt. No GPU restart or scientific parameter change.

The pilot's fixed -6 fork coefficient is a runtime-test coefficient, not a tuned
model. No pilot graph was exported or officially scored. The full training
feature schema, source regularization/calibration and held-out quality tests are
still required. Local solver timing is not a Kaggle runtime guarantee.

Research basis: Hirsch et al. use learned cell-state evidence, lineage
constraints and structured weight fitting for whole-embryo tracking. Our
implementation uses bounded event options and partial source labels rather than
their full node-selection formulation; their reported results do not establish
transfer to Biohub. No paper code or weights were imported.
[Primary paper, Methods](https://arxiv.org/pdf/2208.11467).
Exact-solve and time-limit handling follow the documented SciPy solver statuses.
[SciPy MILP documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html).

## First expanded source batch complete

Antelume batch 0 finished all eight 100-frame movies in 1,005.641 seconds
(16.76 minutes), below its 45-minute limit. Terminal output was backed up with
an exact file-set and SHA256 comparison: 41,392,622 bytes. Its GPU process is
terminal; no further batch is queued. Source image data remains in RAM pending
the remaining data-usefulness and recovery checks.

Current submitted structured-v1 inference was replayed on these graphs, and
the resulting groups and 18-feature edge tables were frozen in 35.156 seconds,
without annotations opened by the preparation script. Eight full graphs pass
schema/frame checks. These files contain predictions, not trained event weights.
Next: prepare source-only physical supervision and inspect expanded event
coverage before scheduling batch 1. Do not reopen selection or validation to
choose event parameters. Submission 56231458 stays unchanged.

Evidence: `trajectory-event-problems-v1-audit.json`,
`trajectory-event-source-v1-b0-full-result.json`,
`trajectory-event-source-v1-b0-full-harvest.json`, and
`trajectory-event-source-v1-b0-features.json`.

Expanded batch 0 source audit is now complete: 12 annotated forks, 10 with all
three cells matched, and both candidate edges available for all 10. Four forks
are exact-frame matches; six are absent at that exact frame. Official timing-
tolerant counts are 5 TP / 8 FP / 7 FN, so the six exact-frame absences must NOT
be advertised as six recoverable official false negatives. Existing-division
protection in the new bounded solver may exclude some additional cases; test
that before fitting. No source-label oracle graph was exported or scored.

After output and feature hashes were verified and no live process referenced
the cache, batch 0's recoverable image cache was retired: 3,112,687,838 bytes
including its manifest. Raw images were not locally backed up; they remain
recoverable from the competition archive and saved CRC/SHA manifest. All model,
graph, log and feature artifacts remain backed up. No foreign job, global OS
cache, root disk or GPU reset was touched.

Batch 1 contains eight further source movies and 3,444,935,013 image bytes.
Its bounded download is in progress under
`/dev/shm/biohub-event-source-v1-b1.4ku0nE`. It must pass its own same-contract
GPU smoke before full launch; later batches are not yet queued.

Batch 1 download completed with all 816 files verified. Its smoke passed in
18.8842 seconds, exact backup 352,691 bytes, and both graph schemas/frame
coverage checks passed. Full run launched at 13:54:37 UTC, PID 58309, under
the same root with a 45-minute watchdog. Contract:
`11de0aec9265b7bbd01146cd7ac079f951aab34b8ed829284a993560d228853f`.
The helper `advance-trajectory-event-source-v1.py` can verify one already-
launched smoke and perform its backup/full launch; it does not launch a second
experiment or restart a missing handle. Batch 1 is active; batches 2–7 remain
unlaunched. The authenticated Kaggle check at about 13:53 UTC still reports
submission 56231458 PENDING, without a returned public score.
