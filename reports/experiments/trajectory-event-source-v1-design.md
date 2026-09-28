# Source prediction collection for event-structured learning

The new submission 56231458 stays frozen. Its hidden evaluation is not a
hyperparameter experiment. One Antelume source-collection experiment may run
while Kaggle evaluates that submission, with independent receipts, no extra
Kaggle launch/quota usage, and no second experimental GPU job. This explicitly
clarifies the initial overly broad plan field requiring every Kaggle GPU run
to finish; it does not relax the sequential-experiment or eight-hour quota rule.
The distinction was communicated before the first source smoke launch.

## Scientific boundary

The current structured layer preserves degrees and divisions. The source audit
finds seven annotated forks across the initial eight movies, only four with all
three cells matched. Their broader candidate graphs contain both true daughter
links, whereas the raw neural edge proposals contain both for only one fork.
This motivates learning event choices on deployed predictor candidates, not
training another independent image-triplet classifier on the same tiny sample.

Collect predictions for every optimization-role movie with an annotated source
division, while retaining all eight previously collected source movies as
anchors. The frozen inventory contains 68 movies and 112 division events:
17 source44 movies / 19 events and 51 source6 movies / 93 events. Sixty new
movies remain after reusing eight complete prediction caches. Five of the
retained source movies contain no annotated divisions. No selection, validation
or public-test movie enters the new collection. Selection of new movies depends
on source annotations and deterministic hashes, never model errors/scores.

Collection runs the same licensed image models, initial ILP, public processing,
and original AR2 repair as the prior source collection. It saves raw candidates,
edge confidence, initial and postprocessed graphs, logs and counts. The frozen
structured v1 layer can subsequently be replayed on CPU from those artifacts;
this collection does not mix in rejected v2/v3/v5/v6 weights or fit a new model.
Future supervision must respect sparse unknown labels and unique physical
matches. No event learner is yet authorized for promotion or submission.

## Execution and recovery

Eight source batches contain at most eight full 100-frame movies each. Batch 0
contains 816 image files, 3,112,522,675 bytes. Every batch is capped at 5.5 GB
of input images and requires at least 1 GiB of free RAM-filesystem space after
download. Actual model allocation is limited to 70% of A10G memory; foreign
compute processes cause a launch rejection and are never stopped.

Each new contract requires a one-movie eight-frame end-to-end smoke, then an
exact terminal backup before the larger run. Full collection cap is 45 minutes;
the initial duration expectation is approximately 20–30 minutes per eight-movie
batch based on earlier A10G collections, subject to the first full measurement.
Do not queue all eight batches merely because this scope exists. Confirm the
first batch's terminal completeness, throughput and data usefulness first.

Runtime and images live only in a narrow `/dev/shm/biohub-event-source-v1-bN.*`
directory; the almost-full instance root disk and unrelated projects are left
alone. Timeout handling writes terminal status and preserves completed per-movie
files, allowing recovery and a later explicit remaining-movie plan. No silent
partial output is accepted as a successful full batch. Local and remote file
sets/hashes must agree before any completed data is removed from RAM.

Batch 0 contract:
`0a2f68bc710f3cb1eed4e9ce7e8c0e890d4f1efd77316fe3a9d208951b9ae9bc`.
Remote root: `/dev/shm/biohub-event-source-v1-b0.sm6eDF`.
Four scope/downloader tests pass; the range transfer's real chunk/CRC test and
full 816-file integrity checks pass. GPU smoke launched at 13:15:18 UTC,
PID 56870. This is not a new tracking score or an independently strong model.

## First smoke passed; first full collection launched

Smoke completed in 17.0858 seconds with status functionality_passed, all input
hashes unchanged, zero annotations opened and zero Kaggle quota used. The first
movie's eight frames produced 323 nodes and 282 edges; both original and repaired
graphs pass independent integer-schema, lineage-degree and complete-frame checks.
Exact local backup: 117,617 bytes. No smoke weights exist: inference only.

Batch 0 full collection launched at 13:16:45 UTC on Antelume, PID 57148, same
contract and image manifest, with a 2,700-second watchdog. Authoritative remote
log: `/dev/shm/biohub-event-source-v1-b0.sm6eDF/full.log`; terminal and progress
files live under `full/`. Do not relaunch if observation times out; inspect that
PID and its terminal files. No other source batch is queued.

Next after terminal: run `scripts/harvest-trajectory-event-source-v1.py --batch 0
--mode full`, verify eight complete movies and all graph/raw artifacts, then
perform source-only event-capacity/supervision checks. Do not retire images until
all desired source features are extracted and experiment output hashes match
local backups. Keep signed archive plans private. Submission 56231458 is
unchanged; none of this new collection enters that submitted kernel.
