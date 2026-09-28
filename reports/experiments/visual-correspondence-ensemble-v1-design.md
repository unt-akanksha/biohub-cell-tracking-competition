# Visual correspondence ensemble: September 14, 2026

Status: new code/data preparation; not yet trained or a submission candidate.

The user requests a materially stronger trained ensemble and authorizes use of
Antelume. Authenticated leaderboard inspection confirms Sergio Alvarez at 0.970,
not a publicly released 0.970 model. Submission 56219125 remains pending at the
02:16 UTC check. No leaderboard score is used to fit or select these models.

## Why this experiment

The earlier owned detector blend regressed all eight source movies, adding 871
false edges and 350 false divisions. LSM feature-24/36 ensembling also failed its
worst-movie gate. Do not restore either blend or retrain the failed 74.7M division
pilot unchanged. The current submitted reference is retained separately.

The refreshed [discussion on appearance and localization](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/740145)
suggests higher-recall proposals followed by learned appearance ranking. Its
claims and figure attachments are not reproduced results. New notebook inventory
still contains old metric-hack entries; those sources were not pulled. Source-only
inspection of [Harmonic Fusion](https://www.kaggle.com/code/flexonafft/biohub-harmonic-fusion)
shows the familiar primary/secondary public weights, with harmonic forward/backward
association fusion; not an independently trained third family. No downloaded
predictions, notebook execution, proxy sweep, or metric exploit is used here.

[Trackastra](https://arxiv.org/abs/2405.15700) motivates candidate-context attention.
[CELLECT](https://www.nature.com/articles/s41592-025-02886-x) motivates learning
visual correspondence rather than adding more copies of the same detector.
This is our implementation inspired by those ideas, not either paper's model,
pretrained weights, reproduced benchmark, or a promised Biohub improvement.

## Frozen first experiment

Two visual association architectures, each trained independently on each source
embryo: a multiscale residual 3D CNN and a 3D CNN plus four-layer 384-dimensional
candidate-context transformer. Both start randomly, with a zero-initialized
residual over the same physical-distance parent prior. They do not train the
detector, relabel unknown cells, or change existing submission graphs.

Use existing hash-verified expanded raw-image triplet shards, excluding the eight
current full-movie diagnostics, four public-test copies, and every sealed-audit
shard. Preserve original optimization/selection movie assignments; no new split
search. Train only on one embryo per fold. Historical project exposure is not
erased; this is not a pristine project-wide CV claim.

Child queries have annotated parents. Parent proposals come from an image-only
3D difference-of-Gaussian detector, with nearest 16 inside 20 micrometres selected
BEFORE consulting truth. A unique child's parent provides backward supervision
without treating unknown children as negatives. A parent proposal within 3.25um
can be positive; other proposals within 7um of its true parent are ambiguous and
receive no negative loss. If there is only ambiguous evidence, exclude that row,
not label it absent. Ground-truth parents are never forced into top-k. Empty/far
sets can be null; 20% true-parent dropout trains explicit missing-parent handling.

Inputs are fine/context 3D patches sampled at 1.625/3.25um per voxel. Stored 15-cubes
permit physically consistent subvoxel center jitter before 11-cube model crops.
Shared XY rotations also transform displacement vectors. Brightness augmentation,
movie-balanced sampling, AdamW, gradient clipping, AMP with actual-update checks.
Child query anchors are GT rather than public detector coordinates: this remaining
domain difference must be evaluated on predicted detections before promotion.

## Gates and budget

Eight pure data-contract tests plus an actual 20-step/reload GPU smoke for EACH
architecture precede extended training. Four members run sequentially, at most
5,000 steps/40 minutes each, with a 168-minute overall hard watchdog. The smoke
must project the full run inside that limit; otherwise revise the declared run
size explicitly before training, never run an untested large job.

Select a member's checkpoint by source-selection NLL only. The preliminary source
screen requires at least 2% NLL improvement over distance, no fewer correct parent
choices, and no missing-parent NLL regression. These are conditional correspondence
diagnostics, NOT complete-movie scores. No member is admitted to a submission on
this evidence alone. Following this screen: cross-embryo testing, paired frozen
mixture vs each individually strong member, then predicted-detection/full-movie
official scoring, per-embryo/worst-movie checks, and two-T4 runtime acceptance.

Reserve 90% of free Antelume VRAM; no Kaggle training launch. Leave shared packages
and RSNA artifacts unchanged. Periodically detect foreign GPU work and checkpoint/
yield our own process. Save one rolling optimizer/RNG resume plus each member's
best weights/history. Refuse less than 850MiB free disk before starting. There is
no instance shutdown, foreign-process termination, or external submission action
in this trainer. User permits clearing GPU/instance memory if needed; no such
disruption is necessary while the GPU is already free.

## Prepared and smoke launched, 02:52 UTC

Eight data tests passed; the existing Antelume environment also passed seven
Torch checks for each architecture, including nonzero-head candidate permutation,
empty parent sets, finite backward gradients, and actual visual-encoder gradients.
Exact capacities: residual CNN 9,014,274 parameters; token transformer 15,849,858.

Data preparation completed in 190.078 CPU seconds: 460 packets / 322,457,784 bytes.
Source optimization contains 705 groups from 27 44b6 movies and 3,549 groups from
64 6bba movies. Source selection contains only 11 groups from three 44b6 movies,
versus 180 groups from seven 6bba movies. This small 44b6 selection set is an
important uncertainty, not hidden evidence of generalization. Ambiguous rows
were excluded (173/2,390 optimization rows respectively), not converted into
false labels. Existing sealed-audit shards were not opened.

Bundle SHA256 `096abae5977aa95d68091cea49bafcdeed98bb879578b056e421b0472c291d4a`.
The exact bundle transferred to the owned Antelume directory. Standalone source
checks run on CPU; actual two-architecture real-data GPU smoke PID 19907 launched
at epoch 1789354359.440313 (02:52:39 UTC). Extended training is NOT launched yet.
Root disk retains about 1.5GiB; RAM inspection showed 14,987MiB available, and no
foreign GPU process. No cache/process cleanup was necessary despite permission.

New source-only notebook inspection also confirms that
[the September 14 runnable notebook](https://www.kaggle.com/code/zhincez/biohub-0-947-lb-runnable-with-public-datasets)
uses the same three weight hashes as our current reference. Its author reports
0.947 after fixing dataset paths; this is not our score or a new trained family.
Source SHA256 `38bca69a477e9090717c59c54358c2434e168042ac29734b9b12922aa7d0f186`.
Harmonic Fusion source SHA256
`e378e723ff30c3bebbe21b68553b3c666c4592141288f32529c1322b790fb44a`.
No public notebook code, pretrained weights, or predicted CSV was used to train
these two new model families.

## Smoke-only repair, 02:56 UTC

Revision1 smoke stopped after 8.861 seconds, before optimizer training, because
the distance-baseline evaluator attempted `.eval()` on its `None` sentinel.
Failure receipt: `visual-correspondence-smoke-v1-failure.json`. The smoke served
its intended purpose: no extended run was started with the broken comparator.

Revision2 adds the explicit `model is not None` guard. Data, model definitions,
objectives, source-selection roles, thresholds, steps and resource limits are
unchanged. Original bundle and failure output are preserved. Data are shared
by hash-verified hard links, not duplicated on the nearly full instance disk.
Revision2 bundle SHA256
`a3a59450791c4d6aef7e75f8d7e2a40c5a41f9dc1a4a14d0440f0a1f0e1712d0`.
New real-data smoke PID 20291 launched at epoch 1789354575.47837. Full training
still requires this revision's passing terminal and throughput estimate.

## Extended training launched, 02:57:32 UTC

Revision2 real-data smoke PASSED in 22.325 seconds. Both architectures performed
20 actual optimizer updates and exactly reloaded their saved predictions. Peak
CUDA allocations were 5,025,313,792 and 5,111,451,648 bytes. Source-selection NLL
decreased from 0.84636 to 0.78032 / 0.80224 while correct choices stayed 162/180.
These tiny-smoke results show functioning optimization, not higher tracking
accuracy. Missing-parent accuracy remains poor at 6/180; longer training must
address that and cannot be promoted from the relative smoke screen alone.

Smoke's conservative four-member projection is 6,669.448 seconds (1.85 hours),
below the 10,080-second watchdog. Exact matching proof is saved as
`visual-correspondence-smoke-v1-r2-result.json`. The sequential extended training
job launched at epoch 1789354652.077115, PID 20495, with fresh random initialization
and no reused smoke weights. Launch receipt is
`visual-correspondence-train-v1-r2-launch.json`.

Expected training finish around 04:49 UTC; hard latest stop 05:45:32 UTC. Actual
member throughput may update that estimate. The trainer checkpoints every 250
steps and preserves each source-selected best model plus rolling optimizer/RNG
state. Research output: `/tmp/biohub-image-context-v2.ScdSdY/visual-correspondence-train-v1-r2`.
Log: same parent, `visual-correspondence-train-v1-r2.log`. The instance will remain
running after training; it is not automatically shut down. No RSNA artifacts,
foreign processes, system RAM caches, or GPU memory needed forced cleanup.

After terminal completion, harvest weights/hashes/history and evaluate genuine
cross-embryo complementarity and complete-movie metrics before any submission.
Current competition submission 56219125 is unchanged; no second submission or
Kaggle GPU research run was launched here.

## Source results and next diagnostic contract

Steady-state training is substantially faster than the conservative smoke-based
projection: completed 6bba CNN and transformer fits took 160.366 and 188.361
seconds, respectively. Each ran exactly 5,000 steps. Source-selected CNN step
3,750: NLL 0.22779, 167/180 correct; transformer step 2,750: NLL 0.29518, 167/180
correct. Both improve the 162/180 distance comparator. Simulated missing-parent
correct choices are 162/180 for both, versus 6/180 in the short smoke; these are
artificial removal diagnostics, not confirmed gains on naturally missing cells.

The 44b6 CNN completed in 163.345 seconds but FAILS its source screen: best step
250, NLL 0.52806 and 9/11 correct, losing a correct choice to the comparator.
Later checkpoints overfit; best selection is retained as declared, not the final
weights. It is not admitted to an ensemble. Fourth fit remains running at this
snapshot. No extra seeds/thresholds or relaxed admission rule are introduced.

Before opening cross-embryo predictions, the new evaluator fixes equal 0.5/0.5
probability averaging of the two admitted architectures per source embryo. It
does not open a fold's target unless both members pass their existing source
screen. Check each member against distance on the opposite embryo's existing
selection movies; the fixed mixture must match the better member's correct
count and improve its NLL, and missing-parent NLL must not regress versus the
comparator. Save per-movie results and first-only/second-only correct counts.
These remain conditional diagnostics; no sealedaudit/fullmovie promotion claim.
`scripts/evaluate-visual-correspondence-ensemble-v1.py` implements the comparison,
with no optimizer or model writes and an initial no-other-GPU-process guard.

`scripts/harvest-visual-correspondence-ensemble-v1.py` backs up all completed
members' best weights/history/results plus rolling optimizer/RNG state. It
requires terminal completion, verifies every file's size/SHA256, uses two bounded
transfer workers, and deletes no remote files.

## Completed: ensemble rejected; recoverable artifacts retained

All four fits completed in **732.553 seconds (12.21 minutes)**, each with exactly
5,000 actual optimizer updates. The initial 1.85-hour projection substantially
overestimated this cached-data run; use measured timing, not that estimate, for
cost reporting. Fourth member (44b6 transformer) also failed its source screen;
its selected checkpoint is step 1,000. Both 44b6 models remain unpromoted.

The frozen cross-embryo comparison completed in 4.423 GPU-job seconds. Models
trained on 6bba were evaluated on the 11 existing 44b6 selection groups:

| Method | Correct / total | NLL |
| --- | ---: | ---: |
| Distance comparator | 10 / 11 | 0.67505 |
| New residual CNN | 9 / 11 | 1.16742 |
| New token transformer | 10 / 11 | 0.83568 |
| Frozen equal-probability ensemble | 10 / 11 | 0.90698 |

The CNN contributes zero uniquely correct decisions; the transformer contributes
one; both miss one. The mixture's missing-parent diagnostic improves (10/11
correct, NLL 0.60715), but that does not offset the failed real-parent ranking/
calibration gate. **Reject this ensemble.** Do not loosen the gate, retune the
mixture on these 11 targets, claim a Kaggle improvement, or add these weights to
the submitted runtime. No full-movie promotion or new submission occurred.
The reverse fold's target predictions were not opened because its two members
failed source selection. This very small cross-embryo check is a rejection signal,
not a complete characterization of embryo generalization.

Result: `reports/experiments/visual-correspondence-cross-embryo-v1-result.json`.
All training weights, optimizer/RNG state, histories and receipts were backed up
with exact sizes and SHA256: 389,656,637 bytes for training, 289,878,545 for smoke.
Local recovery roots are `.biohub/cache/visual-correspondence-train-v1-r2-output`
and `.biohub/cache/visual-correspondence-smoke-v1-r2-output`.

After local AND remote per-file verification and a no-live-GPU-process check,
removed only four redundant remote files: the two smoke best checkpoints, smoke
rolling resume, and completed-training rolling resume. **480,159,184 bytes freed**;
all are recoverable from the local backups. All four full-training best weights
remain on Antelume. Root free space increased from 827,387,904 to 1,307,578,368
bytes. Cleanup receipt: `visual-correspondence-redundant-copies-cleanup-v1.json`.
No RSNA files/processes, package environment, RAM cache, or GPU service changed.

Read-only storage inspection also found an unmounted 232.8GiB NVMe device with
no filesystem shown by lsblk. That does NOT establish that its contents are
disposable; it was not formatted, mounted or modified. The user authorized GPU/
instance memory cleanup, not destruction of an unidentified block device.

GPU jobs are now finished and the GPU is free; the instance remains running and
billing. No other GPU training was silently queued. Goal remains active. The next
useful modeling direction is broader, representative training/validation coverage
(especially the smaller source embryo) and native-resolution appearance inputs,
not another seed/weight sweep over this failed small-sample ensemble. Dataset
expansion is a next step, NOT an already running job. Preserve the current submitted
candidate unchanged while pursuing that stronger evidence.
