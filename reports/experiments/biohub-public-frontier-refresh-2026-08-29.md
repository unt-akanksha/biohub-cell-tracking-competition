# Biohub public frontier refresh — 2026-08-29

Status: authenticated read-only source, output, rules, leaderboard, and
discussion audit completed. The active score target is `0.945`. Public
predictions are controls only and are not eligible submission artifacts.

## Public lineage

| Public kernel | Source SHA-256 | Observed result | Lineage finding |
|---|---|---:|---|
| `grafael/biohub-ct-0940-ema` | `4f0fb3aff772f4aaa0b687477e255c05211412a458d9db085935a7f37f073f33` | displayed `0.940` | `0.99238` normalized line-set Jaccard with the public 0.938 notebook; the substantive addition is a four-frame EMA velocity state with `alpha=0.4`, plus restoration of the older secondary detector blend |
| `grafael/biohub-ct-0938-harmonic` | `6d0f4314afdb1f90021898edb3559841152e21c4e3b197dcae460d368a52aedd` | displayed `0.938` | `0.99652` normalized line-set Jaccard with the public 0.927 lineage; changes are chiefly bidirectional and secondary-detection scalars |
| `notoverkil/biohub-base-0937` | `eb01b88a7128ff167cf73f3b0d1d8fc77f38618a0c9a94de1e7b0d886acf0a30` | displayed `0.937` | public-derived detector/linker with widened safe-division rules and the same four-movie validator family |

The 0.940 notebook is therefore a strong public control, not a new heavy model
family. Submitting it unchanged would be an exact public replica and is
excluded. Its EMA state is a useful hypothesis that may be independently
reimplemented and validated.

## Complete-movie validation audit

The downloaded outputs include the same four held-out complete movies. The
table below recomputes pooled edge and division counts from each
`validator_results.csv`; no threshold was selected from leaderboard feedback.

| Control | Adjusted edge Jaccard | Edge TP / FP / FN | Division TP / FP / FN |
|---|---:|---:|---:|
| public 0.937 | `0.915784` | `2187 / 109 / 106` | `1 / 1 / 4` |
| public 0.938 | `0.916462` | `2186 / 110 / 107` | `1 / 3 / 4` |
| public 0.940 EMA | **`0.917882`** | **`2188 / 103 / 105`** | `1 / 2 / 4` |

The EMA control recovers two more true edges than 0.938 while removing seven
false edges. Its division recall remains only `1/5`. The four-movie sample is
too sparse to select division thresholds reliably, but it identifies the
highest-value lane: recover complete division events without introducing
false forks. One additional true division can move the composite much more
than a small generic edge-linking adjustment.

An oracle feasibility audit of the 0.940 kernel's raw prediction GEFFs adds a
more specific diagnosis. All five annotated division triplets have the parent
and both daughters detected within 7 µm. Four triplets are fully matched within
3 µm; the remaining event has only its parent within 3 µm and daughters about
4.7–5.0 µm from truth. Every raw graph already connects exactly one correct
daughter. In all five cases the omitted daughter is the nearest next-frame
detection to the parent after excluding the existing child. Three omitted
daughters are parent-free; two require replacing an incoming edge. Adding the
five known missing edges as an oracle changes raw edge TP/FN by `+5/-5` and
raw division TP/FN from `0/5` to `5/0` without adding an oracle FP.

This result is feasibility evidence only: truth identified the missing
daughters, so it cannot authorize a distance threshold or a submission. It
does establish that a learned division-parent gate plus constrained
nearest-second-daughter selection has enough candidate recall to close the
target gap. The reproducible audit implementation is
`research/division_recovery_feasibility.py`; its ignored run artifact is
`.biohub/cache/analysis/division-recovery-0940.json`.

## Rules and discussion evidence

- External public data and models are allowed when competitors can obtain them
  with minimal cost; ZebraHub use is organizer-confirmed and declared to have
  no test overlap.
- Winning code must be reproducible and MIT-licensed. The competition data is
  CC0. Notebook submissions are limited to five per day and two final choices.
- Train and hidden test are embryo-disjoint, and the hidden test input is
  swapped into a rerun. Validation must therefore be complete-movie and
  embryo-aware.
- Discussion 737543 reports the common failure mode of very high detection
  recall but poor division Jaccard, and notes that useful division localization
  is substantially tighter than the nominal node-matching radius.
- Discussion 737101 recommends decomposing missing edges into missing endpoints
  versus incorrect associations. Discussion 734604 recommends measuring node
  recall, conditional linking, division candidate recall, and division ranking
  separately.

References:

- <https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/rules>
- <https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/data>
- <https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/737543>
- <https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/737101>
- <https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/734604>
- <https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/734330>

## Active candidate decision

The project-authored v4 external pretraining run uses two T4 GPUs, 46,386,607
parameters per fold, ZSNS004 optimization, and disjoint ZSNS005 selection and
audit windows. It reads no competition data. A recovery controller will verify
both checkpoints and will not launch the ceilinged competition-transfer stage.

The 47,964,082-parameter division localizer now warm-starts directly from that
verified external v4 output. It trains only the independently designed
localization objective, freezes the selection checkpoint before audit, and
changes coordinates only for predicted division parents and their immediate
daughters. The runtime is private and the training notebook has no competition
input or submission command.

To meet `0.945`, localization alone is not assumed sufficient. The candidate
must combine the 0.940 control's strongest reproducible elements with at least
one project-authored, validation-positive component:

1. learned division-event localization and division ranking from the v4 lane;
2. an independently evaluated robust motion state beyond fixed EMA; or
3. both, if each passes its frozen complete-movie gate.

No candidate is authorized for submission merely because training completes.
It must preserve the clean public control's edge quality, improve the frozen
complete-movie composite, remain non-identical to every audited public output,
finish inside the notebook limit, and pass strict output verification.

## Frozen `0.945` promotion path

The first submission candidate is now specified as the attributed clean public
control plus an independently reproduced four-frame velocity EMA and the
project-authored external division gate. The public safe-division heuristic is
disabled. The learned stage may add only the nearest parent-free second
daughter after an existing child; it cannot reassign an edge or modify a node
or coordinate. Every model checkpoint, policy, runtime source, public-control
notebook, and validator control is SHA-256 bound.

Policy calibration has two equivalent execution routes. Antelume uses one A10G
in the isolated `/home/ubuntu/biohub*` workspace. A Kaggle fallback attaches
only the private source runtime, the external ZebraHub shards, and the verified
v4 pretraining output; it attaches no competition source and has no submission
path. Both routes freeze the threshold on ZSNS005 selection windows and open
the disjoint audit windows once.

The external promotion gate compares the candidate with the exact public 0.940
validator artifact (`4dbf2079...e4941f4b`). It requires all of the following:

- candidate output SHA-256 differs from all three audited public submissions;
- at least one learned edge is added, with zero reassignments and zero
  node/coordinate changes;
- complete-movie proxy gain is at least `+0.005` over the public control's
  `0.932168` proxy;
- adjusted edge Jaccard regresses by no more than `0.001`;
- division TP and division Jaccard both improve;
- the two-T4 notebook completes before the 3,000-second watchdog boundary.

Only a hash-bound output passing every condition is submitted. An exactly-once
receipt is written before the external submit command, preventing a controller
restart from producing a duplicate submission.

## Contextual-v3 division-head control

While v4 was training, the two verified contextual-v3 folds were evaluated as
a diagnostic-only control with the same nearest-second-child policy. The
threshold selected on the external ZSNS005 selection windows made `2/2`
correct decisions. Applied unchanged to the disjoint audit windows, it made
`2/4` correct decisions (`0.50` precision, `2` division TP). This is positive
separability evidence, but the precision is below the v4 authorization gate and
the v3 audit had already been opened by its original training run. It cannot
authorize competition evaluation or submission.

The result strengthens the rationale for the v4 objective: useful division
signal already exists in the smaller model, while false-positive suppression
is the remaining bottleneck. The immutable diagnostic artifact is
`.biohub/staging/contextual-v3-division-recovery-diagnostic-v1.json` with
SHA-256 `a149ed7d6d3849bc2cb88159c5e4592d0466287b476e6b1029fb7aad69a32b51`.
The run also exposed and repaired a missing SciPy dependency in the isolated
Antelume Biohub environment before the v4 policy calibration needs it.

## V4 and focused-division outcome

The two-T4 v4 pretraining notebook completed 4,000 optimizer steps in both
folds, but its frozen association gate correctly rejected both checkpoints:
each best checkpoint remained at step zero and numerically identical to the
contextual-v3 initialization. The recovered 46,386,607-parameter checkpoints
were therefore allowed only as prediction-preserving initialization for a
division-specific experiment; they were never promoted as association models.

Two focused Antelume A10G runs optimized 25,178,047 division-relevant
parameters while freezing 21,208,560 association parameters. Independent
`target_44b6` checkpoints improved external ZSNS005 selection AP from
`0.173924` to `0.277423` and `0.219762`. Their frozen ensemble selected four
of five true events at `0.80` precision on selection. On the disjoint audit,
however, the same threshold selected seven events with only one true event
(`0.142857` precision). Event classification itself failed; the result was not
a second-daughter ranking artifact. The focused external gate is rejected and
no model dataset, candidate, or submission was created.

## Real-domain division probe and next lane

The clean 0.940 prediction graphs contain 225 constrained geometric recovery
candidates in the five annotated division frames. Exactly three are
parent-free recoveries that require only one added edge. A train-only probe
cache now contains the 15 context frames (`t-1,t,t+1`) needed to score all 225
candidates; its 23 Kaggle members total 66,682,915 bytes and every member is
SHA-256 recorded. It reads no competition test file and is diagnostic-only.

The probe also records inference-available branch geometry: existing- and
second-child displacement, daughter opposition, step balance, midpoint drift,
and constant-velocity midpoint error. A fixed biological symmetry score ranks
the three safe positives `2/26`, `1/2`, and `4/84` within their respective
event frames. This is useful complementary evidence, but not precise enough by
itself to authorize recovery.

The next training lane replaces external-only calibration with legitimate
competition-train supervision. A private CPU-only Kaggle packager is staging
all 199 train GEFF annotations; GPU, TPU, internet, inference, and submission
are disabled. The resulting reciprocal inventory will exclude the four final
probe movies, train on real two-daughter parents plus annotated one-child
parents from the same frames, split optimization/selection deterministically
within each source embryo, and run all model optimization on the Antelume A10G.
This supersedes the rejected external-only policy while preserving the
non-replica and no-metric-hack constraints.

The annotation inventory is now complete. Across all 199 official train
movies it finds 151 two-daughter parents; the four final-probe movies contain
five and remain excluded. The reciprocal training pool therefore contains
146 positive division events and 1,247 same-frame annotated one-child PU
controls. A deterministic division-stratified split holds out 5 of 24 usable
`44b6` events and 26 of 122 usable `6bba` events for model selection, leaving
19 and 96 respectively for optimization. The image extractor is a private,
CPU-only Kaggle job; model training and scoring remain assigned exclusively to
the Antelume A10G.

The private extractor completed as Kaggle kernel version 3 in 57.35 seconds
on CPU (`gpu_used=false`). It produced 146 hash-verified NPZ shards with 1,393
rows: 115 positives plus 1,039 PU controls for optimization and 31 positives
plus 208 controls for movie-disjoint selection. The 34,245,696-byte archive
SHA-256 is `46db151310808fa455e63a4c8791f14121125ba6bda0d46b5aa2b1d7913990e0`;
the manifest SHA-256 is
`943717472518b917175312bd4ada9e12660d31d3ebf0bf7afd5672ab40442e1e`.
Every shard size and digest was rechecked after download. No final-probe image
was included.

## Independent CPU morphology control

A project-authored 132-feature temporal morphology control was trained on the
same movie-disjoint inventory without GPU use. Five-fold grouped optimization
selected a histogram-gradient model and an extra-trees model; their best
individual out-of-fold AP was `0.632866`. The frozen ensemble achieved
selection AP `0.688111` and recovered 8 of 31 positives before the first false
positive, freezing probability threshold `0.8992587384`.

On the complete-movie probe, all three safe recoveries ranked first within
their frames and occupied the top three scores across all 225 candidates
(`AP=1.0`). Absolute probabilities shifted downward, however, so the frozen
threshold made zero decisions. The CPU gate is rejected as a standalone
policy. Its unusually clean ranking is retained only as an independent
diagnostic or conservative vote for the Antelume model; the held-out result is
not used to lower its threshold. Model SHA-256 is
`dc21d9ba50e5fbeb339ecea5ab2842afab4bd7c82c35d40c227ce95ff3be8dc6`.
