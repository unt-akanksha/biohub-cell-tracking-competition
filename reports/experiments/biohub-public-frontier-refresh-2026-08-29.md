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
