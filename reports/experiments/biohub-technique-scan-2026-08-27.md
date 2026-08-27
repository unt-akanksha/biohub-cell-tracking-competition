# Clean Biohub technique scan — 2026-08-27

## Decision

Do not revive or submit a public-notebook replica. Keep the currently running
project-authored temporal-patch v1 and the predeclared v2 candidate-pair head
unchanged so their evidence remains interpretable. Prepare a separate v3 lane
that combines real ZebraHub appearance pretraining, transition-reliability
gating, and candidate-edge context. It may run only after the reciprocal
competition calibration gate and must retain the exact two-GPU, whole-movie
inference contract.

No public prediction, public notebook source, leaderboard result, competition
submission, or metric-hack mechanism was copied or used for selection in this
scan.

## Public notebook and discussion audit

The current high-vote clean notebook families remain nearest-neighbor/global
assignment, dual-seed detector blending, division augmentation, and learned gap
recovery. Explicit metric-hack notebooks and the patched division exploit were
excluded. These families do not supply a project-owned temporal image model and
therefore do not change the originality conclusion for v1/v2.

The official HOCT paper identifies an edge-representation limitation that is
directly relevant to v2: divisions can entangle node embeddings, while an
independent edge score cannot compare its motion with nearby candidate motions.
Its strongest general lesson is to contextualize candidate edges using geometry,
not to aggregate over the non-homophilic candidate-graph adjacency. The paper
also reports that representation structure mattered more than a deep pretrained
image encoder. Source: <https://arxiv.org/abs/2607.11754>.

This project already tested the public HOCT checkpoints independently. The
completed Biohub probe retained 126,582/126,590 eligible training edges but
regressed the clean selection movies by 0.05384 and untouched acceptance by
0.004340, while adding division false positives. That is decisive evidence
against a HOCT-only or HOCT-dominant submission. No HOCT source or weights will
be copied into v3; only the research-level edge-context hypothesis is retained.

The competition discussion also documents exact repeated frames, global
whole-volume jumps, and a low rate of sparse edge annotation errors:

- <https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/724283>
- <https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/729053>

These observations expose two current training risks. A duplicated image pair
cannot provide genuine appearance discrimination even if its sparse graph
changes, and the fixed 32 µm raw-displacement training radius can skip a
whole-volume jump rather than learn it. V3 should therefore compute a
label-free transition reliability/global-shift estimate from the images,
downweight appearance evidence on effectively duplicated frames, and represent
residual motion after the global shift. Sparse links that are severe local
motion outliers should receive robust weighting, not be silently treated as
equally reliable positives.

## ZebraHub external-data opportunity

The organizer explicitly confirmed that all public ZebraHub images and
`*_tracks.csv` annotations are allowed and do not overlap the competition test
set:
<https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/734330>.

The public single-objective inventory contains full OME-Zarr movies and dense
lineages for ZSNS001, ZSNS003, ZSNS004, and ZSNS005. ZSNS004 and ZSNS005 use
physical spacing `(1.24, 0.439, 0.439)` µm at level 0 and
`(2.48, 0.878, 0.878)` µm at level 1, close enough to the competition spacing
for the existing physical 17-cubed resampler. Sources:

- <https://public.czbiohub.org/royerlab/zebrahub/imaging/single-objective/>
- <https://public.czbiohub.org/royerlab/zebrahub/imaging/single-objective/ZSNS004.ome.zarr/.zattrs>
- <https://public.czbiohub.org/royerlab/zebrahub/imaging/single-objective/ZSNS005.ome.zarr/.zattrs>

The external split is frozen before any model result:

- ZSNS004: external pretraining only;
- ZSNS005: external validation only;
- competition reciprocal folds: fine-tuning/calibration only, unchanged;
- ZSNS003: excluded until its published multiscale shape/scale inconsistency is
  resolved;
- ZSNS001 and its tail: reserved, not used in v3 selection.

Raw movies are hundreds of gigabytes, but their Zarr chunks support range-safe
selective extraction. The derived training asset should contain only
hash-bound, normalized temporal 17-cubed patches, physical coordinates,
candidate masks, and lineage targets for deterministic time blocks. Raw public
movies must not be republished. The source manifest must retain URLs, response
sizes, hashes, physical transforms, citation, organizer authorization, and a
declaration that no competition test data or public predictions are present.

## Frozen v3 scientific direction

1. Pretrain the project-authored physical 3D encoder and pair head on dense
   ZSNS004 temporal patches; select pretraining checkpoints only on ZSNS005.
2. Fine-tune reciprocally on the existing disjoint Biohub folds. ZebraHub must
   never enter Biohub calibration or processed acceptance.
3. Add a bounded transition token containing duplicate-frame confidence,
   label-free global shift, and local motion-density summaries.
4. Refine candidate-edge representations with a project-authored edge-set
   context block. It may compare geometrically nearby edge motions but must not
   copy HOCT code, architecture constants, checkpoints, or predictions.
5. Preserve an independent v2 pair-head control so edge context is admitted
   only if it improves ZSNS005 and both Biohub reciprocal folds.
6. Retain exact zero-appearance/zero-context calibration controls, strict
   per-movie regression gates, the unopened processed-acceptance set, and the
   no-leaderboard-selection rule.

This direction addresses same-organism representation coverage and known
transition failure modes. Merely increasing encoder width is lower value until
these structured sources of error are measured.

The project-authored v3 adapter is also wired to the native Biohub training
contract: overlapping `t-1,t,t+1` temporal volumes supply their central
adjacent frames to the label-free estimator, physical node coordinates produce
residual candidate context, and the contextual head returns the same dense
finite/`-inf` candidate matrix expected by the all-positive loss. Synthetic
integer-shift tests recover the exact physical shift and backpropagate finite
gradients through source embeddings, target embeddings, and division logits.

## Selective extraction smoke

The CPU/network-only extractor completed one frozen ZSNS004 transition at
`t=100`. It read the exact 377,693,868-byte lineage CSV plus 24 level-1 chunks
(76,897,723 bytes), then emitted a 2,886,389-byte derived shard with 17 division
sources, 34 true daughter links, 43 hard-negative candidate links, 113
normalized temporal patches, and 18 image/motion context values per eligible
edge. The label-free full-frame estimate found zero global shift, aligned NCC
0.982819, phase-peak margin 0.985382, reliability 0.975636, and zero duplicate
confidence. A fresh 20,747,761-parameter v3 model completed a finite
forward/backward pass: all 77 candidate logits were finite, the combined
contextual-pair-plus-embedding loss was 1.795639 with pinned seed 31003, and all 64 populated gradient
tensors were finite. The unchanged 20,869,325-parameter v2 smoke remains an
independent control. Exact hashes and shapes are recorded in
`reports/experiments/zebrahub-temporal-patch-shard-smoke.json`.

The frozen ZSNS005 external-validation source independently completed the same
pipeline at `t=100`: 15 division sources, 30 positive links, 32 hard negatives,
62/62 finite contextual logits, and 64/64 finite populated gradient tensors.
Its label-free transition reliability was 0.987244 with zero duplicate
confidence. This is preprocessing and gradient evidence only; the random
initialization losses are not model-selection evidence.

This proves feasibility only; it does not authorize ZebraHub training before
the existing reciprocal competition calibration gate.

## Downstream v3 integration evidence

The v3 lane now has a complete project-owned path from reciprocal training to
whole-movie inference. Its trainer is a thin, fail-closed configuration of the
tested v2 reciprocal engine: one isolated fold worker per GPU, exactly two
visible GPUs at orchestration, the repaired effective Biohub inventories of 96
and 45 source-prefix movies, EMA checkpoints, and no submission command. The
adapter refuses to train without both source and target temporal volumes, so it
cannot silently degrade to the context-free v2 head.

At inference, each complete movie is processed transition-by-transition with a
two-frame rolling cache. The same label-free global shift, duplicate evidence,
reliability, and residual-motion features used in training are calculated from
the real movie frames. Only Trackastra candidate edges receive contextual v3
probabilities; non-candidates remain neutral at 0.5. The exact model-family
metadata is verified before calibration, processed acceptance, or candidate
materialization, and every downstream call supplies the movie image explicitly.

The portable private runtime builder now emits a distinct
`contextual_pair_fusion_v3` package containing the transition estimator,
contextual model, Biohub adapter, two-GPU trainer, calibration, processed
acceptance, and whole-movie candidate builder. Focused regression coverage
passes for the unchanged v2 trainer, v3 metadata/configuration, image-dependent
whole-movie scoring, downstream acceptance contracts, and all three runtime
families. This is implementation evidence, not scientific promotion evidence;
v3 remains staged until the running v1 experiment and clean calibration gates
produce terminal results.

The external-pretraining requirement is now executable and fail-closed rather
than aspirational. Two independent GPU-isolated v3 workers optimize only
hash-verified ZSNS004 derived shards and select EMA checkpoints only on
disjoint ZSNS005 shards. The default evidence floor is 64 training transitions
and 16 validation transitions. Dataset manifests must prove the frozen source
role, organizer authorization lineage, exact bytes/hash, 18-value context
contract, and absence of competition data, public predictions, leaderboard
selection, or submission activity. Biohub v3 fine-tuning now refuses to start
unless both external folds improve after step zero and the aggregate and worker
terminals bind the exact checkpoint hashes. Randomly initialized v3 can no
longer masquerade as the predeclared same-organism candidate; v1 and v2 remain
unchanged controls.
