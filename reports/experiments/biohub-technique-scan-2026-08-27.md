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

## Production ZebraHub pretraining asset

The frozen 64/16 external split is now materialized rather than represented by
single-transition smokes. ZSNS004 contributes 64 training transitions sampled
evenly across four developmental windows; ZSNS005 contributes 16 disjoint
selection transitions across four separately frozen windows.

A prelaunch label audit found that dataset version 3 was structurally valid but
scientifically biased. At ZSNS004 `t=96`, the complete lineage frame contains
10,532 ordinary sources and only 46 division sources. The former selector put
all divisions first, exhausted the 96-target budget, and then discarded most
ordinary sources for lacking a selected hard negative. Across training this
left only 984 sources, 4,065 candidate edges, and 860 division sources: 87.4%
division labels from a frame population where divisions are rare. Version 3 is
therefore excluded without spending GPU time.

The replacement uses seeded local physical neighborhoods, an initial 12.5%
division enrichment quota, a hard 25% per-shard division cap, and requires both
ordinary links and hard negatives. Version 4 contains 4,009/1,010 source nodes,
133,632/34,884 candidate edges, 4,434/1,124 positive edges, and 425/114 division
sources in train/validation respectively. Division prevalence is 10.6%/11.3%,
and training hard negatives increase from 2,221 to 129,198. The 331,655,946-byte
asset contains normalized temporal patches, coordinates, masks, targets, and
context only—no raw CSV, Zarr metadata/chunks, competition data, or public
predictions.

The private Kaggle dataset
`indarkarhana/biohub-zebrahub-contextual-shards-v1` version 4 is the only
admissible remote asset. Version 1 is excluded because the CLI default skipped
the shard directories; version 2 is excluded because its bundled verifier
incorrectly required Kaggle's consumed publication-metadata file; version 3 is
excluded for the division-first sampling bias above. A complete version-4
remote redownload passed all 80 shard and manifest hashes plus array inventory,
shape, mask, hard-negative, balance, finiteness, split, and raw-file checks. Its
authoritative manifest SHA-256 is
`b35738f215413f1ece403ba5c0601adea82e2540c65f37e6465de0d0755cb7bf`.

The v3 pretrainer now runs this complete verifier inside each isolated GPU
worker before reading a shard. Thus a partial Kaggle mount, altered archive, or
source/provenance mismatch fails before optimization. This publication and
verification used no Kaggle GPU and created no competition submission.

The rebuilt portable v3 runtime contains 36 hash-bound source files and passed
its independent integrity check over 534,585 bytes with manifest SHA-256
`e7025eb3c6fbb8b648e540f20a5dd4da9aa0982f478efe8bbd999428db04099e`.
It declares exactly two visible GPUs and contains no submission command. The
focused dataset/runtime checks passed 20/20, and the latest temporal,
contextual, ZebraHub, and appearance-contract regression set passed 100/100. No pretraining launch is
authorized while the current two-GPU v1 evidence gate is still running.

## Staged original pretraining kernel

The code-only runtime is private Kaggle dataset
`indarkarhana/biohub-temporal-contextual-pair-fusion-runtime-v3`. Version 1 is
superseded because it predates microscopy-safe augmentation; version 2 forced
per-step GPU synchronization; version 3 accepted the division-biased shards;
version 4 allowed a post-publication flag to perturb rebuild hashes. Version 5
lacks the reciprocal-parent objective and cached AMP validation; version 6
leaves contextual candidate-edge inference in FP32. Version 7 is the only
admissible pretraining and inference runtime. A complete remote redownload
reproduced the 36-file, 534,585-byte integrity inventory and exact
runtime manifest above. This publication used no GPU.

The deterministic private notebook
`indarkarhana/biohub-zebrahub-contextual-pretrain-v1` is built locally but has
not been pushed or started. Its only inputs are the private runtime and the
hash-pinned ZebraHub version-4 derived shards; it does not attach the Biohub
competition. It refuses any machine other than exactly two visible GPUs, runs
two isolated 20.7M-parameter folds, caps each worker at 21,600 seconds, selects
only on ZSNS005, and writes checkpoints plus evidence without a submission
command. Four notebook-contract tests pass, including deterministic bytes and
both direct-directory and archive-style Kaggle mounts. Launch remains gated on
the current v1 terminal and the eight-hour Kaggle reserve.

## Microscopy-safe regularization before launch

The first staged pretrainer replayed only 64 labeled ZSNS004 transitions for
12,000 steps without image augmentation, creating an avoidable memorization
risk. Primary microscopy studies support domain-specific views rather than
natural-image policies: Cell Painting DINO work removes scale changes at fixed
magnification and adds rotations, while a separate augmentation ablation finds
flip plus intensity variation strongest and resizing harmful. Trackastra also
shows that full spatiotemporal detection context is central for division-aware
association. Relevant sources are the
[Cell Painting DINO study](https://www.nature.com/articles/s41467-025-66778-6),
[morphological SSL augmentation ablation](https://www.nature.com/articles/s41598-025-88825-4),
and [Trackastra](https://arxiv.org/abs/2405.15700).

The training-only path now has a seeded `microscopy_v1` policy: fixed-scale XY
quarter-rotations, independent axis flips, per-patch/channel gain, and mild
Gaussian noise. ZSNS005 remains byte-identical and unaugmented, physical-scale
cropping/resizing is forbidden, and `none` remains an explicit future ablation
mode. The policy and validation mode are written into each worker and aggregate
terminal. CPU determinism, shape, finiteness, range, and fail-closed contract
tests pass. The final implementation derives the discrete spatial transform
from the worker's seeded CPU sampler and keeps gain/noise generation on-device,
avoiding eight GPU synchronization points per optimization step. No GPU was
used for this change.

## Reciprocal lineage objective and runtime hardening

A final prelaunch objective audit found that v3 compared both outgoing and
incoming edge sets in its contextual head, but optimized only the outgoing
child ranking. This left the biological at-most-one-parent constraint to be
learned indirectly. Trackastra's published formulation applies a parental
normalization and its linking stage enforces at most one parent and at most two
children per vertex. The v3 loss now preserves the all-positive outgoing
daughter objective and adds a fixed 0.35-weight incoming-parent ranking term.
Only labeled targets with at least one competing candidate parent contribute;
uncontested targets remain neutral. In the frozen balanced assets, 3,948 of
4,434 training links and 1,019 of 1,124 validation links have this reciprocal
signal, and every labeled target has exactly one true parent. This is a
predeclared structural improvement, not leaderboard tuning. Primary evidence:
[Trackastra paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09819.pdf).

The same audit addressed the previous timeout risk without reducing training
coverage. Each isolated worker now verifies and preloads the immutable 64/16
shards once instead of decompressing and transferring one archive on every
step, and ZSNS005 validation uses CUDA FP16 autocast just like training. Shallow
dictionary copies isolate augmentation assignments from the cached tensors;
validation stays unaugmented. The new loss, empty-competition behavior,
gradients, cache completeness, assignment isolation, metadata propagation, and
downstream fail-closed contracts are covered by focused CPU tests. No GPU,
competition data, leaderboard feedback, or submission was used.

The exact balanced production shards were replayed with seed 31003 after this
change. ZSNS004 `t=100` produced 2,800/2,800 finite candidate logits, outgoing
loss 3.771015, incoming-parent loss 3.378578, total weighted loss 5.882180, and
64/64 finite populated gradients. Disjoint ZSNS005 `t=96` produced
1,258/1,258 finite logits, outgoing loss 2.846684, incoming-parent loss
2.871466, total weighted loss 4.592559, and 64/64 finite gradients. These
random-initialization values are implementation smokes only; checkpoint
selection still requires improvement on the complete frozen ZSNS005 inventory.

The production v3 candidate scorer now also evaluates its edge-token and
contextual edge-head MLPs under CUDA FP16 autocast. Node-patch encoding and the
base Trackastra scorer already used autocast; leaving the dense candidate-edge
head in FP32 would have preserved a large avoidable T4 bottleneck. CPU behavior
is unchanged, outputs are converted to finite FP32 probabilities, and the
two-GPU whole-movie sharding and ten-hour hard stop remain mandatory.
