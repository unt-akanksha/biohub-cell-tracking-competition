# FOCUS detector plus owned flow: source-selection comparison

The previous public-base/owned-flow repair produced zero eligible edges and
was rejected. This route instead changes the detector while retaining the
already-tested owned native image-flow association rule. It is not another
run of the rejected public-neural FOCUS linker or the physical postprocessor.

## Identity and scope

CPU notebook `indarkarhana/biohub-focus-runtime-identity-v1/1` is COMPLETE.
It used neither competition data nor GPU, and did not deserialize the model.
The public Kaggle nuclei weights exactly match author SHA256
`b14a7bd272f824adb1a1073bc3f2af17a95919d5a0c3f1d9011a8d82378d8f3a`,
4,468,804,784 bytes. It recorded29 runtime entries; elapsed53.983s.
Exact CPU source receipt matches the staged script bytes.

The official [model card](https://huggingface.co/Qinghua-thu/FOCUS-3D)
states Apache-2.0 for weights; the [source license](https://github.com/yu-lab-vt/FOCUS-3D/blob/main/LICENSE)
is BSD-3-Clause. No gated download, contact-data sharing or terms acceptance
was performed. This is identity confirmation, not a new model-license discovery.

The [author dataset portal](https://www.quiclab.org.cn/focus-3d/datasets)
lists nuclei data from its own collection and several named laboratories.
It does not provide an exact training manifest for this checkpoint. Therefore
pretraining disjointness from Biohub is UNVERIFIED; do not label a successful
source comparison as fully independent model validation. The full-paper fetch
was unsuccessful. The eight movies are excluded from OUR flow training.

## Small test before full inference

`biohub-focus-source-probe-v1/1` is accepted after8 local tests and a fresh
12.07h Kaggle quota check. Declared cap1h, internal watchdog55min. It preserves
the original frozen raw-detector code/settings and offline dependencies,
removes hidden-test branches and unused historical scorer imports, and checks
actual weights/config/runtime bytes before loading the model on two T4s.

Only first3 frames of the two previously processed training movies
6bba_f1fde7e0 and6bba_23af9eeb are read. Every output centroid must exactly
replay the original raw cache, on GPU and after host download. No labels,
postprocessing, node count targets, score selection or submission are allowed.
Frozen notebook SHA256:
`1ad40fc75748219cc9f6b6b5a1deb8f14cfd67dfd8242ad18dcc0135cc85a6e7`.

Smoke outcome: COMPLETE437.508s; GPU worker phase34.8s. Both three-frame
outputs (785 and285 centroids) exactly replay the original raw arrays after
host download. Verified reportSHA
`02a85eb8740981ca76635385ce237847aade1a8c960af7bbe78ee2e3366ef42b`.
Output verification used a slow recursive reference lookup. The new full
builder replaces it with three explicit mount candidates, tested with glob
and rglob forbidden. The completed smoke notebook was not edited/rebuilt.

## Full source comparison, only after smoke passes

Use the same eight complete source-selection movies from split12eca8b1...:
6bba_283bf9f1,6bba_5b28472a,6bba_c73a1d11,6bba_67ebd073,
6bba_5c039895,6bba_fc5f39dc,6bba_87289e13,6bba_cdcfe533.
No new target-embryo movies. Cache raw detections before any labels. Preserve
every raw centroid when applying native flow3006; no flowD4, public linker,
node pruning, count calibration or threshold search.

Compare current official complete-movie score against the retained parentD4
source0.6298395328 and static linking of the same FOCUS detections. Report
raw edge Jaccard, recall, per-movie and per-embryo values, worst movie, and
division TP/FP/FN. Against parentD4 require positive combined and raw-edge
gains, mean recall loss<=0.005, at least5/8 movies improved, each movie
loss<=0.02 and worst-movie minimum loss<=0.01. Also require positive combined
and raw-edge gains over same-FOCUS static linking. These conditions precede
this route's source scores and cannot be relaxed after observing them.

Passing does not authorize submission: pretraining provenance, target-embryo
validation and exact two-GPU offline full-test delivery remain outstanding.
Historical raw4 detector inference took1193.882s; eight movies are estimated
at roughly40min, not a guarantee. A fresh quota check and declared1h cap are
required before launch. Large model retraining is not scheduled.

## Full cache execution

`indarkarhana/biohub-focus-source-cache-v1/1` is accepted and confirmed
RUNNING. Fresh quota11.94h; declared1h leaves10.94h worst-case. Two T4s,
Internet disabled, no other Biohub GPU job overlaps. Both six-frame smoke
outputs have already been regenerated with the same file hashes in the live
log; the eight complete source movies are still in progress.
Frozen notebookSHA:
`d65eac4ce6354829b13fd92176c3163b45a50efabe2c326f5b5322d7d73f322b`.

21 tests cover the identity audit, smoke/cache builders, explicit nonrecursive
reference resolution and raw-array bounds/geometry. The CPU cache verifier
checks all800 source frames plus6 smoke frames, runtime identity, actual NPZ
files and coordinate replay. No completed source-cache receipt or tracking
score exists yet. After completion, harvest exact version1 and run
`scripts/verify-focus-source-cache.py`; then integrate owned native flow and
complete-movie official scoring.

## Owned-flow integration and automatic follow-up

The one-shot controller is LIVE: PID28520, started2026-09-10T10:08:24Z.
State/logs: `.biohub/automation/focus-source-route-v1/` and sibling stdout/
stderr logs. It waits for exact cache version1, verifies all artifacts, builds
a new immutable source-flow notebook, checks fresh quota before its one-hour
GPU launch, waits for that exact version1, and CPU-scores the outputs. Three-
hour controller bound, no automatic submission or target-embryo access. Do
not manually duplicate its flow launch, cache verifier or final scorer.

The source-flow worker reuses the exact GPU-tested flow model, sampling and
linking implementations from the completed full4 diagnostic. Before source
inference it must replay the original3-frame flow sample exactly on the new
detector's identical285 coordinates. It then produces static and flow graphs
for all8 source movies. Every centroid is retained. The host replays all graph
edges from saved motion caches before opening GT, and freshly re-scores the
retained parent graphs as the third arm.18 worker/contract/scoring tests and
6 controller tests pass, including reserve rejection, no duplicate launches,
no ambiguous-push retry and transient observation handling.

Baseline restoration/check completed on CPU: missing saved source/runtime
files were downloaded from exact `biohub-detector-spatial-tta-selection-v1/1`.
All8 graphs and219,373 nodes verified before labels. Fresh official re-score
is exactly0.6298395328208314; every per-movie dictionary and summary equals
the original pinned report. This is control verification, not a new gain.

The public runtime README (SHA40f1b27a55d3225fb16f21a07aeb064ccd7b24022268b3b6b0f8d07bb79d59d3)
confirms the weight license but provides no training split. Separate CPU-only
`biohub-focus-checkpoint-metadata-v1/1` is accepted after3 tests to inspect
non-tensor metadata with restricted `torch.load(weights_only=True,mmap=True)`.
No unsafe-load fallback, GPU, competition input or model construction.
It completed65.308s. Actual downloaded receipt/source hashes and exact remote
CPU-only version1 were verified. The checkpoint records iteration39999 and
contains optimizer/trainer state; the bounded inspection did not establish a
dataset/split/config manifest. Deeper metadata was truncated, so this is not
proof that no such records exist. No pretraining-disjointness claim is made.
See `focus-checkpoint-metadata-v1-result.json`.

Correct prior architecture-size language: checkpoint bytes are NOT all model
weights. Do not infer1.1B model parameters from its4.47GB size. This inspection
visited887 tensors/411,647,757 elements, not a complete verified parameter
count. A model-only export could reduce packaging overhead, but should be
tested only if this candidate passes validation; no current weights changed.

## Completed comparison and decision

Controller terminal completed10:33:11UTC. Detector cache elapsed2090.451s;
flow elapsed133.158s. All800 source frames/168,586 original detections and
6 detector replay frames verified. Exact3-frame motion replay passed before
source inference. CPU replay reconstructed all graphs before scoring, and
the freshly scored parent exactly matches its pinned original report.

| Arm | Combined score | Raw edge Jaccard | Mean node recall | Division TP/FP/FN |
| --- | ---: | ---: | ---: | --- |
| Retained parentD4 | 0.6298395 | 0.6798300 | 0.9575953 | 2/265/9 |
| FOCUS static | 0.7496408 | 0.7691581 | 0.9643974 | 0/145/11 |
| FOCUS owned flow | 0.7753326 | 0.7937316 | 0.9643974 | 3/186/8 |

Seven of eight movies improve versus parent, but6bba_67ebd073 loses0.0381222
adjusted-edge score (limit0.02). The source gate FAILS without relaxation.
No transfer launch or submission. Preserve this substantial component gain;
do not represent it as a promoted model or public-best score.

Failure localization from the paired complete-movie rows: on67ebd073 the
static FOCUS arm already loses0.0328582 versus parent. Adding motion adds4
correct and12 false edges, worsening this movie another0.0052640. FOCUS
recall is higher (0.9761905 vs0.9588745), but its27,279 detections incur a
larger official node-count adjustment than the parent's19,932. This is not
evidence that arbitrary count pruning is valid; no detections were removed.
The weakest absolute movie remainsc73a1d11: FOCUS recall0.8127544 versus
parent0.8643148, despite adjusted-edge gain0.0494644. Association robustness
and missed detections both remain relevant; pure motion is not the sole
cause of the regression. These are source diagnostics, not fresh validation.

Result SHA256:
`896d3de86fa30fcd860f2c16073d47718118d94e1a43fe70710626c7b32ea169`.
Frozen flow notebookSHA:
`d36fd6ef5fb460ae4964429aded2832b0d44db91a65836476ecea64639f30375`.
No GPU follow-up queued. Source/model bytes and completed notebooks unchanged.
