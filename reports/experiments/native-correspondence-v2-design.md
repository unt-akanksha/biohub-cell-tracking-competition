# Native visual correspondence v2

## Hypothesis and scope, frozen before training

The v1 CNN/transformer ensemble failed its cross-embryo diagnostic. Do not reuse
its failed weights or retune its mixture on the eleven old target examples.
Instead, address concrete data limitations: strided XY decimation, insufficient
temporal coverage, and GT-centered query crops differing from inference.

Movie roles remain identical: 91 optimization movies and 10 selection movies.
All twelve prior full-movie validation/public-test-copy exclusions and sealed
audit boundaries remain closed. Optimization uses up to sixteen uniformly
spaced annotated transitions per movie; selection uses every annotated
transition of its existing movies. This is broader coverage, not a pristine new
holdout: the selection movies were already used in v1.

| Embryo / role | Movies | Known annotated edges | Requested frames |
| --- | ---: | ---: | ---: |
| 44b6 optimization | 27 | 1,471 | 864 |
| 44b6 selection | 3 | 536 | 300 |
| 6bba optimization | 64 | 9,643 | 2,048 |
| 6bba selection | 7 | 5,931 | 698 |

These are available annotations, **not eligible model examples**. Query proposal
recall and ambiguous parent exclusions must be reported explicitly after the
build. Do not compare these counts directly with v1's 4,254 eligible training
groups or 191 eligible selection groups as though eligibility were identical.

## Image and label contract

Authorized Kaggle ZIP range reads verify ETag/range, local ZIP headers, member
names, bounded decompression and CRC. Exactly 4,112 selected members total
16,833,720,820 bytes. No public predictions or competition-test images are read.
Native Blosc uint16 arrays are (64,256,256), physical spacing
(1.625,.40625,.40625) micrometers. Normalize native intensities by .01/.995
quantiles; XY mean-pool 4x4 only for image-only DoG proposals. Proposal positions
include the physical block-center offset. DoG uses the original SciPy reflect
boundaries and .7/1.5 sigmas, threshold .025, 3-voxel local maxima; >4096 aborts.

One-to-one assignment associates annotated child locations with image proposals
within 3.25um. Query crops remain centered at the **image proposals**. Select up
to sixteen parent proposals within 20um before reading parent labels. A parent
within 3.25um can be positive; competing parents within 7um are masked from the
loss. Missing-but-near true parents cause omission, not a false null label.
Unknown children are never negative examples. All omitted queries/edges count
toward the reported data-coverage ceiling.

Extract three native-image trilinear 15-cube patches at .8125,1.625,3.25um
spacing; finest Z detail is interpolation, not invented resolution. Model crops
are 11-cubes, physically consistent .8125um jitter, shared XY rotations and
brightness augmentation. Synthetic true-parent dropout is 20% during training.

## Resource contract and real smoke

Antelume A10G only, no simultaneous foreign GPU job, two Torch CPU threads,
bounded four-thread image reads. Preserve 3GiB available system RAM; temporary
data stays under a new owned `/dev/shm` directory, at most 3GiB. Do not format
the unidentified NVMe or delete RSNA/shared environments. GPU data preparation
has a 45-minute hard limit and per-movie recoverable manifests. Training and
preparation are sequential. No GPU memory reset or Linux cache drop is needed.

Twelve focused CPU tests passed. Real smoke, fixed first four optimization
transitions from the first movie of each embryo, passed in 6.2334 seconds:
17 usable positive groups across seven packets, 1,497,030 bytes. Of 44 known
edges, 23 child queries matched image proposals; six ambiguous rows were
omitted. GPU vs CPU native-patch maximum absolute error was .00046623 after
float16 storage. This proves functionality, not complete cell-detection recall.

Data code manifest: `2e8dc8368961e1ec9fb1f8b9c540180164bd19b6de61cea8a6ca6c363e1cd1c8`.
Movie plan: `60300fbc7251f842e80b3ae1687e78d097b7dcda876908f3832b34174c0a0ef6`.
Full data job PID 23364 started 2026-09-14 04:24:47 UTC; max 45 minutes.
Status at 04:30 UTC: 64/101 movies, 663,626,448 bytes. See terminal receipt for
later authoritative state; this sentence is explicitly a historical snapshot.

## Model contract

Same independent residual 3D CNN and candidate-context transformer families as
v1, now three input channels, random initialization. Train each architecture on
each source embryo sequentially: four experts, 8,000 actual updates each,
batch12, AdamW 2e-4 cosine to2e-6, WD.01, FP16AMP, clip2. Source selection every
500 updates; best NLL checkpoint only. Save weights, rolling optimizer/scaler
and RNG state. No automatic resume is claimed by this trainer.

Both architectures must pass real-data smoke, exact checkpoint reload and
timing before a full launch. Full training max90min; reject launch if measured
projection exceeds85min. No more seed/mixture/threshold trials on target labels.
Source admission requires >=2% NLL improvement, no fewer correct parent choices,
and missing-parent NLL nonregression. Only source-admitted components can enter
cross-embryo diagnostics. Equal probability mixture is frozen in advance, only
if both components individually pass source admission. Never average weak
members merely to increase model count.

Training contract: `dd71155624ed7a3c627e1218720b8d8a639602349a62142b5498ae89f34310bf`.
Full training is **not yet launched** at this design snapshot. Existing Kaggle
submission 56219125 is unchanged; no v2 score or submission readiness is claimed.
Conditional diagnostic success is insufficient: require complete-movie patched
official scoring, per-embryo/worst-movie nonregression and Kaggle runtime acceptance.

## Native data build complete; model smoke started

All 101 movies completed in **504.060 seconds (8.40 minutes)**. Dataset packets
total **1,216,826,026 bytes**. No storage/memory limit was hit and no RSNA files or
processes were modified. Authoritative manifest:
`reports/experiments/native-correspondence-v2-data-result.json`.

| Embryo / role | Known edges | Matched image queries | Positive | True null | Ambiguous omitted | Eligible |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 44b6 optimization | 1,471 | 1,116 | 973 | 18 | 125 | 991 |
| 44b6 selection | 536 | 378 | 314 | 0 | 64 | 314 |
| 6bba optimization | 9,643 | 5,029 | 3,912 | 34 | 1,083 | 3,946 |
| 6bba selection | 5,931 | 3,495 | 2,776 | 61 | 658 | 2,837 |

Usable optimization groups increase from 4,254 to **4,937** (~16%), not 2.6x;
usable selection groups increase from 191 to **3,151**. Input fidelity/query
placement and selection coverage are the principal changes. Proposal recall
remains limited, especially 6bba; conditional association results cannot be
presented as full-movie tracking scores or detector recall improvements.

The three-channel CNN has 9,015,138 parameters; the transformer has 15,850,722.
Both passed eight CPU Torch contracts. Real-data training smoke PID **29459**
started **04:33:20 UTC**, thirty updates per architecture on source6bba with
complete source selection/reload checks. Full four-member training is still
gated on its result/timing. Dataset local backup was launched concurrently
(transfer only, no second GPU experiment); do not claim verified recovery until
the harvest receipt exists.

## Timing guard caught an excessive first projection

First real training smoke passed both models and exact checkpoint reload in
56.585 seconds. However, dividing the entire short training stage (including
checkpoint writing and startup) by thirty updates projected **7,374.56 seconds**
for four 8,000-update fits. That exceeds the frozen 5,100-second launch cap.
**No full training was launched from that proof.**

Timing-only revision r2 separates updates from validation and checkpoint I/O:
fifty smoke updates, discard the first twenty for warm-up, CUDA-synchronize
each measured update, then budget max(1.25 x mean, p95) plus measured paired
evaluation and 240 seconds fixed overhead. Architecture, data, optimizer,
admission gates and full 8,000-update budget are unchanged. Rerun full source
selection/reload smoke under the new code identity before any full launch.
This is a runtime measurement repair, not a score-based model retry.

R2 training contract:
`cf0a15f9b46d848de9f7f72c3064683ecc6ca8e795e0da8d9b8dc51183e29bda`.
First smoke artifacts backed up and size/SHA verified: **289,911,647 bytes**.
Only its redundant remote rolling optimizer checkpoint and two smoke best
weights were removed, recoverable from the local backup. Reports remain remote;
RSNA/shared files and all earlier full-training best weights are untouched.
Receipt: `native-correspondence-v2-training-smoke-harvest.json`.

The r2 functional smoke also passed, but its conservative steady-state projection
was **10,218.01 seconds**, again rejecting full launch. Per-update times alternated
around .02-.03s and .2-.3s. Code inspection found cuDNN autotuning enabled while
the sparse encoder batch cardinality changes each update; repeated shape-specific
algorithm searches are the runtime hypothesis. R3 disables cuDNN benchmarking
without changing architecture, training data, optimizer or admission gates, and
repeats the same fifty-update smoke. Do not claim the speed fix before measurement.
R3 contract: `bda044511d3f37b33caa76d5f226fa64c4542d2705d304749e60f95c52869593`.

R2 smoke artifacts also backed up (289,913,817 bytes), then only its three redundant
remote smoke checkpoints removed with local+remote digest checks. All recoverable.
No foreign project or shared environment touched.

The **complete native dataset is now locally backed up and SHA/size verified**:
2,103 packets, 1,216,826,026 bytes, manifest
`f2861ce9f21307507ed521716ca3bf9a7adf3b76ce9f8216e4407233616c693a`.
Transfer plus verification took 740.235 seconds; data remains remote for training.
Local root `.biohub/cache/native-correspondence-v2-data`; receipt
`native-correspondence-v2-data-harvest.json`.

Prepared an evaluation-only reusable-node-embedding head factorization to avoid
re-encoding every parent for every query at inference. Both architectures passed
nonzero-head CPU comparisons: factorization bit-identical, different encoder
batch partition max error 9.53674e-7 with identical choices. This is not a GPU
throughput result, not full-movie validation, and not yet integrated into Kaggle.

## R3 passed; full training actually launched

Disabling cuDNN autotuning removed the large latency spikes in the matching
real-data smoke. Both architectures passed fifty actual updates and exact
selected-checkpoint reload in **40.916 seconds**. Conservative update budgets
(max mean x1.25, p95) are .05316s CNN and .06283s transformer, paired selection
evaluation 4.778s and5.430s. Four 8,000-update fits now project **2,422.51 seconds
(40.38 minutes)**, below the unchanged 85-minute launch cap.

Full training PID **31430** started **2026-09-14 04:49:17 UTC** on Antelume A10G,
one member at a time: 6bba CNN, 6bba transformer, 44b6 CNN, 44b6 transformer.
Expected training completion around **05:30 UTC**; hard watchdog **06:19:17 UTC**.
These are training estimates, not submission readiness; cross-embryo and complete
movie/runtimes gates remain. The process yields/checkpoints for a foreign GPU
job and never kills another project. Instance remains on/billing after completion;
no automatic shutdown is authorized or configured.

Receipts: `native-correspondence-v2-training-smoke-r3-result.json` and
`native-correspondence-v2-full-training-launch.json`. Cloud outputs:
`/tmp/biohub-image-context-v2.ScdSdY/native-correspondence-v2-training-full`.
Live log is its sibling `native-correspondence-v2-training-full.log`.
Runtime bundle is `native-correspondence-v2-training-r3-bundle`, contract
`bda044511d3f37b33caa76d5f226fa64c4542d2705d304749e60f95c52869593`.

Next: once terminal, back up weights/optimizer/RNG and evaluate only source-
admitted members using `scripts/evaluate-native-correspondence-v2.py` on the
opposite embryo's existing full selection coverage. Its weights are fixed .5/.5
if both components are admitted; no target threshold/weight search. A conditional
pass still needs frozen complete-movie integration and official score/runtime
acceptance before a new submission. Do not poll idle status repeatedly while
training can run; inference factorization and data checks are ready for follow-up.

### Autonomous successor and backups

Conditional cross-embryo successor is now queued, CPU watcher PID **32305**,
launched **04:59 UTC**. It requires the exact completed four-member training
contract, waits for its GPU process to exit, refuses foreign GPU occupancy, and
runs only the frozen conditional diagnostic (five-minute hard limit). It cannot
submit, change mixture weights, train additional members, or stop the instance.
Successor manifest SHA
`1a4d5663cc15e4ccaad1cc402afbcc024c71fb3f829b859e800739f56f005065`.
Remote terminal output `native-correspondence-v2-cross-embryo-result.json` and
queue receipt `native-correspondence-v2-cross-embryo-queue-result.json` under the
owned root. Local launch receipt `native-correspondence-v2-cross-embryo-queue-launch.json`.

R3 smoke weights/optimizer/RNG/history also backed up and verified locally:
289,913,855 bytes. Its remote copies are retained while full training is active.
Fifteen focused archive/data tests passed in7.12s; diff check has no errors
(pre-existing LF/CRLF warning only). The existing submission remains unchanged.

## Full training and frozen cross-embryo diagnostic complete

All four models completed8,000actual updates each, **1,437.561 seconds (23.96min)**
total. Source6bba CNN selected step5,500:2,744/2,837 correct,NLL.110654 vs
distance2,629/.696594. Source6bba transformer selected step3,000:2,684/.220477.
Both source gates pass. Source44b6 CNN selected step500:310/314 vsdistance312,
so rejected despite NLL.068253. Its transformer passes source at312/314,
NLL.069460, but must still pass opposite-embryo diagnostics.

Frozen opposite-embryo diagnostic took15.262seconds:

| Train6bba -> selection44b6 | Correct /314 | NLL | Missing correct /314 | Missing NLL | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Distance | 312 | .569383 | 0 | 7.389667 | Comparator |
| CNN | 312 | .120457 | 289 | .290737 | PASS |
| Transformer | 312 | .075223 | 231 | .717893 | PASS |
| Fixed equal mixture | 312 | .077953 | 277 | .388666 | PASS |

Both networks miss the same two true-parent cases; no unique correct parent
choices. The ensemble's evidence is improved calibration and missing-parent
handling, **not greater parent-ranking accuracy on this fold**. It is not a
Kaggle/tracking score. The reverse source44 transformer gets2,622/2,837 vsdistance
2,627, fails; do not include either source44 model. Source-vs-cross comparator
counts differ by two near ties on6bba after softmax rounding, while NLL agrees;
reverse rejection holds against either2,627 or2,629, so admission is unaffected.

The predeclared CNN-first admission rule selects exactly the two source6bba
models with .5/.5 weights. No target weight/threshold search. Training terminal
SHA`42224d3762ee3c7ab9bc5c6997e5efd5cf72c5eb5e848b1925b69c4500ef7a46`.
All389,800,593bytes of full training weights/history/optimizer/RNG backed up and
SHA/size verified locally; remote copies retained. See training-full-harvest,
training-full-result and cross-embryo-result receipts in this directory.

GPU cache probe on fixed first optimization packet (four groups,24unique patches)
passes both selected trained models with **zero logit error and identical choices**.
This tiny probe establishes functionality, not full-movie speed. A native-image
graph-repair smoke is now the next stage; see `native-graph-repair-v2-design.md`.
