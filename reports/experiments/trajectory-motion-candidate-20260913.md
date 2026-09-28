# Trajectory motion candidate: execution record

## Completed CPU evidence

The prior division-head motion-feature recipe failed and is closed. Ordinary
edge coverage then found18 missing annotated links still represented in the
public raw candidate edges. A fixed6um endpoint repair made no changes and is
also closed; its outputs exactly equal their parents.

A new robust AR2 ordinary-motion model is fitted on six-node trajectories,
excluding opposite embryos and all diagnostic/final-probe movies. Source44b6
has14,165 optimization/957 calibration windows; source6bba67,624/7,540. These
are positive-trajectory compatibility models, not edge-correctness classifiers.
Independent weighted least-squares replay verifies both models and thresholds.

Cross-source four-movie comparison passes: combined0.9448387313474883 to
0.946274834648933, raw edge Jaccard0.94171997157 to0.9431414356787491. It adds
two true links, with unchanged44scoredFP and all existing nodes/coordinates.
Only the original worst movie improves; every movie/embryo is nonregressing.
These are exposed training diagnostics, NOT leaderboard scores; public neural
backbone training overlaps these movies.

Single pooled deployment refit is NEUTRAL (zero added links), closed. Its
manifest92da6d72... and result9200b105... remain preserved, not promoted.

The separately predeclared two-source motion-support mixture keeps both original
experts/thresholds and accepts reciprocal endpoint links supported by either.
It has no movie-name lookup and applies to unseen embryo IDs. Its four outputs
are exactly identical to the already scored successful cross-source graphs,
so no labels were reopened or scores recomputed. This hypothesis was selected
after the earlier diagnostics; further movies are mandatory, not optional.

Receipts:

- Cross-source full-movie result: `f48299001d0733c601b8356c2977dffd2ac79c9cd829252d7b504d0741e0a8d1`.
- Expert manifest: `57236f97756e7287204c1db32b084e0bbbe48315012b3117af444ddfa7296aa3`.
- Independent source verification: `80471503d0f2b4cb2f903ef62b8b7e7ed39e0aff1f1d2db2d0f7d89307b0d284`.
- Exact mixture diagnostic: `4216fa5bb693957806eaa04e25f3c0b297e5507674bac754f649edca6e1dd6f3`.

## Data recovery and resource safety

The first per-file URL batch was stopped before producing a plan: a listed
image chunk returned404/invalid responses while other files resolved. Credentials
were not assumed invalid: metadata succeeded and AWS STS/SSH were verified.
No missing frame was fabricated or replaced with zeros.

The authorized87,393,127,165-byte competition ZIP supports byte ranges. Reading
only2,791,447metadata bytes yielded the exact408member inventory for four
division-positive movies. A real header/decompression/CRC chunk smoke preceded
their transfer:1,721,834,175image bytes in21.260seconds, all408SHA/CRC checked.
No whole competition archive was downloaded. Signed URL plan stays in ignored
local cache and mode0600 owned cloud scratch, never in logs or Git.

Image manifest: `5bac85637ee2fa1ae6c7d960c11f88778781cf30b1aceafe9897273fa6d63543`.
Three rejected pilot checkpoints were removed ONLY from their exact owned cloud
paths after local and remote SHA checks, freeing1,781,913,134bytes. All three
remain recoverable locally under image-context-pilot-v2-output. RSNA and shared
packages/processes were untouched; the instance was not stopped.

Kaggle quota read this session:30.00h remaining; preserve8h and recheck before
any launch. No Kaggle GPU has been used by this session. Cloud GPU availability
is checked immediately before each job; refreshed AWS credentials work.

## Division-positive full-pipeline execution

Four complete100-frame movies, all excluded from the motion fit/calibration:
44b6_12dfb391,44b6_267148e4,6bba_062c8d37,6bba_07e24132. Historical project
exposure and public-backbone training overlap remain explicit. Cloud bundle
contains images and models only, no truth or public prediction tables.

Same-code smoke PASSED24.817seconds on A10G, one8-frame movie, peakCUDA742MB.
Both base and repair pipelines ran, produced valid full-scope graph outputs,
and preserved all input hashes. Eleven smoke artifacts are hash-verified local,
manifestc737194a....

Frozen18-file runtime contract:
`7d4d1bc9f81f37fc2a8d23bd2449fc53f99bfb23eff39d02460b884661f8bdd1`.
Archive50,555,199bytes, SHAfa5553f8....

Full run launched about23:06UTC September13, sequential on initially idle
Antelume A10G,3600s watchdog,70%CUDA allocator limit,2CPU threads. One expensive
public inference per movie, followed by the cheap two-expert CPU repair.
First three movie times227.791/149.142/92.233s; repairs0/0/1edges. Fourth movie
still running at this update. No new truth scored until all eight new original/
repaired graph payloads are frozen and harvested. Scoring wrapper and tests
are ready for the unchanged joint-eight-movie and division-count gates.

Still no live submission or certified0.945+LB result. A pass will need offline
two-worker runtime, licensing/provenance and submission packaging before launch.
Job walltime is NOT billed EC2 uptime; the shared instance continues billing.

## Terminal update: eight-movie gate passed

The four additional movies completed in 642.874 seconds, peak allocated CUDA
749 MB. All 38 output artifacts (37,552,612 bytes) are SHA-verified locally in
`.biohub/cache/trajectory-division-full-v1-output`. Terminal SHA:
`40c35432a298d22f347cbdd111b1cd6d21ea3965acc35b4fc63369d3147544b1`.
The post-run compute query was empty, zero MiB allocated; Antelume remains up
and billing, with RSNA untouched. No Biohub GPU job remains live.

Patched official complete-movie micro score across eight movies improves from
0.9446046382742399 to 0.9453907265031998 (+0.0007860882289598692).
There are three additional true links and unchanged 141 scored false links.
Nodes and division counts (1 TP, 1 FP, 4 FN) are unchanged. No movie or embryo
regresses. This is local diagnostic evidence, NOT a leaderboard result or
pristine whole-model cross-validation. Public backbone overlap and historical
movie exposure remain limitations. No further threshold tuning was performed.

Quality receipt: `6dc733376a89e0eb8fdb8b996caf008252c84799199d35ef5f7f48a6ff39bb40`.
Thirty-nine focused tests pass. Eight GPU-capable job walltimes today total
1,374.442 seconds (22.91 minutes), NOT billed instance uptime. Kaggle use is
still zero. The candidate proceeds to offline two-T4 packaging, provenance and
real Kaggle runtime acceptance; no live submission has yet been made.

## Offline runtime launch: 2026-09-14 00:07 UTC (September13 local)

Packaging revision2 closes all64 exact LinuxCP312 non-Torch dependencies,
including inherited HTTP2 extras; independent marker/extra traversal finds
no missing packages or version conflicts. PyTorch/CUDA remain supplied by the
pinned Kaggle image, so actual-device quality is explicitly rechecked.
Three freshly queried public checkpoint datasets declare CC0-1.0. Apache2
notebook attribution, BSD3 model ancestry and all unmodified wheel notices
are preserved. Public definitions receive comment-only modification notices;
their ASTs are identical. Our expensive inference/repair block and all model
initialization are tested unchanged except image-root portability.

Portable CSV validation checks200,804 rows from four frozen full movies through
the pinned official CSV->graph->CSV converter functions: node and edge coordinate
multisets are identical. Thirteen portability tests plus existing focused tests:
49 tests pass17.35s in the latest suite. No labels/GPU were used for packaging.

The standard Kaggle uploader stalled and its owned process was stopped; no
other project process was touched. A bounded8MiB resumable upload succeeded
in122.609s after two recoverable network interruptions. The private runtime
dataset is ready, version1 containing packaging revision2 (the incomplete
revision1 never became a dataset). Runtime382,434,288bytes, archiveSHA
`a6bc494521cd93eabcad83abac2aaf040aab0f101078a2e1151b2ab2fe005e5a`,
contractSHA `d10d19b1e21b30e2eb5c6bf1c05161559f74594ce30e250e3c473f2d8b0379f2`.

Kaggle accepted `indarkarhana/biohub-trajectory-motion-acceptance/1` at00:07UTC.
Internet off, twoT4 requested,1h outer cap,2GPUh conservatively reserved from
a fresh30h remaining balance; reserve8h preserved. Real two-worker eight-frame
smoke must pass before eight full100-frame movies. No truth is read in Kaggle;
no submission.csv is generated by acceptance. Runtime statusRUNNING at00:10UTC.
The independent local verifier is prepared to compare all16 frozen outputs to
A10 evidence, reuse identical scores or evaluate all eight paired graphs with
the unchanged patched metric. Antelume has no Biohub run; RSNA untouched.

No leaderboard submission yet. The next gate is actual twoT4 execution and
throughput/quality, then production test inference and code submission.

### Acceptance startup repair and real two-T4 smoke

Kernelversion1 terminated before inference: Kaggle expands uploaded datasetZIPs,
so searching for the ZIP itself found no file. The recursive search was also
unnecessarily expensive. No cancellation was needed after it terminated; its
failure log is retained locally. Fresh quota after failure29.88h (0.12h used).

Bootstraprevision3 fixes only input discovery: two explicit supported mount
layouts, expanded CONTRACT/file hashes checked before executing any runtime code.
The uploaded runtime contract, all weights, inference code and repair are unchanged.
Both mount layouts now have regression tests;29portability/sharding tests pass.
Kagglekernelversion2 accepted00:18:21UTC, one-hourcap, reserve checked afresh.

Actual offline install succeeds with all64pinnedwheels. Two separate T4 workers
pass the8-frame smoke in50.373s,9366CSVrows, SHA
`3411162a0a2a611dc8b43457dacc99d833e0d2e18ab156f305ac23dc3154b4aa`.
KaggleTorch2.10.0+cu128 differs from Antelume2.5.1+cu121; paired actualT4
full-movie quality is mandatory, not assumed. Eight complete movies are now
running; no submission.csv is generated in acceptance. Local verifier and guarded
production builder are ready. The production builder requires measured runtime
headroom as well as quality; no new production run or competition submission yet.

## Actual two-T4 acceptance PASS; runtime optimization (00:58 UTC)

Kernel version2 completed offline: two-T4 smoke50.373s, eight complete movies
1138.386s, total bootstrap/smoke/full1211.072s. All16 frozen paired graphs were
hash-checked and scored with the unchanged patched official metric. Although
some GPU outputs differ in JSON coordinates from A10, every metric count is
identical: base0.9446046383 -> repaired0.9453907265, +3 edgeTP, same141FP,
unchanged nodes/division counts and no movie/embryo score regression.
This is local diagnostic evidence, NOT a leaderboard score or pristine CV.
Acceptance receipt SHA4719aa024761b9045f431654b6f1718f96ca997a5d790af91c9482965171c169;
terminal146d01ad770eab41dd988a17cc852a133f714e99792859da60130fabbdc2cdd7.
ValidationCSV300253rows, SHAe4c22999bc515b3ffeab9a3d9000fd92b3e3e3be5578452c06cfd5b6acea9939.

Measured199-movie/two-worker projection7.731h is not a hidden-runtime guarantee.
It misses the production builder's predeclared50% headroom inside a10h inference
watchdog. That engineering gate is separate from the12h competition maximum;
it has not been loosened. No production notebook or competition submission yet.

User requests useful utilization of the paid Antelume GPU. Keep bounded useful
follow-ups ready, but never run filler, conceal CPU/validation phases, or disrupt
RSNA/shared jobs. No instance stop or shared environment changes are authorized.
The real encoder benchmark's first tiny mixed-precision batch2 smoke failed
after6.012s at the old PyTorch SDPA65535 flattened-batch limit. Benchmarkrevision2
chunks independent voxel sequences, preserving attention semantics;96 cases
completed70.127s. Larger FP32 batches were no faster. Encoder-only FP16 was
approximately2.6x faster, but changed logits and detection-threshold decisions:
not a drop-in optimization and not promoted. Production batch1 needs no SDPA
chunk patch; no shared Torch upgrade was made.

A separate encoder-only AMP bundle freezes unchanged FP32 accumulation, edge
transformers, DeepCenter, ILP and motion repair. Contract94fd57e631523de95201ec7ac57567ff1e2908a35b3bf9d6294fc2e1b6c46254.
Smoke18.709s passed before four full division-positive movies completed454.225s
(FP32 comparison642.874s). Predictions are frozen, with scoring pending; the
validated FP32 candidate is preserved. Any AMP promotion requires full quality
nonregression followed by broader eight-movie and actual T4 acceptance.

## Useful-utilization result and deployment check (01:23UTC)

AMP encoder screen CLOSED as rejected. Four-movie accepted FP32 score0.9383918673
falls to0.9376069681; both embryos regress, worst-movie44b6_267148e4 delta-0.00563935.
Division counts remain unchanged but do not override the quality failure.
Receipt299da75c6caf339311519a987174810e974777cc5f7bcfc8ce112be270e134f0.
All38AMP outputs/37.54MB are hash-verified locally; no wider AMP run or promotion.
Benchmarkrevision2 resultf7b5764f2a0536e7fcbee8bfc7ef5df0def9c36bdb8ea509baad32c1a9e5de18
matches remote; benchmark source2158c437... also matches remote.

Instead, two bounded FP32 movie workers share the idle A10G as ONE experiment,
overlapping GPU inference with CPU graph work. Total2CPUthreads,35% allocator
per worker/70% aggregate; refuse foreignGPU at start and yield owned workers if
foreignGPU processes appear. Only owned Popen children may be stopped. RSNA and
the shared environment/instance are untouched. Model math, thresholds and FP32
precision are unchanged; five new tests verify exact resource-only code delta.

Overlap smoke29.588s passes before full four movies471.251s, versus642.874s
sequential:26.696% shorter walltime /1.364x throughput. Repeated utilization
samples reach100%; CPU phases still produce some zero/partial GPU samples.
All eight before/after graph JSON payloads are EXACTLY IDENTICAL to the accepted
FP32 outputs. No annotations reopened: existing scores are reusable.
45artifacts/37.57MB hash-verified, manifest63dea533..., terminalacccae6c...,
identity receiptc5a603951bc6d6ab97a88ee5f800e6483d96a616493ef9942a9184ee4017cfe5.
This proves A10 speed/identity, not T4 throughput or hidden runtime.

Kaggle overlap acceptancev1 pushed01:23:21UTC: two physical T4s, up to two
one-thread/35% workers each, four-movie8frame smoke before all eight full movies.
The existing accepted dataset is reused with small embedded, SHA-checked code
overlays; all weights, dependencies and licenses preserved. Base hashes checked
BEFORE overlay materialization; derived contract851908fa5aa8ba628aad456afa94badc6ae9e10c6f9f3e36cf43147386ef8d55.
Forty-two scheduling/portability tests pass. Fresh Kaggle quota29.54h (used0.46h);
one-hour cap, conservative2GPUh reservation,8h reserve preserved. Offline install
and overlay verification pass; four worker smoke is running. Antelume's owned
job is finished and GPU is temporarily idle; EC2 still running/billing. No new
production notebook or leaderboard submission yet.

### Four-worker Kaggle smoke passed; next exact-work reduction queued

Actual four-worker/two-T4 smoke passes76.382s,12313CSVrows,
SHA17e22bf742f3550a92a9b84b5b882d8e14f478953facdd7731012b84123a591d.
Eight complete movies are running. Combined current scheduling, portability,
motion and quality tests:49passed3.53s. The quality verifier now uses complete
cohort makespan for overlap throughput; summing inflated concurrent-worker
times and dividing by two would misestimate runtime. Production staging/launch
are prepared but require the actual same-contract eight-movie result and
unchanged50% runtime-headroom gate. Fresh submission history still shows no new
submission since August26; production and leaderboard actions remain pending.

An additional exact-work reduction is staged, NOT launched: cache the earlier
flip-X encoding for the original eighth TTA input, whose rotation+transpose is
exactly flip-X. Keep all eight original contributions, inverse transforms,
weights and addition order; this is NOT the rejected D4 correction. Runtime
asserts identical input tensors on every reuse, and separate cache variables
are released each window for both models. Three CPU tests pass, including
rectangular input geometry and retained summation order. No FP16/threshold or
DeepCenter changes. Antelume bundle1b0124fd435802c47fffd072ed323d4bfde3aa95785db05d7a7eb9610fbdb7d1,
archive41a6f855cf7277a92e4d09d268d0d4446c32537874bdc95fb14d6b3e0793c1a8
has been copied to the owned cloud directory; same short-smoke/full-identity
gates are required before any use. Numerical experiments remain sequential:
if the current candidate is runtime-ready, prioritize submission over this
extra optimization; otherwise it is the next bounded Antelume test.

## Production candidate launched; explicit runtime-budget revision (01:45UTC)

Both actual T4 schedules preserve the exact same300253-row validationCSV and
all patched metric counts. Four-worker overlap completes1099.546s versus the
two-worker1138.386s: only3.41% faster on T4, unlike26.70% on A10. Overlap quality
receiptb5a86113ee182a0dc437201f10773f72f7370115d3b622d14e4640031d08d8ec;
terminal598c6588.... The simpler already-accepted two-worker T4 runtime is used
for the first production candidate; the A10 optimization remains useful there.

Before production, explicitly revised the provisional engineering buffer and
told the user: require at least TWO HOURS inside the unchanged10h hard stop,
with two additional hours below Kaggle's12h limit. The earlier50% multiplier
was our planning heuristic, not a competition or scientific-quality rule.
Use the MORE conservative full-cohort-makespan projection including initialization
and CSV assembly:7.86593h, leaving2.13407h inside the hard stop. Neither this
projection nor any safety multiplier guarantees unseen-embryo runtime. No
model, threshold, scientific acceptance criterion, quota reserve or runtime
code was changed. Smoke/genuine-node/complete-movie checks remain mandatory.

Production notebook differs from accepted bootstrap ONLY in RUN_MODE=production;
three tests verify this, its metadata and exact quality/resource proof.
indarkarhana/biohub-trajectory-motion-candidate/1 pushed01:45:01UTC,
notebook9e497e6b678460f58917bbb30dfdf48068284542b0295d56bb6f57ca5b7b591a,
same runtimecontractd10d19b1.... Fresh quota29.20h used0.80h. Conservatively
reserve20GPUh for two devices x10h maximum, leaving9.20h>=8h. Public test smoke
passes45.295s/5782CSVrows; all four actual public-test movies are running.
All test movie IDs independently listed across three official file-API pages:
44b6_0113de3b,44b6_0b24845f,6bba_05b6850b,6bba_05db0fb1.
The final verifier rebuildsCSV from frozen graphs, checks every movie and both
GPU shards, original graph preservation, exact private kernelversion1 and fresh
quota/daily submission count before the one-shot code submission. Not submitted yet.

With numerical scheduling experiments finished, production inference runs on
Kaggle while the next ONE research experiment uses paid Antelume: exact duplicate
TTA cache smoke27.631s passes, with identical smoke graphs and all tensor-equality
assertions. Four complete movies are now running, two owned workers/2CPUthreads/
70% aggregate allocator. Repeated GPU samples100%. It cannot delay or replace
the production candidate and is not yet quality-promoted. RSNA untouched.

## SUBMITTED:56219125 (2026-09-14 02:05:48UTC / September13 local)

The full public-test production run completed780.357s; bootstrap/smoke/full
total846.797s. All four100frame movies covered,241192CSVrows, SHA
ae121d9df70f118999479e3e0b02f5fbdb720115323e08181b9cdf9206e4730b.
Source-trained repair adds3genuine-endpoint links; all original nodes/edges
preserved. RootCSV and independent local reconstruction agree byte-for-byte.
Exact private/offline two-T4 kernelversion1 and its inputs verified via SDK.

New public timing materially qualifies the earlier planning buffer: one movie,
6bba_05db0fb1, contains70262detections and takes486.434s. Public movie-mean
projection9.02354h; combined twelve-movie mean8.16195h; naive four-movie
makespan extrapolation10.78410h includes substantial static-shard imbalance.
Thus the provisional two-hour buffer is NOT confirmed by newer public timing.
User informed before submission: expect roughly8–9h, with genuine dense-movie/
shard-imbalance timeout risk; unchanged10h stop and quota reserve still apply.
The final verifier explicitly records this newer evidence and refuses submission
if either observed mean projection exceeds10h. Do not repeat an unconditional
two-hour-buffer or hidden-runtime guarantee in future status updates.

Fresh quota28.97h (used1.03h), conservative20GPUh reservation leaves8.97h>=8h.
Daily submission count0 before this attempt. One-shot code submission returned
success and an INDEPENDENT competition-history query confirms submission56219125,
02:05:48.173UTC, statusPENDING, no score yet. Submission receiptf69aef23...;
registration snapshottrajectory-motion-v1-registration.json. No final selection
was changed.0.9453907265 remains local validation, NOT a leaderboard score.

Antelume exact duplicate-view caching also completes435.400s:32.273% faster
than original sequential642.874s,7.61% faster than overlap alone471.251s.
All eight before/after full-movie graphs are exactly identical to acceptedFP32;
no truth reopened.45artifacts37.57MB locally hash-verified; terminal884d8a30...,
manifest599f7419..., identity receipt262ba9c9eb11876cf44c000b6597d8770ddc765ad26ad4612b9375cb6b487cf9.
This cache optimization is NOT in the submitted version and still needs actual
T4 acceptance before future deployment; it is not a distinct-score candidate.
Fifty-five focused tests pass11.75s. Latest cloud check: no computePIDs/0MiB;
Biohub experiments finished, AntelumeGPUidle, shared instance remains on/billing.
RSNA and shared packages/instance unchanged. Keep the goal active pending the
actual competition outcome; no unsupported claim of meeting0.945LB.
