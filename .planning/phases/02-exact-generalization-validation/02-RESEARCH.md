# Phase 2 Research: Exact Generalization Validation

**Researched:** 2026-08-24 UTC  
**Phase:** 2 — Exact Generalization Validation  
**Scope:** CPU-only scorer, data manifests, complete-movie evaluation, diagnostics, uncertainty, and promotion policy  
**GPU used:** 0 hours

## Executive Recommendation

Build Phase 2 around the organizer's `tracking_cellmot` package at Git commit
`075fc5f5a52d11077f9dc2b074644618f26939e2`, the merge that includes the weakly
connected-component exploit patch. Do not rewrite the official edge or division
metric. Call `tracking_cellmot.metrics.evaluate` on each complete movie, derive the
per-movie row with `per_sample_metrics`, and aggregate with `summarise`. Wrap those
functions with stricter local controls for scorer provenance, complete coverage,
metadata availability, graph integrity, submission-space round trips, diagnostic
decomposition, paired movie bootstrap comparison, and promotion.

The upstream convenience script is not safe enough to be the decision boundary:
it evaluates only the intersection of prediction and truth filenames, silently
skips unreadable graphs, falls back to a default scale when image metadata is
missing, and can omit adjusted-score rows when `estimated_number_of_nodes` is
absent. The local adapter must turn every one of those situations into a hard
failure before scoring.

Freeze two reciprocal source-group folds:

- train/calibrate on every `44b6_*` movie and evaluate every complete `6bba_*`
  movie;
- train/calibrate on every `6bba_*` movie and evaluate every complete `44b6_*`
  movie.

The first real promotion report must contain the union of the two held-out
prediction sets, scored as one complete split and also decomposed by fold, embryo,
and movie. Calibration, threshold selection, and early stopping belong only to the
training side of each fold.

## Evidence Labels

This document uses three labels deliberately:

- **Authoritative fact** — directly implemented or stated by the organizer's
  pinned source, official Kaggle pages, organizer announcement, TracksData, or
  GEFF reference source.
- **Recommended design** — a project decision proposed for reproducibility or
  scientific rigor; it is not a claim about Kaggle's private scoring container.
- **Unresolved blocker** — evidence needed for a real-data report is not locally
  available or is not published by the organizer.

## Authoritative Upstream Pin

### Repository and patch identity

**Authoritative facts:**

| Item | Exact identity | Evidence |
|---|---|---|
| Organizer repository | `royerlab/kaggle-cell-tracking-competition` | Public BSD-3-Clause repository |
| Current `main`/`HEAD` at research time | `075fc5f5a52d11077f9dc2b074644618f26939e2` | `git ls-remote` on 2026-08-24 UTC |
| Exploit-patch commit | `aa65e90aeb8a774ebb1b549e547787b87ac8a01c` | Commit title: “updating metric to patch weakly connected component exploit” |
| Patched merge commit | `075fc5f5a52d11077f9dc2b074644618f26939e2` | Merge of the metric-fix branch plus frozen sandbox tests |
| Immediate pre-patch commit | `7396b7e98e61844e799152ddda7e5493084cc8f3` | Useful only for proving the adversarial fixture changes behavior |
| Package name/import root | distribution `tracking-cellmot`, import `tracking_cellmot` | Pinned `pyproject.toml` and source tree |
| Package version metadata | `0.1.0` | Pinned `pyproject.toml` |
| Python constraint | `>=3.11,<3.14` | Pinned `pyproject.toml` |
| License | BSD-3-Clause | Pinned repository `LICENSE` |

The current AI-SPEC's conceptual framework choice is correct, but the exact
language should be `tracking_cellmot` from the pinned organizer repository—not a
separate `royerlab/tracking_cellmot` repository. The authoritative scoring call
graph is:

```python
from tracking_cellmot.metrics import (
    evaluate,
    node_recall,
    per_sample_metrics,
    summarise,
)
```

`evaluate_datasets` is **not** the competition final-score path because it reports
raw micro edge Jaccard and does not apply the per-sample node-count adjustment.

### Critical source hashes

The scorer lock should include the commit and SHA-256 of every source file that
can change final evaluation or submission-space conversion. These hashes were
computed from commit `075fc5f5a52d11077f9dc2b074644618f26939e2`:

| File | SHA-256 |
|---|---|
| `metrics.md` | `22e025fac9e66bd97331b2ffdb5851ec867162dcfcda372fe3420260430cc3e0` |
| `src/tracking_cellmot/metrics.py` | `ab11310db0ada78ebb408fcd913bd001aed0dd5f8c1179e0a45dbb7152d6ebde` |
| `src/tracking_cellmot/division_metrics.py` | `ef7472347a06842982bda795f896fd60fd7cb2e429fb87a72c6b9ede50c49d29` |
| `src/tracking_cellmot/io.py` | `fef87f49a8cea1a7f943493e5eb61ebbba5049b5536749f82732b0e6ba57f2ed` |
| `scripts/evaluate.py` | `0c259fd75ab7cc2174c266ac057ac8b901a19852c64e5ba2889867144ed3fc83` |
| `scripts/geffs_to_csv.py` | `c3a9a3f2fdf8cac622221dce5ee1427c0e46157974ca37358e46ec318e7a421a` |
| `scripts/csv_to_geffs.py` | `6ff289b978549c0745588a9f0fb455ae9d6507a3f1e570f3c3b77da36b853c20` |
| `pyproject.toml` | `5ccfd14d33bccc855c5177feb174178ded95947c05c62df2a6df5e38e972b0f3` |
| `LICENSE` | `4cd604324b1a9f0c420786a92501430a7ff9189f46bdcdfc485aa43bfc16973b` |

**Recommended design:** create `config/official-scorer.lock.json` with repository
URL, commit, patch commit, retrieval timestamp, license identifier/hash, critical
file hashes, adapter version, constants, full environment-lock hash, and the
frozen fixture-result hash. At runtime, verify the checked-out files and imported
module paths against this lock before opening candidate graphs.

### Dependency constraints and the floating dependency problem

**Authoritative facts:** the pinned organizer `pyproject.toml` declares
`torch>=2.9.1`, `tracksdata @ git+https://github.com/royerlab/tracksdata@main`,
`zarr>=3.0.10`, `scipy`, and `tqdm`. The evaluation code additionally imports
Polars and GEFF through TracksData. TracksData uses `DistanceMatching`, groups
nodes by `t`, performs a per-timepoint optimal bipartite assignment, and computes
spatial weights as `1 / (1 + physical_distance)` for pairs at or below the
threshold.

The organizer did not publish a lockfile. Therefore the repository commit alone
does not define a byte-for-byte dependency environment. `tracksdata@main` floats.
The TracksData commit that was `main` immediately before the scorer patch was
merged was `39dccf3a243e44274759468cb31b2ad9e7fc1d09` (2026-07-13); its matching
implementation is unchanged in current main as of this research. TracksData at
that commit is BSD-3-Clause and requires GEFF `>=1.1.3.1.1` among its dependencies.

**Recommended design:** use `39dccf3a243e44274759468cb31b2ad9e7fc1d09`
as the initial patch-era TracksData pin, then freeze every resolved package and
wheel hash in an evaluation lock after the complete upstream and local regression
suite passes. Prefer a pinned vendor checkout or direct Git dependency plus an
environment lock; do not copy and maintain a divergent metric implementation.
Install the organizer checkout without allowing its floating TracksData
requirement to win resolution. The evaluation environment can remain CPU-only,
but it must prove compatibility with the official source rather than removing
imports speculatively.

**Unresolved blocker:** Kaggle does not expose the private scorer container's full
dependency lock. Exact source parity is achievable; byte-identical private
container parity is not currently provable. Record this explicitly instead of
claiming it. If an organizer publishes a container or lock later, update the
scorer lock only by creating a new version and rerunning every fixture.

## Authoritative Metric Semantics

### Node matching

**Authoritative facts:**

- Prediction nodes are matched to GT nodes separately at each integer timepoint.
- Matching is one-to-one optimal bipartite assignment.
- Distance is Euclidean after multiplying `(z, y, x)` voxel coordinates by the
  physical scale.
- The competition scale is `(1.625, 0.40625, 0.40625)` micrometres per voxel.
- Pairs with distance `<= 7.0` micrometres are eligible.
- The organizer's `evaluate` mutates the prediction graph by resetting and
  writing match attributes. Every independent evaluation/diagnostic must operate
  on a fresh graph or copy.

Negative-time nodes are not a supported way to obtain a legitimate score. The
TracksData matcher iterates nonnegative timepoints up to the maximum time. The
local graph validator should reject negative/sentinel time or coordinates before
the organizer code sees them.

### Ordinary edge counts

**Authoritative facts:** after matching:

- A predicted edge is TP when both endpoints map to a GT edge.
- A GT edge without a mapped predicted edge is FN.
- A non-TP predicted edge is FP only when its matched source has a GT outgoing
  edge or its matched target has a GT incoming edge; other edges in unlabeled
  regions are ignored because labels are sparse.
- Only predicted edges with `t_target - t_source == 1` are retained by the
  metric.
- Duplicate source-target edges are deduplicated.
- When multiple prediction edges collapse onto the same matched GT edge, the
  metric retains one.
- Predicted out-degree above two is capped to two lowest edge IDs for ordinary
  edge evaluation.

The last three behaviors are scorer defenses, not graph-construction features.
The project validator should reject duplicate edges, merged branches, and
out-degree above two rather than depend on score-order side effects.

Per movie:

```text
edge_jaccard = edge_tp / (edge_tp + edge_fp + edge_fn)
total_node_ratio = (num_pred_nodes - estimated_number_of_nodes)
                   / estimated_number_of_nodes
adjusted_edge_jaccard = max(
    0,
    edge_jaccard * (1 - 0.1 * total_node_ratio),
)
```

Because under-prediction makes `total_node_ratio` negative, a valid adjusted
score can exceed `1.0`; do not enforce an incorrect `[0, 1]` bound. Non-finite
values remain invalid for promotion.

### Patched divisions

**Authoritative facts:** a GT division has exactly two outgoing edges; any
prediction node with at least two outgoing edges is considered a predicted fork.
The patched metric evaluates a local
grandparent → dividing-parent → children → grandchildren window:

- the fork must be the matched GT parent-side node or its immediate successor;
- two GT daughter lineages must map to two distinct direct prediction branches;
- the support must be directed and local, not merely in the same weakly
  connected component;
- direct-child evidence takes precedence, with unambiguous grandchildren used
  only as fallback;
- shared/merged child or grandchild branches are malformed;
- branch evidence spanning distinct reliable GT components makes a fork FP;
- maximum-cardinality bipartite matching prevents a predicted fork from
  recovering multiple GT divisions;
- an unpaired GT division is FN;
- evaluable local rejects, leftover candidates, cross-component forks, and
  malformed forks are deduplicated by predicted fork ID and counted as FP;
- an unmatched structurally valid fork with no local annotated evidence can be
  ignored under sparse-label semantics.

The organizer's exact exploit topology is frozen in
`tests/test_division_sandbox_examples.py` as `hack2`. Under the patched scorer at
its fixture threshold it must produce edge TP/FP/FN `5/4/5` and division TP/FP/FN
`0/4/2`, with predicted forks `105`, `106`, `122`, and `124` all classified FP.

### Complete-split aggregation

**Authoritative facts:** the competition final score uses `summarise`, not a mean
of movie scores:

- raw edge Jaccard is computed from edge TP/FP/FN summed over valid movies;
- adjusted edge Jaccard is a weighted mean of per-movie adjusted values using
  `edge_tp + edge_fp + edge_fn` as the movie weight;
- division Jaccard is computed from division TP/FP/FN summed over movies;
- final score is `adjusted_edge_jaccard + 0.1 * division_jaccard`;
- if the entire evaluated set contains no division events, the division term is
  dropped;
- upstream `summarise` reports node recall as an unweighted mean of per-movie
  recall. Node recall is diagnostic and not part of the final score.

The local report should retain the organizer macro node recall and additionally
report a clearly named `node_recall_micro = sum(matched_unique_gt_nodes) /
sum(gt_nodes)`. Never label the latter as an organizer score component.

## Data Availability and Frozen Manifest Construction

### What is available now

**Observed locally:** the workspace has no `.geff`, `.zarr`, model weights,
prediction CSV, checkpoint, or prior exact-report artifacts. `.biohub/cache`
contains audited public notebook sources only. The experiment ledger contains
historical summaries, not the underlying graphs.

**Observed remotely without downloading payloads:** Kaggle file metadata is
accessible. A complete metadata-only pass reported 24,886 individual files and
87,609,892,618 bytes. A second pass verified four visible example-test Zarr
datasets and 119 paired train GEFF/Zarr dataset names before Kaggle returned HTTP
429. Project context states 199 training movies, but that count was not fully
reconfirmed in this pass. Manifest generation must derive and assert the actual
count from the mounted data, never hard-code 199 as a substitute for coverage.

**Official Kaggle facts:** each train sample is a paired Zarr v3 image and GEFF
graph; image arrays are `(T,Z,Y,X)`, typically `(100,64,256,256)` `uint16`, with
one-timepoint chunks; training labels are sparse; `estimated_number_of_nodes` is
in GEFF metadata; names are `{embryo_id}_{field_of_view}`; the embryo prefix is
the source identity; train and hidden test embryos are disjoint.

### Manifest algorithm

**Recommended design:** `biohub manifest build` should scan a supplied training
root and fail closed unless every discovered sample satisfies all of the
following:

1. Exactly one `<sample>.zarr` directory and one `<sample>.geff` directory exist.
2. The sample stem matches `^(44b6|6bba)_.+$`; the first prefix is the source
   group. Unknown or conflicting identity is an error.
3. Zarr metadata declares a 4D `(T,Z,Y,X)` image, finite positive scale, expected
   axis order, and usable spatial bounds. For official data, scale must equal
   `(1.625,0.40625,0.40625)` unless a future official source change is explicitly
   locked.
4. GEFF metadata is readable, directed, contains finite `estimated_number_of_nodes
   > 0`, and exposes `t,z,y,x` plus node/edge arrays.
5. Every GT node and edge passes schema, time, finite-coordinate, endpoint, and
   topology checks.
6. Dataset names, resolved paths, source groups, and artifact identities are
   unique.

Each sorted movie record should contain:

```json
{
  "sample_id": "6bba_...",
  "embryo_id": "6bba",
  "field_of_view_id": "...",
  "image_relpath": "train/6bba_....zarr",
  "truth_relpath": "train/6bba_....geff",
  "shape_tzyx": [100, 64, 256, 256],
  "dtype": "uint16",
  "chunks_tzyx": [1, 64, 256, 256],
  "scale_zyx_um": [1.625, 0.40625, 0.40625],
  "estimated_number_of_nodes": 12345,
  "gt_node_count": 0,
  "gt_edge_count": 0,
  "gt_division_count": 0,
  "image_metadata_sha256": "...",
  "geff_tree_sha256": "..."
}
```

Hash all GEFF content and the Zarr metadata actually used for scoring. Avoid
rereading roughly 88 GB of image chunks solely to construct every evaluation
report. For later training provenance, also store a canonical Kaggle source
inventory fingerprint over sorted remote path/size/creation metadata or a
separately computed image-tree Merkle hash. Label an inventory fingerprint
accurately; it is not a byte hash of image chunks.

The manifest contains two folds with sorted membership arrays:

```text
fold-44b6-to-6bba: train/calibrate embryo 44b6, evaluate embryo 6bba
fold-6bba-to-44b6: train/calibrate embryo 6bba, evaluate embryo 44b6
```

The overlap audit must prove:

- no sample ID occurs in more than one membership within a fold;
- train and evaluation embryo/source sets are disjoint;
- resolved paths and GEFF hashes do not cross sides unexpectedly;
- every discovered official train movie occurs exactly once on the evaluation
  side across the reciprocal folds;
- no movie is split into clips or edges;
- calibration and threshold-selection IDs are subsets only of the fold's
  training side.

Compute `manifest_sha256` from canonical semantic content with the self-hash field
omitted. A creation timestamp belongs in a separate evidence envelope or must be
excluded from semantic identity. Identical data and policy should regenerate the
same manifest bytes and hash.

## Validation Architecture

```text
Pinned organizer checkout + dependency lock
                    |
                    v
          scorer provenance verifier
                    |
Official train root -> manifest builder -> immutable reciprocal manifest
                    |                         |
                    |                         v
Baseline GEFFs -----+----> exact coverage + graph integrity preflight
Candidate GEFFs ----+                         |
                                              v
                       submission-space canonicalization / CSV round trip
                                              |
                                              v
                        fresh graph copies -> organizer evaluate per movie
                                              |
                           +------------------+------------------+
                           |                  |                  |
                           v                  v                  v
                    sufficient counts   diagnostics       paired deltas
                           |                  |                  |
                           +------------------+------------------+
                                              v
                              official summarise + paired bootstrap
                                              |
                                              v
                         canonical exact report + policy decision
                                              |
                                              v
                               append hashes to experiment ledger
```

### Architectural boundaries

1. **`scorer_lock.py`** verifies source/dependency provenance and imports the
   pinned organizer API. It does not implement an alternative scorer.
2. **`manifests.py`** discovers official data, derives source identity, freezes
   folds, hashes evidence, and proves overlap/coverage invariants.
3. **`graphs.py`** loads GEFF and validates schemas/topology without changing
   metric semantics.
4. **`submission_io.py`** owns canonical GEFF→CSV→GEFF conversion and semantic
   round-trip checks. It may call pinned organizer helpers but adds strict schema
   and complete-dataset validation.
5. **`evaluation.py`** calls organizer functions sequentially per complete movie,
   preserves raw sufficient statistics, and computes exact aggregates.
6. **`diagnostics.py`** computes explanatory quantities from copied match state.
   Diagnostics never replace official counts.
7. **`comparison.py`** performs identical-manifest baseline/candidate pairing,
   deltas, worst-movie analysis, and deterministic bootstrap.
8. **`promotion.py`** applies a versioned policy to a complete report and appends
   an evidence hash to the existing ledger.

Stream graphs movie-by-movie. Do not load image volumes during scoring; only Zarr
metadata is required for scale/shape. Evaluation order is canonical sample ID
order. Optional multiprocessing is deferred until sequential and parallel outputs
are proven byte-identical.

### Fail-closed execution order

1. Verify scorer lock and imported module file hashes.
2. Verify manifest self-hash and source inventory.
3. Require baseline and candidate prediction sets to equal the expected
   evaluation set exactly—no missing or extra datasets.
4. Validate every graph and every submission-space projection.
5. Prove direct canonical GEFF and CSV-round-tripped GEFF parity.
6. Score baseline and candidate with fresh graph instances.
7. Require all metric rows finite where the official score requires values.
8. Aggregate, compare, bootstrap, and render diagnostics.
9. Apply policy only if all earlier gates are complete.

## Graph and Coverage Integrity

**Recommended design:** reject a candidate before scoring if any movie has:

- missing/extra/duplicate dataset identity;
- unreadable or partial GEFF;
- missing, duplicate, or non-integer node IDs;
- non-integer time or submission-space coordinates;
- non-finite values;
- time or coordinates outside image bounds;
- duplicate edges, dangling endpoints, self-loops, backward edges, or edges with
  `t_target != t_source + 1`;
- in-degree above one, out-degree above two, shared/merged daughter branches, or
  any cycle;
- known sentinel/out-of-volume hub or fake-fork exploit signature;
- missing/invalid scale or `estimated_number_of_nodes`;
- declared movie/hash disagreement with the manifest.

Some of these cases are ignored, capped, or converted to no-match by the official
metric. Rejecting them is an intentional project integrity policy, not a claim
that the organizer always throws an exception.

## Adversarial Regression Fixtures

Start with upstream graph builders/expectations where licensing permits, add a
thin independent assertion layer, and freeze expected raw counts—not only final
floats.

| Fixture | Required assertion |
|---|---|
| Perfect linear graph | all GT edges TP; no FP/FN; node recall 1 |
| Empty/no-edge prediction | no accidental NaN promotion; expected FN counts |
| 7 µm boundary | exactly 7 µm matches; a declared epsilon over does not |
| Time-aware match | spatially identical node at another `t` does not match |
| Anisotropic scale | Z/Y/X scaling changes eligibility exactly as official metadata declares |
| Sparse-label endpoint cases | extra edge at unlabeled track boundary ignored; interior conflict penalized |
| Exact/equal/over/under node totals | adjusted formula and possible score above 1 frozen |
| Missing node estimate | local adapter rejects even though upstream can yield NaN/skip |
| Aggregation imbalance | official weighted/micro result differs from per-movie mean and wins |
| Perfect/on-time division | division `1/0/0` |
| Early and late local division | valid ±1-frame local topology recovers GT |
| Missed/spurious division | expected FN/FP |
| Exact patched `hack2` exploit | edge `5/4/5`, division `0/4/2`, all four fork IDs FP |
| Far-out sentinel hub/fake chains | integrity rejection before scoring |
| Fork reuse | one predicted fork cannot recover multiple GT divisions |
| Cross-component daughters | fork FP, deduplicated once |
| Merged/shared child or grandchild | malformed fork FP |
| Duplicate edges/merged matches | cannot inflate TP or score above intended counts |
| Third child/out-degree >2 | local integrity rejection; separate parity test documents upstream cap |
| Self, reverse, and frame-skip edges | local rejection; separate parity test documents upstream filtering |
| NaN/Inf/negative time | local integrity rejection |
| Missing/extra movie | coverage rejection before organizer call |
| No divisions in entire split | official division term is dropped, not replaced silently with zero |
| CSV/GEFF integer graph round trip | semantic graph and exact metric counts identical |
| Subvoxel prediction | cannot be promoted until projected to integer submission space |

Run a second regression against the pre-patch commit only to demonstrate that the
exploit fixture distinguishes versions. The pre-patch scorer must never be
available through the normal evaluation CLI.

## CSV/GEFF Round Trip

**Authoritative facts:** Kaggle submission coordinates are integer voxels. The
organizer's `geffs_to_csv.py` rounds `z,y,x` to integers, writes node rows followed
by edge rows, and adds a consecutive throwaway `id`. `csv_to_geffs.py` rebuilds
graphs with fresh internal node IDs and remaps edges through submitted node IDs.
Therefore byte identity and original node-ID identity are not expected; semantic
graph identity is.

The official converter is lossy for non-integer prediction coordinates. The
authoritative local promotion score should be calculated in **submission space**:

1. validate native candidate GEFF;
2. project coordinates with the exact pinned converter semantics;
3. emit canonical CSV with exact schema and all expected movies;
4. rebuild GEFF;
5. validate the rebuilt graph;
6. compare node count, edge count, time/coordinate multiset, and directed topology
   under the node-ID mapping;
7. score canonical direct GEFF and rebuilt GEFF and require identical TP/FP/FN,
   per-movie rows, and aggregates.

If native subvoxel GEFF differs after projection, retain the native score only as
a non-authoritative diagnostic. Promotion uses the round-tripped integer graph.
Canonical CSV validation also requires exact header/order, consecutive IDs,
recognized row types, `-1` in irrelevant fields, unique node IDs per dataset,
valid edge references, deterministic row ordering, and exact manifest coverage.

## Exact Report and Diagnostic Decomposition

### Deterministic report core

The canonical report should include:

```text
schema_version
scorer_lock_sha256
environment_lock_sha256
manifest_sha256
promotion_policy_sha256
baseline {run_id, graph_inventory_sha256, model/config/code hashes}
candidate {run_id, graph_inventory_sha256, model/config/code hashes}
coverage {expected, present, missing, extra, invalid}
pooled {baseline, candidate, delta}
by_embryo
by_fold
by_movie
diagnostics
bootstrap
integrity_checks
decision_inputs
```

Each baseline/candidate metric group retains:

- edge TP/FP/FN and raw micro edge Jaccard;
- per-movie adjusted edge Jaccard inputs, weights, and official aggregate;
- division TP/FP/FN and micro division Jaccard;
- predicted node count, estimated node count, and node-count ratio;
- per-movie node recall, organizer macro node recall, and clearly named micro node
  recall;
- final score.

Store integer sufficient statistics and canonical decimal strings. Keep wall-clock
runtime, peak memory, and creation timestamp in an evidence envelope that points
to the deterministic report-core hash; otherwise identical inputs cannot produce
byte-identical reports.

The existing Phase 1 `validate_complete_metrics` contract is only a compatibility
minimum: it requires pooled score components, division counts, embryo/fold maps,
and worst-movie delta, but not edge TP/FP/FN, scorer/manifest/policy hashes,
coverage, round-trip evidence, or diagnostics. Phase 2 should add a strict exact-
report validator and then project its required summary into the older ledger
fields; it should not weaken the exact report to fit the current minimum schema.

### Detection/link separation

After organizer node matching on a fresh graph copy, compute:

- **endpoint availability:** number/fraction of GT edges for which both GT
  endpoints have matched prediction nodes, independent of whether the prediction
  linked them;
- **oracle-link ceiling:** construct only the GT-consistent edges between available
  matched prediction endpoints, with zero additional valid FP edges, then derive
  raw and adjusted edge ceilings with the candidate's unchanged node-count
  penalty;
- **conditional association recall:** `actual_edge_tp /
  available_gt_edges`, with a defined NaN/no-event case;
- **conditional valid-edge precision/Jaccard:** retain actual valid prediction FP
  so an aggressive linker cannot hide behind high conditional recall.

These are diagnostics derived from the authoritative match state. They must not be
fed back as substitute score components.

### Error strata

Recommended diagnostic strata are:

- physical GT displacement bins with fixed versioned micrometre boundaries;
- movie-density bins based on `estimated_number_of_nodes / (T * physical spatial
  volume)`, with boundaries learned only from the training side and frozen in the
  manifest/policy;
- division timing support at `-1`, `0`, and `+1` frames and local failure category;
- node-count ratio, endpoint availability, conditional linking, and oracle gap per
  movie;
- the lowest candidate movie, largest negative candidate-baseline movie delta,
  and top regressions by displacement/density/division strata.

Where a diagnostic requires reproducing private organizer helper logic, label it
non-authoritative and prove its totals against organizer counts. Final edge and
division counts always come from `evaluate`.

## Baseline Comparison and Bootstrap

**Recommended design:** comparison is paired by sample ID on the same frozen
manifest. Never compare reports produced with different scorer, manifest,
round-trip, or policy hashes.

Use a paired, embryo-stratified movie bootstrap:

- RNG: NumPy `Generator(PCG64)` with a policy-declared seed;
- default repetitions: 10,000;
- within each embryo, sample complete movie IDs with replacement while preserving
  that embryo's movie count;
- use the same resampled IDs for baseline and candidate;
- recompute organizer aggregation from resampled per-movie sufficient-stat rows;
- retain candidate-minus-baseline score, adjusted-edge, raw-edge, division, and
  node-recall deltas;
- report percentile 2.5/50/97.5 percentiles and `P(delta > 0)`;
- preserve organizer behavior in a replicate with no division events by dropping
  the division term, while recording the number of such replicates.

Do not bootstrap edges as independent observations; the movie is the held-out
biological unit available here. The interval is a stability diagnostic, not proof
that two embryos represent all hidden embryos.

Define `worst_movie_delta` as the minimum paired per-movie final-score delta and
record the movie ID and complete component deltas. Also report the lowest absolute
candidate movie so a shared baseline failure is visible.

## Promotion Policy

Create `config/promotion-policy.json` with version, exact numeric thresholds,
metric-field names, scorer/manifest requirements, and its own SHA-256. Separate
hard integrity gates from scientific performance gates.

### Hard rejection

Return deterministic `reject` for any scorer/source/dependency mismatch, manifest
mismatch, source overlap, missing/extra movie, invalid graph/CSV, non-finite
required metric, round-trip disagreement, known metric-exploit signature, missing
baseline pairing, or missing decision evidence. Public leaderboard score never
appears in this decision input.

### Default promote

The initial policy should require all of:

- pooled exact final-score delta strictly above a versioned minimum (the phase
  requirement permits `0`; a small practical minimum may be chosen explicitly);
- both embryo/fold score deltas at or above a small declared bilateral tolerance;
- paired-bootstrap lower bound above a declared stability floor;
- pooled micro node-recall delta `>= 0`;
- division Jaccard delta `>= 0` and no loss of division TP unless explicitly
  reviewed;
- worst-movie delta above a declared collapse floor;
- identical complete coverage and all hard gates passed.

Threshold values such as bilateral tolerance, bootstrap floor, and worst-movie
floor are project choices, not organizer facts. Freeze them before evaluating a
candidate; do not tune them after seeing the candidate result.

### Review required

A complete, integrity-clean candidate that fails a soft scientific gate may become
`review_required`, never automatic `promote`. An exception record must identify
the failed gates, quantitative trade-off, policy/report hashes, approver, reason,
and downstream experiment authorized. The existing ledger currently allows
`promote`, `retain`, `retire`, and `inconclusive`; implementation should add a
first-class `review_required` decision or use a new promotion-event schema. Do not
silently map it to promotion.

## Suggested CLI and File Layout

```text
config/
├── official-scorer.lock.json
├── evaluation-policy.json
└── promotion-policy.json
vendor/
└── kaggle-cell-tracking-competition/   # pinned checkout/submodule, ignored artifacts
src/biohub_tracker/
├── scorer_lock.py
├── manifests.py
├── graphs.py
├── submission_io.py
├── evaluation.py
├── diagnostics.py
├── comparison.py
└── promotion.py
tests/fixtures/metric/
├── graph_specs/
├── csv_roundtrip/
└── expected/
manifests/
├── reciprocal-embryo-v1.json
└── README.md
reports/exact/                         # compact reports tracked as policy permits
```

Suggested command surface:

```text
biohub scorer verify
biohub manifest build --data-root ... --output ...
biohub manifest verify --manifest ... --data-root ...
biohub graph validate --pred-dir ... --manifest ... --fold ...
biohub evaluate exact --baseline-dir ... --candidate-dir ... --manifest ...
biohub promote evaluate --report ... --policy ... --run-id ...
```

Every mutating command should write immutable or atomic content-addressed output,
and the CLI should print the core evidence hash.

## Plan Decomposition

### Plan 02-01 — Pin and prove the authoritative scorer

- materialize the organizer checkout/lock and exact environment lock;
- build the thin scorer-provenance adapter;
- port/freeze upstream ordinary-edge, adjusted-penalty, patched-division,
  aggregation, and exploit fixtures;
- verify the dependency pin and source hashes;
- document that `evaluate_datasets` and raw per-movie means are not final scoring.

### Plan 02-02 — Freeze leakage-safe data and submission identity

- implement deterministic train-root discovery and reciprocal embryo manifests;
- record metadata/content fingerprints and overlap audits;
- implement complete-coverage and graph-integrity validation;
- implement canonical submission-space GEFF/CSV round trips and parity fixtures;
- test unknown identity, duplicate membership, missing movie, sentinel, and schema
  failures.

### Plan 02-03 — Produce exact reports and diagnostics

- evaluate baseline/candidate sequentially per complete movie through fresh graph
  copies;
- aggregate pooled/by-embryo/by-fold/by-movie official metrics;
- add endpoint, oracle, conditional-link, displacement, density, division-window,
  and worst-movie diagnostics;
- implement paired stratified deterministic movie bootstrap;
- prove canonical report-core regeneration.

### Plan 02-04 — Enforce promotion and ledger integration

- implement versioned hard/soft policy and `review_required` state;
- append report/policy/manifest/scorer hashes to the experiment ledger;
- extend progress rendering with exact gate status;
- execute a CPU-only end-to-end report on real complete predictions once data and
  baseline/candidate GEFF artifacts are mounted.

All four plans consume zero Kaggle GPU hours. Real-data execution may use Kaggle
CPU or a local mount, but must not launch a GPU job.

## Unresolved Blockers and Decisions for Planning

1. **No real training data or prediction graphs are local.** Synthetic tests and
   implementation can proceed, but a genuine reciprocal exact report cannot be
   produced until official train GEFF/Zarr metadata plus baseline/candidate GEFFs
   are mounted or retrieved through a bounded CPU job.
2. **The organizer dependency environment floats.** Start from patch-era
   TracksData `39dccf3...`, freeze a full environment, and require parity. Record
   that the private scoring container lock remains unpublished.
3. **Remote inventory enumeration hit HTTP 429.** The manifest builder must count
   the mounted dataset itself. Do not promote using the contextual “199 movies”
   count without a generated coverage proof.
4. **Kaggle's Evaluation page retains a short pre-patch connected-component
   description.** Its own link points to the organizer repository, and the
   organizer patch announcement plus current pinned source define the local
   authority. Record both page and source hashes; prefer the patched source for
   executable semantics.
5. **Promotion thresholds beyond the qualitative requirements are project
   choices.** Freeze numeric tolerances before seeing Phase 3 candidate results.
6. **Round-trip parity requires integer submission space.** Decide explicitly
   whether earlier native subvoxel results are retained only as diagnostics or
   recomputed from their original prediction artifacts.

## Primary Sources

- [Pinned organizer repository](https://github.com/royerlab/kaggle-cell-tracking-competition/tree/075fc5f5a52d11077f9dc2b074644618f26939e2)
- [Patched metric specification](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/075fc5f5a52d11077f9dc2b074644618f26939e2/metrics.md)
- [Authoritative metric implementation](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/075fc5f5a52d11077f9dc2b074644618f26939e2/src/tracking_cellmot/metrics.py)
- [Patched division implementation](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/075fc5f5a52d11077f9dc2b074644618f26939e2/src/tracking_cellmot/division_metrics.py)
- [Organizer evaluation script](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/075fc5f5a52d11077f9dc2b074644618f26939e2/scripts/evaluate.py)
- [Organizer GEFF→CSV helper](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/075fc5f5a52d11077f9dc2b074644618f26939e2/scripts/geffs_to_csv.py)
- [Organizer CSV→GEFF helper](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/075fc5f5a52d11077f9dc2b074644618f26939e2/scripts/csv_to_geffs.py)
- [Exploit-patch commit](https://github.com/royerlab/kaggle-cell-tracking-competition/commit/aa65e90aeb8a774ebb1b549e547787b87ac8a01c)
- [Patched merge commit](https://github.com/royerlab/kaggle-cell-tracking-competition/commit/075fc5f5a52d11077f9dc2b074644618f26939e2)
- [Organizer metric-patch announcement](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/727154)
- [Official competition data description](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/data)
- [Official competition evaluation page](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/overview/evaluation)
- [Patch-era TracksData source](https://github.com/royerlab/tracksdata/tree/39dccf3a243e44274759468cb31b2ad9e7fc1d09)
- [TracksData distance matching implementation](https://github.com/royerlab/tracksdata/blob/39dccf3a243e44274759468cb31b2ad9e7fc1d09/src/tracksdata/metrics/_matching.py)
- [TracksData per-timepoint optimal matching](https://github.com/royerlab/tracksdata/blob/39dccf3a243e44274759468cb31b2ad9e7fc1d09/src/tracksdata/metrics/_ctc_metrics.py)
- [GEFF reference implementation/specification](https://github.com/live-image-tracking-tools/geff)

## Research Conclusion

Phase 2 is implementable without GPU and without inventing metric semantics. The
organizer source and patch are sufficiently explicit to build a high-confidence
local authority, provided the implementation compensates for upstream
fail-open coverage behavior and records the unresolved dependency-container
provenance honestly. Synthetic/adversarial implementation can start immediately;
real reciprocal validation remains blocked only on mounting official data and the
actual baseline/candidate prediction graphs.
