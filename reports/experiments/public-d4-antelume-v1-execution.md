# Antelume public D4 correction — September 10, 2026

## Terminal update,22:51 UTC

Full job COMPLETE932.901s; bothGPUsmokes+full983.012s jobwalltime. All8graphs
and57terminal/supportingfiles backed up and remote/local hash-verified. GPU
compute processes empty at22:43–22:44; shared instance left running. CPU rescore
COMPLETE: pooled0.944839→0.947619, but both6bba movies regress and the frozen
embryo/movie gate FAILS. No promotion/submission;0/5. Sparse-annotation wrapper
correction and full-file checks documented in public-d4-full-movie-v1-result.md.
CPU attribution shows all1,418annotated cells matched by correctedrawdetector;
losses arise later in graphselection/repair/pruning/localization. This is not
independentCV or exhaustivecellrecall. Next work can use cached predictions onCPU.
All earlier running descriptions below are historical checkpoints.

## 22:30 UTC execution checkpoint

User confirmed Antelume is up. One idle A10G was verified before each launch;
RSNA and shared environments were not altered. Isolated working directory:
`/tmp/biohub-d4-preflight-v1.o4fsjO` on instance `i-0d12195df0d3558f3`.
No Kaggle GPU hours consumed; no competition submission made.

### Completed functionality evidence

- Actual extracted public encoder/DeepCenter code also passes the independent
  NumPy execution audit: square/rectangular cases, feature averaging, cache
  behavior; maximum error about 1.11e-16. This is not model-quality evidence.
  Audit SHA256: `44781088b3be6c7365db315f0cead7b1b94ededecdf884452a8e763aa098da18`.
- Real GPU paired preflight: **21.614 seconds**, exact primary/secondary best
  checkpoints and epoch-2 DeepCenter. Original encoders and DeepCenter each
  executed 8 calls but 7 unique inputs; corrected arm executed 8 unique inputs.
  Finite, nonidentical model outputs; peak encoder allocation 654,326,784 bytes.
  Receipt SHA256: `d7e1d92a263941ab14563f402e093ab12e48c388f0cd4bf7dad12ffb5ca405ac`.
- Four complete image movies downloaded directly to the isolated directory:
  408 files, **1,578,992,913 bytes**, 22.947 seconds. GCS size/MD5 and local
  SHA256 checks, all 100 frames per movie present. No truth or Kaggle account
  credentials copied to Antelume. Short-lived signed URLs stay in ignored,
  private transport files and never appear in reports.
- End-to-end eight-frame paired smoke: **28.496 seconds**, original 1,620 nodes /
  1,413 edges; corrected 1,621 / 1,413. Includes real association, exact public
  ILP objective, public repair/DeepCenter, CSV-equivalent integer coordinates,
  schema and lineage checks. Both arms passed; no truth opened or scores taken.
  Smoke SHA256: `6574d8d2a9e2e2ffa21d1ac1b94a0e09f09269900dabf8a340d3bca5be791cde`.
- **21 focused tests passed in 2.65 seconds**, covering source correction,
  preflight extraction, executed geometry, frozen config, graph checks, and
  predeclared quality gates.

Cold/warm preflight timings differ: do not claim a twofold speedup. Full paired
throughput is the relevant evidence. Gurobi is unavailable; the existing
tracksdata solver successfully uses its SCIP fallback. No package installed.

### Running complete-movie diagnostic

Started approximately 22:28 UTC, local session **29029**, remote PID **2973**.
Sequential original/corrected arms for:

- `44b6_24264f12`
- `44b6_81c256f0`
- `6bba_23af9eeb`
- `6bba_f1fde7e0`

Contract SHA256:
`45af25de58dc146d0820dd705dde80cc5b7f678e285938fa349ed95363bff56f`.
Runtime output: `/tmp/biohub-d4-preflight-v1.o4fsjO/full-v1`.
One-hour internal watchdog plus outer 3,620-second bound. Two CPU threads,
one CUDA GPU, allocator ceiling 70%; no parallel experiments. Initial estimate
15–25 minutes is provisional. At 22:30, first arm had finished inference/ILP
and was actively doing image-guided post-processing (GPU utilization 100%).

Only scientific change: six D4 rotation arguments in public primary/secondary
detection/association, plus two in DeepCenter. Runtime adapters extract reviewed
definitions, relocate two log paths, and replace broad I/O with strict image-only
metadata access. Normalization, scales, weights, hyperparameters, solver objective,
repair logic and final integer rounding remain the public reference's settings.
Public notebook installers, proxy sweeps, test selection and prediction downloads
are never executed. Original source and prior completed experiments stay frozen.

### Frozen decision and remaining work

All eight graphs must be persisted and verified before local truth is read.
Score using patched official commit `075fc5f5a52d11077f9dc2b074644618f26939e2`.
Require strictly positive pooled combined and raw edge-Jaccard gain, nonregressing
combined score for each embryo and each movie, and explicit worst-movie analysis.
No threshold search in response to these results.

Scorer prepared before results:
`scripts/score-public-d4-full-movie-v1.py`, SHA256
`28c630797c7298da05be40376a4de7cb920ad6cfdd773e146e44ef552a676b93`;
comparison SHA256 `59b0090f8f68632faccdb7101c07ecbfc3f3a9077f96bf181bc80e595456bd36`.

**These are previously exposed, public-checkpoint training diagnostics, not
independent validation.** A pass is useful evidence for this narrow bug fix,
but does not by itself satisfy the project's submission-promotion gate.
No quality gain or superiority over reported public 0.947 has been established.
Qualified submissions today remain **0/5**. Harvest predictions and logs, release
our GPU process naturally, rescore locally, and record the actual decision.

## 22:35 UTC verification and environment

First complete paired movie persisted: original160.734s / corrected160.441s;
27,090 / 27,206 nodes and26,147 /26,264 edges. These are output counts, not scores.
The longer run remains active. Synthetic-only official-scoring and complete-file
handoff tests now bring the focused suite to **23 passed in24.98s**; no real
validation labels opened by those tests.

Read-only inspection of the actual shared environment (no changes): Python3.12.3,
torch2.5.1+cu121, NumPy2.5.2, SciPy1.18.1, Polars1.44.1, tracksdata0.1.0rc8,
Zarr3.3.0, GEFF1.3.1.1.3, geff-spec1.3.0, PySCIPOpt6.2.1, ilpy0.6.0,
blosc2 4.11.0, numcodecs0.16.5, tqdm4.70.0. Both arms use this same environment.
It is **not** a claim of bitwise reproduction of the original Kaggle Docker image.

Package metadata reports common open-source licenses for these selected roots;
this is not a completed transitive binary-license audit. The public upstream
[PySCIPOpt license](https://github.com/scipopt/PySCIPOpt/blob/master/LICENSE)
is MIT and [SCIP license](https://github.com/scipopt/scip/blob/master/LICENSE)
is Apache-2.0. Exact bundled solver/binary notices still belong in a distribution
manifest; a future offline Kaggle acceptance run must verify package compatibility.
The rules page remained inaccessible as browser text; this does not constitute a
fresh full rules review. The successfully captured19:23UTC rules review remains
the latest usable rules evidence in biohub-rules-refresh-20260910-1923.md.
