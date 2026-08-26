# public-0927-clean-repro-v1

Status: preflight retry (`public-0927-clean-repro-v2`)

## Hypothesis

The public dual-seed TemporalUNet3D + node-transformer + ILP pipeline can improve the current clean public score from `0.913` toward `0.927` when it adds harmonic forward/reverse association, intensity-weighted centroid refinement, and constrained division repair.

## Candidate lineage

- Source kernel: `evgendvorkin/biohub-0-927-lb`
- Source kernel status: `COMPLETE`
- Source notebook SHA-256: `08507f9123d9f40e185d0db8eda3dd21cb405e50febb720827c2e655f68d5ec1`
- Fork notebook SHA-256 after watchdog injection and Windows-safe JSON normalization: `19d4c0f525be6325689af7bc37180098e5d19556d53636ed042e819b7fdeea24`
- Scientific-cell comparison: all 33 upstream cells are exact; the fork adds only one leading watchdog cell and one terminal-evidence cell.
- Upstream execution span: 33.727 minutes.
- Fork accelerator policy: GPU enabled; TPU and internet disabled; private kernel.

## Public model inputs

- `pilkwang/biohub-tracking-support-pack-50ep-v1`
  - primary checkpoint SHA-256: `12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771`
- `pilkwang/biohub-temporal-unet3d-seed314159-v1`
  - secondary checkpoint SHA-256: `9bac2fa0dadc4a6fc1899e0caf187f4b553e0a7cd90ba1261a68b35ffe9e305f`
- `pilkwang/biohub-deepcenter-unet3d-center-prior-v1`
  - DeepCenter checkpoint SHA-256: `8040999a92f6b7bbd98fa8cf458141e045c0f9ad7c936bdb3b18e1f7edafe2a0`

## Main configuration

- Detection threshold: `0.96875`
- Primary/secondary association ensemble: secondary edge weight `0.15`, detection weight `0.475`
- Harmonic reverse-association weight: `0.30`
- Minimum dual-seed frame retention: `0.90`
- Gap close distance: `5.8 um`
- Division parent/sister/existing-child radii: `8/11/10 um`
- Division constraints: mid-track parent, mutual-nearest orphan daughter, forward divergence, and DeepCenter veto
- Centroid refinement: local background-subtracted intensity centroid, maximum shift `2.8 um`

## Clean-method audit

Passed for this candidate.

- Submission generation reads test images and public model checkpoints only.
- The upstream runtime integrity report records `ground_truth_accessed=false` for test prediction.
- No artificial sentinel nodes, negative coordinates, fixed test labels, or submission-derived score manipulation are present.
- Output topology is valid: maximum indegree `1`, maximum outdegree `2` across all four test movies.
- Metric code and training labels occur only in a held-out validator after test submission generation, with explicit train/test stem exclusion.
- Two superficially strong public notebooks were excluded because they append artificial `-10000` coordinate nodes/edges after inference.

## Upstream held-out evidence

Four embryo-held samples, complete movies:

| Metric | Value |
|---|---:|
| Edge TP / FP / FN | 2186 / 108 / 107 |
| Weighted adjusted edge Jaccard | 0.915158 |
| Division TP / FP / FN | 1 / 2 / 4 |
| Division Jaccard | 0.142857 |
| Proxy score (`edge + 0.1 * division`) | 0.929444 |

Validator CSV SHA-256: `b2b18eaeff608dae26202fe2a2c42aeac1987e0f592becd3dc75a2e0e490e777`.

## Upstream test-output evidence

- Test movies: 4 complete movies, 100 frames each.
- Submission rows: 240,126.
- Submission SHA-256: `33c179b0449b9cdd186f06a653cddc8cf12359f008982f6713cdf30784a52e6a`.
- Learned prediction time reported in `run_stats.csv`: 9.101562 minutes total.
- Frame-retention audit: 400 frames covered; 60 frames fell back to the primary detector when ensemble retention was below 0.90.

## Resource decision

- Declared maximum runtime: `1.00 h`.
- Hard watchdog stop: 50 minutes, leaving 10 minutes for output persistence.
- GPU quota before authorization: to be re-read live; last observed `30.00 h`.
- Protected reserve: `8.00 h`.
- No submission will be made until our fork completes and its output audit passes.

## Operational retry history

- `public-0927-clean-repro-v1` was authorized with `30.00 h` remaining and `29.00 h` projected after the declared run.
- Its guarded push failed before Kaggle created a kernel, so it consumed no GPU quota.
- Root cause: the local notebook utility emitted raw UTF-8, while Kaggle CLI on Windows opened the source with cp1252 and raised `UnicodeDecodeError` on multilingual markdown.
- `public-0927-clean-repro-v2` ASCII-escapes notebook JSON without changing any parsed scientific cell. A local default-codec read now passes.
- The launcher now records sanitized Kaggle stderr for any future nonzero push exit.
