# Stack Research

**Domain:** Sparse-supervision 3D+time cell detection, tracking, and Kaggle competition operations  
**Researched:** 2026-08-23  
**Confidence:** High for the baseline and operational stack; medium for the final model scale until a T4 throughput benchmark is run

## Recommended Core

| Layer | Choice | Why | Confidence |
|-------|--------|-----|------------|
| Language | Python 3.11 or the exact Kaggle notebook Python version | The official project supports Python `>=3.11,<3.14`; matching Kaggle prevents binary and serialization drift | High |
| Deep learning | PyTorch, pinned to the Kaggle-compatible version | The official detector/linker is PyTorch and current AMP uses `torch.amp.autocast` plus `torch.amp.GradScaler` | High |
| Competition base | Pin a commit of `royerlab/kaggle-cell-tracking-competition` | Reuses the authoritative I/O, scorer, transforms, detector/linker contract, and CSV/GEFF round trip | High |
| Image storage | OME-Zarr / Zarr 3 through the official loader | Native competition format is `(T,Z,Y,X)` OME-Zarr; chunk-aware reads are necessary for T4 throughput | High |
| Graph storage | `tracksdata` + GEFF | Native sparse graph representation and official metric integration | High |
| Numeric/data | NumPy, SciPy, Polars | Matches official baseline dependencies and supports assignment, graph tables, and fast validation | High |
| Environment | `uv.lock` locally plus an exported Kaggle wheel/dataset manifest | Reproducible training and an Internet-disabled submission notebook require two explicit environments | Medium |
| Tests/lint | pytest and Ruff | Official repository already uses them; add scorer regression, coverage, quota, and schema tests | High |
| Experiment registry | Git-tracked JSONL + CSV/Markdown projections, with large artifacts referenced by SHA-256 | Works offline, diffable, append-only, Kaggle-friendly, and avoids depending on a hosted tracker | High |
| Orchestration | PowerShell wrappers locally; Python entry points inside Kaggle | The host is Windows/PowerShell while Kaggle is Linux; policy belongs in platform-neutral Python where possible | High |

## Model Stack

Start from the official joint `TemporalUNet3D` and `SimpleNodeTransformer` contract rather than replacing every component at once. The first training candidate should add capacity and supervision where current evidence says the ceiling lies:

- Residual 3D U-Net encoder/decoder with temporal attention at two scales.
- Detection heatmap plus subvoxel offset and uncertainty heads.
- Learned temporal displacement/affinity features pooled at predicted candidate nodes.
- Cross-attention association transformer trained on predicted detections as well as ground-truth nodes.
- Explicit division head or fork-ranking objective using rare-event sampling.
- Optional ZebraHub selective-SSM affinity features only after reciprocal calibration passes.

Use FP16 AMP on Kaggle T4, gradient accumulation, activation checkpointing only where memory requires it, and persistent resumable checkpoints. Do not enable `torch.compile` by default: dynamic graph/candidate shapes and compilation startup may erase the gain within short Kaggle runs. Benchmark it as an isolated option.

## Dependency Policy

1. Pin the official baseline commit and record it in every run.
2. Freeze the exact Kaggle package inventory used for each promoted model.
3. Package weights as private Kaggle datasets for inference; never assume Internet access.
4. Record the license and public source URL of every external dataset and pretrained checkpoint.
5. Keep credentials, downloaded competition data, weights, Zarr caches, and notebook outputs out of Git.

## What Not to Use Initially

- A hosted-only experiment tracker: it cannot be relied on inside Internet-disabled submission notebooks.
- A wholesale MONAI/SwinUNETR migration before a baseline throughput profile: the additional abstraction and memory cost would consume the quota without isolating the source of gains.
- FP32 training on T4: prior competition reports and PyTorch guidance make mixed precision the practical default.
- A single public-checkpoint ensemble as the main research lane: current clean notebooks already saturate that family around the same score range.

## Sources

- [Official competition overview and code constraints](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/overview)
- [Official baseline repository](https://github.com/royerlab/kaggle-cell-tracking-competition)
- [Official baseline dependency specification](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/pyproject.toml)
- [PyTorch AMP documentation](https://docs.pytorch.org/docs/stable/amp.html)

