<!-- GSD:project-start source:PROJECT.md -->

## Project

**Biohub Cell Tracking Competition**

This is a competition operating system and modeling workspace for the Kaggle Biohub Cell Tracking During Development challenge. It gives one competitor a repeatable way to audit the live competition, run resource-safe experiments, compare them with exact out-of-fold evidence, train stronger generalizable cell-detection and lineage-linking models, and ship reproducible submissions before the September 29, 2026 deadline.

The workspace treats leaderboard pages and post-processing folklore as hypotheses, not evidence. Every promoted change must survive the patched official metric, complete-movie validation, embryo-held-out checks, resource accounting, and a no-metric-hack audit.

**Core Value:** Reliably improve the clean, private-test-generalizable Biohub tracking score without exploiting the metric or consuming the final 8 hours of Kaggle GPU quota.

### Constraints

- **GPU reserve**: Always preserve at least 8 hours of Kaggle GPU quota; before launch, reject a job if current remaining hours minus its worst-case declared runtime would fall below 8
- **Current budget**: Audit snapshot shows 30 hours remaining and therefore at most 22 spendable hours; recompute from `kaggle quota` before every launch because quotas refresh and usage changes
- **Notebook runtime**: Kaggle competition submissions permit at most 12 hours on GPU or CPU; internal watchdogs must stop earlier and persist resumable checkpoints
- **Execution**: GPU experiments run sequentially so quota, output ownership, and attribution remain unambiguous
- **Inference**: Competition submission notebooks must run with Internet disabled and produce `submission.csv`
- **Timeline**: Team merger and entry deadline is 2026-09-22 23:59 UTC; final submission deadline is 2026-09-29 23:59 UTC
- **Submission limits**: At most five submissions per day and at most two selected final submissions
- **Data**: External data and pretrained models must be public, free or reasonably accessible, documented, and license-compatible
- **Licensing**: Winning code must be reproducible and released under the required open-source terms; dependencies must permit commercial use
- **Reproducibility**: Training and inference code, environment, hashes, architecture, preprocessing, losses, seeds, and hyperparameters must be recoverable
- **Metric integrity**: The patched official scorer is authoritative; no metric hack may enter training selection, validation, or submission generation
- **Validation**: Promotions require complete-movie micro-averaged scoring, per-embryo reporting, and worst-movie analysis
- **Workspace**: Git tracks project planning and code; generated datasets, model weights, Kaggle downloads, secrets, and large artifacts stay out of Git

<!-- GSD:project-end -->

<!-- GSD:stack-start source:research/STACK.md -->

## Technology Stack

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

- Residual 3D U-Net encoder/decoder with temporal attention at two scales.
- Detection heatmap plus subvoxel offset and uncertainty heads.
- Learned temporal displacement/affinity features pooled at predicted candidate nodes.
- Cross-attention association transformer trained on predicted detections as well as ground-truth nodes.
- Explicit division head or fork-ranking objective using rare-event sampling.
- Optional ZebraHub selective-SSM affinity features only after reciprocal calibration passes.

## Dependency Policy

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

<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->

## Conventions

Conventions not yet established. Will populate as patterns emerge during development.
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->

## Architecture

Architecture not yet mapped. Follow existing patterns found in the codebase.
<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->

## Project Skills

No project skills found. Add skills to any of: `.claude/skills/`, `.agents/skills/`, `.cursor/skills/`, `.github/skills/`, or `.codex/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->

## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:

- `$gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `$gsd-debug` for investigation and bug fixing
- `$gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->

<!-- GSD:profile-start -->

## Developer Profile

> Profile not yet configured. Run `$gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
