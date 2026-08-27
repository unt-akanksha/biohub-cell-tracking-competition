# Temporal patch v1 runtime-mount failure

Date: 2026-08-27

`indarkarhana/biohub-temporal-patch-dual-fold-v1` failed after 11.964 seconds,
before the production dense GPU probe, optimizer training, or any model output.
This is infrastructure evidence and says nothing about the model hypothesis.

## Cause

Kaggle exposed the dataset's uploaded `trackastra_source` directory as an
already mounted directory. Notebook setup copied top-level files and extracted
explicit ZIP files, but ignored mounted directories. The fail-closed runtime
verifier then correctly rejected the missing
`trackastra_source/trackastra/__init__.py`.

- Launcher terminal SHA-256:
  `e3a890cb886188f58768cf31003f50a2637ee9b31d7a12e8ad7f7679439d971c`
- Kernel log SHA-256:
  `a2c6a807cdcca991c2e604f6e68ec8641ac1fa2845187ce17d545e83f7dc236b`
- GPU quota before/after: `21.29 / 21.28` hours remaining.
- Public predictions used: false.
- Leaderboard selection used: false.
- Submission or competition artifact created: false.

## Repair

The exact generated materializer now:

- recursively copies mounted directories;
- extracts explicit ZIP archives into the same intended layout;
- rejects a directory/ZIP name collision as ambiguous;
- is regression-tested by executing the generated function against both mount
  representations.

The retry changes no model, data split, seed, objective, threshold, step count,
GPU count, wall-time ceiling, or submission policy. Its preflight report is
`artifacts/preflights/temporal-patch-dual-fold-v1-runtime-mount-retry-standard.json`,
SHA-256
`60cc59087dc3da8f37d00727f06d98bad63fc5c076a5ad27e949a320daf3f510`.
The first immutable retry report is retained as historical evidence of the
guard's safe refusal over noncanonical check names.
