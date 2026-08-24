# Reciprocal Embryo Manifests

Production manifests are generated only from a mounted official training root.
This directory intentionally does not contain `reciprocal-embryo-v1.json`; the
fixture data under `tests/` is not evidence of official movie coverage.

Build from the complete mounted train directory:

```powershell
.biohub/evaluation-venv/Scripts/python.exe -m biohub_tracker manifest build `
  --data-root D:/biohub/train `
  --scorer-lock config/official-scorer.lock.json `
  --output manifests/reciprocal-embryo-v1.json
```

Verify the self-hash, reciprocal memberships, overlap audit, and mounted source
inventory before an evaluation:

```powershell
.biohub/evaluation-venv/Scripts/python.exe -m biohub_tracker manifest verify `
  --manifest manifests/reciprocal-embryo-v1.json `
  --data-root D:/biohub/train `
  --scorer-lock config/official-scorer.lock.json
```

The builder derives the actual movie count. It never substitutes a contextual
count, random split, clip split, missing-pair intersection, or default scale.
The manifest semantic hash binds sorted sample identities, exact metadata used
for validation, GEFF tree hashes, each reciprocal fold membership, and the
machine-readable overlap audit. Creation time is evidence metadata outside the
semantic core.

Acceptance for the production file requires Plan 02-04's CPU-only mounted-data
control, local verification, and immutable ledger reconciliation. No Kaggle GPU
job or competition submission is part of manifest generation.
