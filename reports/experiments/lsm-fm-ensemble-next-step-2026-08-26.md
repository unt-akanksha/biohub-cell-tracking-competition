# LSM-FM feature-diverse ensemble candidate

Status: implemented and staged offline; not uploaded or launched.

## Hypothesis

The existing feature-24 detector and active feature-36 detector use different
official LSM-FM pretrained students. Feature-24 was initialized from the
image-only objective; feature-36 adds image-text alignment to masked
reconstruction and teacher distillation. Their held-out movie errors can be
combined without copying public predictions or public notebook code.

## Frozen comparison

The evaluator declares five candidates before reading selection labels:

1. feature-24 with the current probability centroid;
2. feature-36 with the current probability centroid;
3. feature-36 with log-probability quadratic refinement;
4. a fixed 0.5/0.5 probability ensemble with the current centroid;
5. the same fixed ensemble with log-probability quadratic refinement.

No ensemble weight is tuned. One candidate is chosen globally across all eight
selection movies. It must achieve pooled recall at least `0.80`, worst-movie
recall at least `0.65`, and no per-movie regression below `-0.01` relative to
feature-36. The four acceptance movies remain sealed until that choice is
frozen, and the existing promotion gates remain unchanged.

## Staging evidence

- Private runtime: `biohub-lsm-fm-ensemble-runtime-v1`
- Runtime manifest SHA-256:
  `81c70961de251cbd20442cd618fba57b972df7b52de113a2ad9822357b0f3fc1`
- Runtime bytes: `220186177`
- Focused tests: `12 passed`
- Public leaderboard used for selection: no
- Public predictions or Kaggle code copied: no
- Competition submission path: absent

The runtime will only be uploaded and the evaluator built if the active
feature-36 run produces a valid hash-bound checkpoint and the quota guard still
preserves the protected reserve.
