# CELLECT + SpatialDINO transfer audit

Date: 2026-08-26

Status: promising independent next-model design; no Kaggle run launched.

## Why this source is unusually relevant

[CELLECT](https://www.nature.com/articles/s41592-025-02886-x) is a 2025
Nature Methods method built for adjacent-frame 3D microscopy. Its feature
network jointly predicts cell-center confidence, a 64-channel voxel embedding,
cell size, and division probability. Contrastive supervision pulls embeddings
of the same tracked cell together and separates different cells. The paper
reports cross-modality generalization from C. elegans embryo data and explicitly
tests confocal and light-sheet volumes, making it more aligned with Biohub than
generic natural-image or spot-detection pretraining.

The paper's [official repository](https://github.com/zzz333za/CELLECT) was
audited at commit `3586070926f7f1fd5d8df37456861d22bdc63236`. It includes source,
pretrained weights, and a GPL-2.0 license. The license permits use and
modification but requires distributed derivative source to remain GPL-2.0;
that is compatible with an open-code competition workflow but must be retained
in any shipped derivative.

## Reproducible checkpoint audit

The `x3rdstr0` checkpoint trio loads strictly with the repository's current
fallback architecture:

| Role | Parameters | SHA-256 |
|---|---:|---|
| two-frame 3D feature network | 2,541,972 | `3b56e5e58a5b5fd636edb53dbc994c95ac80a15edea0b3a13f9faf2d1c2e6117` |
| inter-frame matcher | 191,850 | `ac8e92259a1cbe345bc4741094fd1125a21f27138d5c259def023aa1bd89048f` |
| intra-frame matcher | 191,716 | `d0bb6e14f68455940cd70a46beb279e5447864cc0f943fd3386135240f6abccf` |

The alternative `x3rd` feature checkpoint is
`f3e8e7303976dc5fb6e52ade12d1ff7953d2a28e30c06274b9f309562cd89ce5`.
Both families are small enough for a clean transfer screen before spending a
large training budget.

## Why direct submission is rejected

Submitting CELLECT output directly would replace one replica concern with
another external pretrained pipeline and would not test the user's heavy,
Biohub-adapted hypothesis. The upstream inference code also contains hardcoded
device and threshold behavior, assumes `(H,W,Z)` axis order, and retries a
different architecture through a broad exception. Those behaviors must not be
used verbatim in a guarded notebook.

## Strong candidate design

Use CELLECT only as a public, frozen temporal-embedding teacher and transfer
initialization inside a new Biohub-trained model:

1. Retain the 29.5M SpatialDINO/UNETR detector as the semantic and
   high-resolution path.
2. Add adjacent-frame temporal fusion trained on Biohub links, with a
   64-channel contrastive embedding head and a calibrated division head.
3. Distill CELLECT embedding relations only on geometrically feasible
   neighbors; organizer links provide the authoritative positive pairs and
   hard negatives come from nearby different cells.
4. Learn center heatmap, association embedding, and division jointly while all
   twelve clean validation movies remain excluded.
5. Validate detection first, then degree-preserving association on disjoint
   complete movies. Never use leaderboard score for model or threshold
   selection.

This yields an approximately 32.4M-parameter starting system before the new
temporal heads, whose final checkpoint is learned on Biohub and whose outputs
are not copied from either CELLECT or the public Kaggle notebooks.

## Launch order

Do not contend with the active `spatialdino-pu-adaptation-v1` run. First consume
that run's clean evidence, then run the already-preflighted SpatialDINO
degree-preserving appearance repair. A CELLECT transfer screen is justified
only if the active detector misses recall or the appearance experiment shows
that current SpatialDINO patch features are too coarse to select useful swaps.
