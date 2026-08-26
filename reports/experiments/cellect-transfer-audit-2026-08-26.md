# CELLECT transfer audit — 2026-08-26

## Outcome

Retain CELLECT as a future association/division candidate, not as the immediate
fallback detector. Its official release is materially different from public
Kaggle notebooks and supplies pretrained 3D temporal embeddings, but its code
and weights need a deliberate GPL-compatible integration boundary.

## Primary sources

- Nature Methods paper: https://www.nature.com/articles/s41592-025-02886-x
- Official repository: https://github.com/zzz333za/CELLECT
- Audited repository commit: `3586070926f7f1fd5d8df37456861d22bdc63236`
- Repository license: GPL-2.0

The paper reports sparse-annotation 3D cell tracking with a two-frame U-Net,
contrastive per-cell embeddings, an intra-frame MLP, and an inter-frame/division
MLP. It is directly relevant to the challenge's association and division terms,
whereas LSM-FM is currently testing detector localization.

## Audited release

The repository contains two complete pretrained triplets. Safe tensor-only
inspection found:

- U-Net: 2,541,972 parameters
- inter-frame MLP: 193,156 parameters
- intra-frame/division MLP: 193,022 parameters
- total per triplet: 2,928,150 parameters

Default inference uses two consecutive 3D frames, patch-wise processing, and a
`zratio` of 5. Biohub's raw voxel geometry is 4:1 anisotropic in Z, so the model
has a promising geometry prior but must be recalibrated rather than run at its
default.

Pretrained SHA-256 values:

- `U-ext+-x3rd-149.0-4.6540.pth`: `f3e8e7303976dc5fb6e52ade12d1ff7953d2a28e30c06274b9f309562cd89ce5`
- `EX+-x3rd-149.0-4.6540.pth`: `4b0dd794c525e2053a0fe02a7d60253597d6cc669d0d4840b5ffe6481762b07e`
- `EN+-x3rd-149.0-4.6540.pth`: `deb21e312309f81f3784ff49f6ac2de0f9bf623453f0fe5394b317ac28666933`
- `U-ext+-x3rdstr0-149.0-3.4599.pth`: `3b56e5e58a5b5fd636edb53dbc994c95ac80a15edea0b3a13f9faf2d1c2e6117`
- `EX+-x3rdstr0-149.0-3.4599.pth`: `ac8e92259a1cbe345bc4741094fd1125a21f27138d5c259def023aa1bd89048f`
- `EN+-x3rdstr0-149.0-3.4599.pth`: `d0bb6e14f68455940cd70a46beb279e5447864cc0f943fd3386135240f6abccf`

## Constraints and next gate

The repository has a GPL-2.0 license and does not state a separate license for
the model weights. Do not copy its implementation into the current permissive
runtime without an explicit GPL-compatible distribution decision. A safe next
experiment would either:

1. isolate the official GPL pipeline as a clearly attributed executable and
   publish the corresponding source, or
2. use the paper only to design an independently trained temporal-embedding
   model without importing the official weights or code.

Any CELLECT-derived candidate must use frozen complete-movie validation and
cannot be promoted by leaderboard score. It should be tested only after the
current LSM-FM detector result clarifies whether detection or association is the
next bottleneck.
