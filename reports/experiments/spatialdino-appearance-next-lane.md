# SpatialDINO appearance association lane

Status: local feasibility passed; not scheduled ahead of the active HOCT run.

## Motivation

The current HOCT adapter supplies exact time and physical position but replaces
unavailable morphology/intensity channels with the checkpoint training means.
It is therefore an intentionally geometry-led model. SpatialDINO offers a
complementary signal: frozen dense 3D fluorescence-microscopy embeddings that
the upstream project uses for segmentation and tracking in crowded 4D data.
This lane will not copy the upstream tracking output. It will sample frozen
appearance embeddings at our independently chosen detector nodes and add only a
bounded parent-choice term to an accepted association model.

## Bound upstream artifact

- Repository: <https://github.com/kirchhausenlab/spatialdino>
- Inspected commit: `ca3ab86b34430d963f12a3909baaeb9343c63b7d`
- License: MIT
- Checkpoint: upstream GUI default `vits8`, step 244999
- Downloaded bytes: 86,068,439
- SHA-256: `47f199d2e8644ca11be2d5679494bd9607f9e9a7a0b448e35b85391d70c94ed8`
- Parameter count: 21,501,312
- Patch grid: 8 x 8 x 8; patch embedding dimension: 384

The released no-xFormers import path references an unavailable `SwiGLU` base
even though this checkpoint uses the ordinary MLP. A temporary local
compatibility shim was required for the smoke test. Any Kaggle runtime must
vendor that one-line fallback explicitly and test both checkpoint loading and
no-xFormers inference before launch.

## Local evidence

A real checkpoint load matched all 174 tensors. A CPU forward pass on a
single-channel 64 x 64 x 64 volume returned finite
`[1, 390, 8, 8, 8]` features in 0.291 seconds: 384 patch channels plus six
attention-head channels. The tracked adapter samples only the patch channels,
maps raw Biohub z into a four-times isotropic encoder grid, L2-normalizes node
embeddings, and derives candidate-local cosine margins. Unit tests cover patch
center alignment, trilinear interpolation, candidate grouping, and fail-closed
shape/bounds checks.

## Experiment contract

1. Freeze the upstream backbone; never train on leaderboard feedback.
2. Extract features only for the four clean validation movies first. Use the
   same two selection movies and two untouched acceptance movies as HOCT.
3. Test zero appearance weight plus a small predeclared bounded grid. Appearance
   may resolve competing parents but may not create nodes or long-range edges.
4. Apply the term only where a target has multiple plausible parents. Preserve
   base support for single-candidate edges and cap any logit adjustment.
5. Promote only for positive pooled acceptance delta, no embryo-prefix
   regression beyond 0.01, and improvement on ambiguity-stratified edges.
6. If accepted, run the frozen encoder on the four test movies and compose it
   with the accepted linker. The candidate must remain edge-different from the
   public submission and must pass the normal submission-integrity audit.

This is the preferred next association experiment because it adds information
that the public U-Net/linker family and our point-only HOCT adapter do not
currently consume. The positive-unlabeled Spotiflow detector remains the next
end-to-end node lane.
