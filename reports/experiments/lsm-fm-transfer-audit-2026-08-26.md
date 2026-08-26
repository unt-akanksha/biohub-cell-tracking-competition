# 3D light-sheet foundation-model transfer audit — 2026-08-26

## Decision

Retain the image-only SwinUNETR checkpoint as a clean fallback detector
initialization, but do not schedule it while
`spatialdino-pu-selective-distillation-v2` is active. If the active run misses
its unchanged selection gate, test a Biohub-owned one-channel detection head on
this backbone under the same movie split and selection/acceptance gates.

This is a materially different model family, not a public Biohub notebook or a
copy of public predictions.

## Primary sources and immutable evidence

- Paper: <https://arxiv.org/abs/2605.26026>
- Official repository: <https://github.com/AdinaScheinfeld/lsm_fm_public_repo>
- Audited repository commit:
  `cb772551652c3ee762048936e396743fe9a65e10`
- Official Zenodo model record: <https://doi.org/10.5281/zenodo.20146516>
- Zenodo record access: open
- Zenodo weight license: CC-BY-4.0
- Downloaded checkpoint: `swinunetr_image_only_best.ckpt`
- Size: `278085300` bytes
- Zenodo MD5: `eabe42d6a6755dcb8ac5f7b7d98c4ae5`
- Verified local SHA-256:
  `ef0f3d100f9a9aaa5d9a48bb9b07e7f1b0e0d1cc690cdcd308b1e0446bd01634`

The repository README displays an MIT badge but the pinned repository tree has
no `LICENSE` file. Therefore, a transfer experiment should use only the
CC-BY-4.0 checkpoint through our existing MONAI 1.5.1 implementation
(Apache-2.0), not copy the repository's finetuning code.

## Checkpoint inspection

The checkpoint was loaded with PyTorch's restricted weights-only unpickler after
explicitly allowlisting its MONAI metadata and NumPy dtype classes. It contains:

- image-only masked reconstruction plus teacher-distillation pretraining;
- all five listed light-sheet sources enabled;
- single-channel input;
- source patch size `96^3`, downsampled to isotropic `64^3`;
- feature size 24 and a 512-channel voxel output;
- 1,000 configured pretraining epochs;
- 325 state tensors and 34,254,054 state parameters total;
- 16,656,946 parameters in each of the student and EMA teacher encoders.

This is unusually well aligned with our existing Biohub PU pipeline, which also
normalizes each frame to a single-channel isotropic `64^3` volume. It avoids the
2D/MIP and channel-composition mismatch found in the Cell-DINO audit.

## Proposed clean experiment, only if needed

1. Instantiate MONAI 1.5.1 `SwinUNETR` with one input channel, feature size 24,
   and one output channel.
2. Load `student_encoder.*` weights except the pretrained 512-channel output
   layer; initialize a new one-channel heatmap head.
3. Reuse the existing organizer-positive and conservative two-teacher PU target
   construction. Do not copy public nodes or predictions.
4. Preserve the frozen validation movies, threshold source, and the `0.80`
   pooled / `0.65` worst-movie selection gates.
5. Open untouched acceptance only after both selection gates pass. Never use the
   public leaderboard to choose the model or threshold.

The fallback has a strong modality and geometry match, but it is not assumed to
be better merely because it is newer. Its value must be established by the same
complete-movie evidence used for SpatialDINO.
