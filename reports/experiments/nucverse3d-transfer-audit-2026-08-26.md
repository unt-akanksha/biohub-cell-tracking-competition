# NucVerse3D Biohub transfer audit — 2026-08-26

## Decision

Retain NucVerse3D as a genuinely independent, future detector/centroid prior,
but do not spend Kaggle GPU on it before the active LSM-FM center enhancer and
the staged public-node refinement gates complete. Its centroid-directed 3D
gradient representation is relevant to Biohub localization, but the released
weights are packaged inside a 7.93 GB archive and use a separate TensorFlow
2.16.2 inference stack. The near-term evidence-per-GPU-hour is therefore lower
than refining the existing independent LSM-FM probability field.

This decision is based on scientific fit and deployment cost, not public
leaderboard scores.

## Immutable primary-source evidence

- Paper: <https://www.nature.com/articles/s41598-026-51994-x>
- Official repository: <https://github.com/Segovia-lab/NucVerse3D>
- Audited repository commit:
  `d809a2e6cf380342708b7a9107574b259e6b34eb`
- Repository license: MIT
- Official Zenodo record: <https://doi.org/10.5281/zenodo.18517324>
- Zenodo record license: CC-BY-4.0
- `models.zip`: 7,929,759,188 bytes,
  MD5 `d7e69967c0077ea3030048b8d213d98`

The source tree was cloned only into the ignored research cache. No source,
weights, or predictions were copied into the Biohub implementation.

## Scientific fit

NucVerse3D is a 3D residual-attention U-Net that jointly predicts a foreground
mask and a three-axis gradient field pointing toward nuclear centroids. It was
trained across heterogeneous 3D data, including light-sheet microscopy,
C. elegans embryos, and a Zebrafish NucMM-Z benchmark. The generalized model
uses isotropic object-size normalization and the paper reports that the pooled
model remains competitive with dataset-specific models.

Relevant released settings are:

- one-channel input;
- `64 × 128 × 128` inference patches;
- base width 32;
- TensorFlow 2.16.2 with CUDA;
- generalized scaled model target radius 10 voxels;
- released Zebrafish NucMM-Z mean radius 7 voxels;
- probability-mask plus centroid-gradient outputs.

Biohub frames are single-channel `64 × 256 × 256` volumes with native voxel
spacing `(1.625, 0.40625, 0.40625)` µm. The input geometry is tractable, but a
Biohub radius must be estimated from images without consulting held-out labels,
and the modality/domain difference must be validated rather than assumed away.

## Evidence-gated future experiment

If the current LSM-FM coordinate lanes fail, a clean NucVerse3D probe may:

1. extract only the generalized scaled model from the official archive and
   verify its checksum and CC-BY-4.0 provenance;
2. estimate a global Biohub object scale from training-image autocorrelation or
   organizer-owned metadata, never from validation recall;
3. convert gradient-field attractors to point centroids while calibrating count
   only from the organizer node-count estimate;
4. use the same eight selection and four untouched acceptance movies;
5. require a strict matched-node gain with bounded per-movie regression before
   any topology integration.

No competition submission should be produced by the probe.
