# Synthetic pretraining source audit

Source kernel: `josefreitasalvesneto/biohub-synthetic-dataset`  
Discussion: <https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/732103>  
Declared license: CC0

The completed kernel output is 18.503 GiB and its downloaded manifest reports:

- 1,539 fully labeled static native volumes, each `64×256×256`;
- 2,174 six-frame pooled sequences;
- 4,056,226 sequence nodes and 165,267 labeled divisions;
- native voxel scale `(1.625, 0.40625, 0.40625)` µm;
- motion calibrated to median 1.86 µm/frame, lag-1 persistence 0.30, and
  7.24 µm sister separation.

This is useful, but it is not plug-and-play.

## Required repairs

1. Static samples are internally consistent: `volume` is native
   `64×256×256`, `centroids` are native `(z,y,x)`, and `voxel_um` is native.
   Downsample the image with `volume[:, ::4, ::4]` and divide centroid Y/X by
   four for an isotropic `64³` detector target.
2. Sequence `volumes` are already pooled to `(T,64,64,64)`, but `nodes` store
   native Y/X positions up to 255. Divide sequence node Y/X by four exactly
   once. The supplied `voxel_um_pooled=(1.625,1.625,1.625)` otherwise conflicts
   with the raw node coordinates.
3. The synthetic division rate is 4.074%, versus about 0.26% in the real labels.
   Division loss/prior must be multiplied by roughly `0.0026 / 0.04074 ≈ 0.064`
   for calibrated fine-tuning, while oversampled mini-batches may still be used
   for representation learning.
4. Daughter nodes share the parent's synthetic `tid`; use the explicit `edges`
   and `divisions` arrays as lineage truth rather than the `tid` column.
5. Nucleus texture/contrast and embryo-to-embryo variation are acknowledged
   domain gaps. Use this source for pretraining, followed by real-data
   positive-unlabeled adaptation; do not validate only on synthetic samples.

## Candidate training design

- Detector: initialize official Spotiflow `synth_3d` or the better acceptance
  arm, train on corrected static centroids with dense heatmap/flow supervision,
  then adapt to Biohub using forced sparse GT positives plus high-confidence
  teacher pseudo-positives and a low-weight unlabeled consistency loss.
- Association/division: pretrain Trackastra-style four-to-six-frame graph
  windows from corrected sequence nodes/edges, oversample true division windows,
  and calibrate the division head/prior on real Biohub annotations.
- Validation: keep the eight selection and four acceptance fields out of all
  real-data adaptation. Report detector availability and conditional link
  accuracy separately before assembling a candidate graph.

The source is higher-upside than further public post-processing, but its large
volume should be streamed directly as an attached Kaggle kernel output rather
than copied into our private runtime.

## Implemented association stage (2026-08-26)

The Trackastra trainer now reads only the graph members of 384 deterministic
six-frame NPZ sequences; it does not decompress their unused image arrays. Node
Y/X is divided by four exactly once and the resulting pooled coordinates are
converted by the same physical scale as real Biohub nodes. The schedule applies
1,200 synthetic representation-learning steps before 5,000 real Biohub graph
steps. Synthetic division loss is multiplied by the audited prior ratio
(`~0.064`) and division-window preference is limited to 0.15 before the real
stage restores its native supervision. Initial, post-synthetic, and post-real
validation are all recorded, followed by the existing four complete clean
movies and official-formula proxy grid.
