# Backward image-motion hypothesis — not a trained candidate

The scan of research, scripts and experiment reports found no matching
completed dense-flow experiment under the searched optical-flow, displacement-
field, photometric and VoxelMorph terms. This is a scoped search, not proof
that no earlier notebook ever explored motion.

## Primary-source findings

[ELEPHANT](https://elifesciences.org/articles/69380) estimates backward 3D
displacements with an approximately5.9M-parameter network. Its sparse flow
supervision is combined with image-similarity and smoothness losses, and flow
is sampled to estimate previous-frame nucleus positions. Its linking and
interpolation heuristics are not being adopted. This is evidence for a
modeling approach, not a Biohub performance claim.

[Amat, Myers and Keller](https://www.janelia.org/publication/fast-and-robust-optical-flow-time-lapse-microscopy-using-super-voxels)
studied optical flow specifically for large microscopy volumes, including
neighboring-cell motion and difficult divisions. This supports testing image
motion rather than assuming a coordinate-only prior is sufficient.

[VoxelMorph](https://github.com/voxelmorph/voxelmorph) provides generic learned
registration; its current repository lists Apache2.0 and warns that the PyTorch
interface is changing. No package, pretrained medical model or external data
was installed or used. Generic registration is not evidence of cell-tracking
accuracy. In particular, we should not impose invertibility on mitosis.

The [competition discussion](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/723655)
also suggests optical flow and self-supervision. This is participant advice,
not a validated solution or a leaderboard-based selection criterion.

## Proposed independent test

Predict a physical displacement at each current-frame child toward its
previous-frame parent. Two daughters can point toward one parent without
asking a single forward displacement vector to represent two destinations.
Keep detections unchanged. Train only on the frozen source-training movies;
unannotated cells must not become negative labels. Preserve actual null and
division cases, and do not require a one-to-one deformation.

Before a real fit: test warp direction, axis order, physical/downsample units,
point sampling, gradients, boundaries, and two-daughter convergence on tiny
synthetic inputs. Then a small real-training-data fit must demonstrate motion
learning before full-movie inference and the pinned official scorer. Existing
causal motion and b642 remain controls. No target-audit data has been opened,
no optical-flow training has launched, and no submission is authorized.
