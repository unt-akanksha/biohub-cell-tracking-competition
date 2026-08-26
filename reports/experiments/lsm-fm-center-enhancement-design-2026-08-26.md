# LSM-FM center-enhancement design — 2026-08-26

## Decision

Stage an independently implemented learned residual center enhancer as the next
detector experiment if the inference-only refinement run does not cross its
frozen worst-movie gate. Do not import CELLECT implementation code or weights.

The frozen feature-36 LSM-FM detector already projects the limiting movie's
node count almost exactly. A global radius-2 probability-squared centroid
improved its annotated-node recall from `1,065 / 1,659` (`0.6419529837`) to
`1,078 / 1,659` (`0.6497890295`), exactly one match below the `0.65` gate, while
improving every selection movie. This makes a learned count-preserving residual
correction more targeted than another density threshold, similar heatmap
ensemble, or larger association model.

## Research basis

- CELLECT (Nature Methods, 2025) uses learned center enhancement and
  intra-frame redundancy reasoning for sparse-annotation 3D cell tracking:
  https://www.nature.com/articles/s41592-025-02886-x
- Official CELLECT code is GPL-2.0 and its released weights do not state a
  separate license. The audited repository and weights therefore remain
  isolated; this experiment uses only the paper-level idea.
- LSM-FM supplies the frozen 35,072,515-parameter microscopy detector and
  remains attributed under CC-BY-4.0:
  https://arxiv.org/abs/2605.26026

## Candidate

The new head is a compact residual 3D CNN over `7×7×7` patches with two input
channels: locally normalized raw intensity and the frozen feature-36
probability map. A spatial-softmax center volume produces a bounded sub-voxel
offset. Training uses dense Gaussian center cross-entropy plus robust offset
regression.

The target is a residual from the globally stronger feature-36 centroid
(`radius=2`, probability-squared weighting), not from the integer local maximum.
This keeps the control numerically comparable with the near-gate evidence.

## Sparse-label and selection rules

- Exclude all eight selection and four acceptance movies from training.
- Match frozen peaks to available annotations one-to-one in physical units.
- Train only on accepted matched offsets. Unmatched peaks remain unknown and
  are never treated as background or false positives.
- Preserve peak identities, confidence values, confidence ranking, and density
  calibration.
- Predeclare only three global candidates: unchanged control, half learned
  residual, and full learned residual. No per-movie choice is permitted.
- Require pooled selection recall at least `0.80`, worst-movie recall at least
  `0.65`, and no movie more than `0.01` below control before opening acceptance.
- Do not use leaderboard feedback and do not create a competition submission.

## Promotion and resource boundary

Acceptance remains sealed unless selection passes. Promotion additionally
requires the existing public clean-reproduction comparator gates. The staged
training/validation notebook is a one-GPU experiment; any later competition
submission must independently enforce exactly two visible GPUs and whole-movie
sharding. The experiment is not authorized to submit.
