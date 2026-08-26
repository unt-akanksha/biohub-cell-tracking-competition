# trackastra-graph-finetune-v5

Status: failed before the first optimizer step after 116 seconds; superseded by
V6 with the CUDA autocast loss fix. No checkpoint or submission was produced.

V5 supersedes the unlaunched V4 before any GPU was spent. The change addresses
a specific association-model distribution mismatch: the complete validation
movies contain 6,050 to 44,958 detector nodes, while supervised lineage graphs
are sparse and the annotations are positive-unlabeled. A tracker trained only
on clean graph windows can therefore fail when inference tiles are dominated by
unrelated detections.

Each 512-token real-training window now reserves a 128-token budget for true
lineage nodes and samples up to three artificial distractors per true node.
Sixty percent are hard local negatives and forty percent cover the full spatial
tile. Distractors remain at least four scaled voxels from known positive nodes,
never receive positive association or division targets, and are still eligible
as hard negative pairs inside the 64-voxel association radius. Synthetic graph
pretraining uses a milder 0.5 ratio so its topology signal is not drowned out.

The public 0.927 output remains only a frozen same-node comparator. One complete
movie per embryo selects the association method and thresholds, the other movie
per embryo is read only after freeze, and promotion still requires a positive
acceptance delta with no material worst-movie regression. The later submission
kernel hash-binds the accepted checkpoint and refuses both byte-identical and
edge-identical public replicas.

A full-capacity local smoke used the official 27.5M checkpoint with 126 true
nodes, 386 distractors, and 92 retained positive edges. Forward and backward
completed with finite loss and gradients, establishing that the new density
regime fits the exact 512-token training path before Kaggle GPU launch.

While V5 trains, a hash-bound audit of the owned baseline kernel output found
that stable edge IDs retain the original public link confidence for 92.85% to
97.63% of the final CSV edges. A probability-aware hybrid path is now tested in
source, but it remains disabled for submission until the same selection/frozen
acceptance protocol validates it; the running V5 artifact is unchanged.

## Terminal result

Kaggle loaded all 195 real graphs and 384 synthetic graphs successfully, then
failed during the initial clean window validation. Trackastra exposes normalized
probabilities, and PyTorch refuses probability-space binary cross entropy while
CUDA autocast is active. V6 retains the same scientific configuration but runs
that numerically sensitive BCE region explicitly in float32 outside autocast.
The failure consumed 0.0322 GPU hours and left 29.12 hours available.
