# Dense deformation correspondence: functionality experiment

The eight native temporal division fits failed cross-embryo qualification.
Rather than add capacity to 65 optimization division labels, test dense known
correspondences from invertible image transformations. This is a new supervision
mechanism, not another threshold sweep or a claim of recovered real divisions.

Start from the licensed primary LF-DCTTA checkpoint already audited locally,
SHA-256 `12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771`.
Freeze its detector and temporal U-Net; train only its existing cross-attention
linker. Do not import the newly published dense detector weights: their Kaggle
dataset metadata currently declares an unknown license. Do not use the new
all199 notebook: its entire inference is copying a precomputed submission CSV.

Use one predetermined middle frame of the first optimization movie in each
embryo in the fixed native v2 role plan. No GT positions, counts or edges enter
this smoke. Extract image peaks once using the original detector at 0.965.
Apply an analytically invertible triangular sinusoidal shear plus translation
to the image and coordinates. Exact transform identity supplies correspondence
labels even for unannotated image features; no unknown biological cell is
labeled as a negative. Permute target ordering; exclude targets outside a
two-voxel image margin. Source competitors remain present. No node-count target,
GT-coordinate modification, test adaptation, division synthesis or fake output.

First require CPU inverse/grid/label tests, then a <=10-minute Antelume smoke:
strict weight load, integer translation image/point alignment, finite forward
and backward, frozen encoder/detector hashes, nonzero linker gradients, checkpoint
reload, real image provenance and throughput. Twenty-five updates are a
functionality test on mixed sources; these weights are forbidden for selection
or submission. A full training recipe and separate real-data acceptance must
be frozen afterward before any deployment. Synthetic fit quality alone cannot
justify a tracking candidate; image warps do not reproduce biological division.

Motivation, not proof of Biohub gains:
[SuperGlue](https://arxiv.org/abs/1911.11763) learns matching and rejection through
attention; [Trackastra](https://arxiv.org/abs/2405.15700) learns cell associations
within temporal windows. The [FOCUS-3D discussion](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/738217)
reports augmented dense-pair training as a participant hypothesis. No code or
weights from those papers are copied here. Preserve the existing clean submission.

## Frozen full experiment (before real selection predictions)

Smoke passed in 7.46 seconds, 25 updates in 2.08 seconds. All 60 linker parameter
tensors received nonzero gradients, frozen detector/encoder hash was unchanged,
and reload was exact. The synthetic training fit changed 673/753 to 751/753;
this is not held-out tracking evidence. The smoke output is fully backed up.

Freeze two independent source-embryo adaptations from the original public
checkpoint, never from the mixed-source smoke. Use all 91 existing optimization
movies, one predetermined middle frame each, four seeded nonrigid views per
movie: 364 dense pairs. Translation uniform +/-3 isotropic voxels, triangular
shear amplitudes +/-2 voxels, target gain 0.85-1.15; source/target crop margin two
voxels. Analytic transform IDs supply labels, with random target order. The
detector and U-Net stay frozen; use full FP32 for reproducible cached features
and linker optimization. Peak count >1536 aborts rather than truncating nodes.

Each source fit receives 2,000 AdamW updates, LR 1e-5, weight decay .01, gradient
clip 1, and 1e-4 times SUM squared distance from the original linker parameters.
Seed 9731 for 44b6 and 9732 for 6bba. Uniform sampling of four-view movie packets,
final checkpoint only; no tuning, early stopping or extra seeds. Persist the
feature packets, initial provenance, optimizer, sampler/CPU/CUDA RNG and history.
Frozen detector/encoder bytes must remain exact.

Screen on four evenly spaced, already-assigned transitions in each of the ten
existing selection movies (three 44b6 and seven 6bba). This is only a real-pair
screen, not the complete-movie official-score promotion gate. One-to-one image
proposal/GT matching at 3.25 um establishes unambiguous known parent identities;
report eligible and all annotated edge counts. Original and adapted models use
identical cached features/candidates. The source gate requires at least one
additional correct parent, strictly lower cross entropy, and no movie losing a
correct parent. Only a source-passing final model opens the opposite embryo;
apply the identical gate, without changing any threshold or model choice.
The public initialization overlaps competition training, so these results are
adaptation-held-out diagnostics, NOT clean embryo-held-out model validation.

Run sequentially on Antelume, watchdog 1,800 seconds with a 60-second checkpoint
margin, foreign-GPU/RAM checks every 25 updates. Estimated extraction plus
training 8-15 minutes; reject/record any technical failure without using its
partial weights. Even two passing experts still require a fixed mixture test,
all eight complete-movie patched scores and production runtime acceptance.
