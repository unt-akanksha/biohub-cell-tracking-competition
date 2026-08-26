# DynaCLR + SpatialDINO transfer audit

Date: 2026-08-26

Status: objective-level corroboration only; no checkpoint imported and no GPU run launched.

## Primary-source audit

[DynaCLR](https://arxiv.org/abs/2410.11281) learns temporally regularized
single-cell representations by making images of the same tracked cell at
nearby timepoints close in embedding space. The official
[VisCy repository](https://github.com/mehta-lab/VisCy) was audited at commit
`4b62365c0df25929bffc7f01b3bb2d1c11d69cce`. The repository is BSD-3-Clause;
the audited license SHA-256 is
`d86bab07df83b7dff1b4ff946f4014f2e64662ee322f586d35ff56ba580a2aa7`.

The current upstream repository has evolved beyond the paper's original
ConvNeXt setup. Its most relevant recipe freezes
`facebook/dinov3-convnext-tiny-pretrain-lvd1689m`, trains a
`768 -> 768 -> 128` MLP using NT-Xent at temperature `0.5`, and uses nearby
members of the same lineage as positives. The recipe also uses 2-GPU DDP,
large batches, per-timepoint normalization, and strong spatial/intensity
augmentation.

Reproducible source hashes:

| Source | SHA-256 |
|---|---|
| DINOv3 temporal training recipe | `54f4fb63e555465cbbd82f1bb0f02ad50ba884cd2bdd0fb22782303ab42f0c1c` |
| frozen-DINOv3 MLP model recipe | `70969fadebd286754e760061ebd3fff864ee9c582292e5074f2e8058c1df1495` |
| DINOv3 microscopy wrapper | `a6082e72e01e8fc87bc3fb1ca8d567209a51607b279e8284a2a3548c048a1cb0` |

## Biohub decision

Direct checkpoint transfer is rejected for the current iteration. The public
models were trained on phase/fluorescence single-cell patches and their recipe
reduces depth before a 2D global embedding. Biohub needs voxel-aware 3D center
evidence, candidate-specific association, and calibrated divisions; importing
the upstream output would introduce domain and task mismatch without proving a
clean gain.

The useful result is narrower and already represented in our independent
implementation:

- freeze a strong spatial foundation encoder initially;
- learn an explicit temporal projection on organizer-provided adjacent links;
- use augmentation-invariant contrastive supervision;
- restrict negatives to geometrically feasible competing cells;
- keep division as a separately calibrated sparse event.

`research/temporal_contrastive/` implements those task-specific principles
without copying VisCy source or weights. Unlike the upstream global-patch
objective, it fuses adjacent high-resolution feature maps and rejects a batch
when the candidate graph omits a ground-truth link.

## Evidence gate

No DynaCLR-derived experiment should run before the active SpatialDINO detector
and the repaired degree-preserving appearance validation are reconciled. If
SpatialDINO already separates hard association swaps, a new temporal model is
unnecessary. If it does not, the next controlled ablation is the Biohub-owned
temporal head versus frozen SpatialDINO features, not a DynaCLR submission.
