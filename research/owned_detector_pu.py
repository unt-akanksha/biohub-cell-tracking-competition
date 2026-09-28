"""Sparse detector targets for two inverse-aligned views of OUR frozen model.

Target construction only. The future collector must enforce checkpoint/split
provenance. View agreement is a pseudo-label, not independent teacher evidence.
"""
import numpy as np
from spotiflow_biohub.pu_targets import build_pu_targets,weighted_pu_bce_with_logits


def contract():
    return dict(version=1,high_threshold=.96875,low_support_threshold=.10,
        consensus_radius_um=2.5,annotation_merge_radius_um=2.5,positive_sigma_voxels=1.,
        positive_weight_floor=.05,support_dilation_voxels=2,background_weight=.01,
        voxel_size_um=[1.625]*3,teachers='Two inverse-aligned views of the same owned source-trained detector',
        uncertainty='View support is unknown; not a negative label',
        graph_count_or_selection_score_used=False)


def teacher_probabilities(logits):
    """Convert AMP logits BEFORE sigmoid, avoiding artificial FP16 plateaus."""
    import torch
    if not torch.isfinite(logits).all():
        raise ValueError('Finite teacher logits required')
    return logits.float().sigmoid()


def targets(native,aligned,annotations):
    config=contract()
    native=np.asarray(native); aligned=np.asarray(aligned)
    points=np.asarray(annotations,dtype=np.float32)
    if points.size==0: points=points.reshape(0,3)
    if (native.ndim!=3 or aligned.shape!=native.shape or points.ndim!=2 or points.shape[1]!=3
        or not np.isfinite(points).all() or np.any(points<0) or np.any(points>np.asarray(native.shape)-1)):
        raise ValueError('Aligned volumes and in-grid annotated centers required; no clipping')
    built=build_pu_targets(native,aligned,points,
        high_threshold=config['high_threshold'],low_support_threshold=config['low_support_threshold'],
        consensus_radius=config['consensus_radius_um'],annotation_merge_radius=config['annotation_merge_radius_um'],
        positive_sigma=config['positive_sigma_voxels'],positive_weight_floor=config['positive_weight_floor'],
        support_dilation_voxels=config['support_dilation_voxels'],background_weight=config['background_weight'],
        voxel_size=config['voxel_size_um'])
    # Gaussian tails below the positive mask must NOT be misclassified by the
    # shared loss's `target>0` predicate as positive samples with background
    # weights. Preserve exactly the builder's three-region mask semantics.
    heatmap=np.where(built.positive_mask,built.heatmap,0.).astype(np.float32)
    if (np.any(built.weights[built.unknown_mask]!=0)
        or np.any(heatmap[built.safe_background_mask]!=0)):
        raise ValueError('Positive/unlabeled/background region semantics violated')
    return dict(heatmap=heatmap,weights=built.weights,positive_mask=built.positive_mask,
        background_mask=built.safe_background_mask,unknown_mask=built.unknown_mask,
        positive_coords=built.positive_coords,consensus_count=built.consensus_count,
        forced_annotation_count=built.forced_annotation_count)


def loss(logits,target,weights):
    import torch
    if (not torch.isfinite(logits).all() or not torch.isfinite(target).all()
        or not torch.isfinite(weights).all() or (target<0).any() or (target>1).any()
        or (weights<0).any() or (weights>1).any()):
        raise ValueError('Finite logits and bounded targets/weights required')
    return weighted_pu_bce_with_logits(logits,target,weights)
