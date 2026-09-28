"""Owned view-consensus PU targets with peak ordering in raw-logit space.

Like official inference, max-pool BEFORE sigmoid. Probability saturation must
not fabricate local maxima. No graph count or validation score enters labels.
"""
import numpy as np
from scipy.ndimage import maximum_filter
from scipy.spatial import cKDTree
from scipy.special import expit
from owned_detector_pu import contract as probability_contract
from spotiflow_biohub.pu_targets import PeakSet,match_teacher_peaks,add_forced_annotations,points_to_gaussian_heatmap


def contract():
    return dict(probability_contract(),version=2,peak_order='raw_logits_before_sigmoid',
        high_logit=float(np.log(.96875/(1-.96875))),low_logit=float(np.log(.1/(1-.1))))


def volume(values):
    values=np.asarray(values,dtype=np.float32)
    if values.ndim!=3 or not np.isfinite(values).all():
        raise ValueError('Finite3D raw detector logits required')
    return values


def peaks(logits):
    logits=volume(logits)
    maxima=maximum_filter(logits,size=3,mode='constant',cval=-np.inf)
    candidates=np.argwhere((logits>=contract()['high_logit'])&(logits==maxima))
    if not len(candidates):
        return PeakSet(np.empty((0,3),np.float32),np.empty(0,np.float32))
    values=logits[tuple(candidates.T)]
    order=sorted(range(len(candidates)),key=lambda i:(-float(values[i]),*map(int,candidates[i])))
    tree=cKDTree(candidates); suppressed=set(); kept=[]
    for index in order:
        if index in suppressed: continue
        kept.append(index); suppressed.update(tree.query_ball_point(candidates[index],r=1.))
    chosen=np.asarray(kept,dtype=np.int64)
    return PeakSet(candidates[chosen].astype(np.float32),expit(values[chosen].astype(np.float64)).astype(np.float32))


def targets(native,aligned,annotations):
    native,aligned=volume(native),volume(aligned)
    points=np.asarray(annotations,dtype=np.float32)
    if points.size==0: points=points.reshape(0,3)
    if (aligned.shape!=native.shape or points.ndim!=2 or points.shape[1]!=3
        or not np.isfinite(points).all() or np.any(points<0) or np.any(points>np.asarray(native.shape)-1)):
        raise ValueError('Aligned raw volumes and in-grid annotations required')
    config=contract()
    consensus=match_teacher_peaks(peaks(native),peaks(aligned),match_radius=config['consensus_radius_um'],
        voxel_size=config['voxel_size_um'])
    positive_coords,forced=add_forced_annotations(consensus,points,merge_radius=config['annotation_merge_radius_um'],
        voxel_size=config['voxel_size_um'])
    heatmap=points_to_gaussian_heatmap(positive_coords,native.shape,sigma=config['positive_sigma_voxels'])
    positive=heatmap>=config['positive_weight_floor']
    support=(native>=config['low_logit'])|(aligned>=config['low_logit'])
    support=maximum_filter(support,size=2*config['support_dilation_voxels']+1,mode='constant',cval=False)
    background=~support&~positive
    unknown=~(positive|background)
    weights=np.zeros(native.shape,np.float32)
    weights[positive]=heatmap[positive]; weights[background]=config['background_weight']
    return dict(heatmap=np.where(positive,heatmap,0.).astype(np.float32),weights=weights,
        positive_mask=positive,background_mask=background,unknown_mask=unknown,
        positive_coords=positive_coords,consensus_count=len(consensus),forced_annotation_count=forced)
