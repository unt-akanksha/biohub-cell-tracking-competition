"""Image-derived proposals and sparse-label-safe backward correspondence data.

Training labels may supervise the unique parent of an annotated child. They
never classify an unannotated child or an ambiguous nearby parent as negative.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter, map_coordinates
from scipy.spatial import cKDTree

VOXEL_UM = 1.625
PATCH_SIZE = 15
MAX_CANDIDATES = 16
RADIUS_UM = 20.0
POSITIVE_UM = 3.25
AMBIGUOUS_UM = 7.0
EXCLUDED = {
    '44b6_24264f12', '44b6_81c256f0', '6bba_23af9eeb', '6bba_f1fde7e0',
    '44b6_12dfb391', '44b6_267148e4', '6bba_062c8d37', '6bba_07e24132',
    '44b6_0113de3b', '44b6_0b24845f', '6bba_05b6850b', '6bba_05db0fb1',
}


def normalized_image(volume):
    image = np.asarray(volume, np.float32)
    if image.shape != (64, 64, 64) or not np.isfinite(image).all():
        raise ValueError('Finite, isotropically pooled 64-cube required')
    lo, hi = np.quantile(image, [0.01, 0.995])
    return np.clip((image - lo) / max(float(hi - lo), 1.0), 0, 2).astype(np.float32)


def image_proposals(image):
    response = gaussian_filter(image, .7) - gaussian_filter(image, 1.5)
    peaks = (response == maximum_filter(response, size=3)) & (response > .025)
    coords = np.argwhere(peaks).astype(np.float32)
    if len(coords) > 4096:
        raise ValueError('Proposal memory cap exceeded; do not truncate detections')
    return coords


def nearest_candidates(parent_coords, child_coord):
    distance = np.linalg.norm((parent_coords - child_coord) * VOXEL_UM, axis=1)
    eligible = np.flatnonzero(distance <= RADIUS_UM)
    return eligible[np.argsort(distance[eligible], kind='stable')[:MAX_CANDIDATES]]


def supervised_parent(candidates, true_parent):
    """Return chosen row (-1=null), safe loss mask, or None if ambiguous.

    Candidate generation must occur before this function: truth must never be
    inserted into the top-k set. A missing but nearby parent is not a null label.
    """
    distance = np.linalg.norm((candidates - true_parent) * VOXEL_UM, axis=1)
    mask = distance > AMBIGUOUS_UM
    chosen = int(np.argmin(distance)) if len(distance) else -1
    if chosen >= 0 and distance[chosen] <= POSITIVE_UM:
        mask[chosen] = True
        return chosen, mask
    if np.any(distance <= AMBIGUOUS_UM):
        return None
    return -1, np.ones(len(candidates), bool)


def patches_at(image, positions):
    axis = np.arange(PATCH_SIZE, dtype=np.float32) - (PATCH_SIZE - 1) / 2
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'))
    result = np.empty((len(positions), 2, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE), np.float16)
    for row, point in enumerate(positions):
        for channel, scale in enumerate((1.0, 2.0)):
            result[row, channel] = map_coordinates(
                image, grid * scale + point[:, None, None, None],
                order=1, mode='nearest', prefilter=False)
    return result


def validate_roles(records):
    seen = {}
    for record in records:
        stem, role = record['stem'], record['role']
        if stem in EXCLUDED or role not in ('optimization', 'selection'):
            raise ValueError('Excluded or sealed movie entered fitting bundle')
        if stem in seen and seen[stem] != role:
            raise ValueError('Movie leakage between optimization and selection')
        seen[stem] = role
    return seen
