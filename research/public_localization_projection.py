"""Bound public post-processing drift using genuine detector coordinates.

No labels, image IDs, node-count targets, candidate insertion, deletion, or
topology changes. The fixed radius is one downsampled detector voxel (1.625um).
Only already-retained original detections are eligible; inserted gap nodes are
left untouched because they do not have an original detector reference.
"""
from __future__ import annotations
import copy
import numpy as np

SCALE = np.asarray((1.625, .40625, .40625), dtype=np.float64)
RADIUS_UM = 1.625


def project_offsets(offsets):
    """Symmetric integer ray projection into a physical one-voxel ball."""
    offsets = np.asarray(offsets)
    if (offsets.ndim != 2 or offsets.shape[1] != 3 or not np.isfinite(offsets).all()
            or not np.equal(offsets, np.rint(offsets)).all()):
        raise ValueError('Finite integer ZYX offsets required')
    offsets = offsets.astype(np.int64)
    squared = np.sum((offsets * SCALE)**2, axis=1)
    selected = squared > RADIUS_UM**2
    result = offsets.copy()
    values = offsets[selected]
    if not len(values):
        return result
    lo, hi = np.zeros(len(values)), np.ones(len(values))
    for _ in range(48):
        mid = (lo + hi) / 2
        proposal = np.rint(mid[:, None] * values)
        accepted = np.sum((proposal * SCALE)**2, axis=1) <= RADIUS_UM**2
        lo = np.where(accepted, mid, lo)
        hi = np.where(accepted, hi, mid)
    result[selected] = np.rint(lo[:, None] * values).astype(np.int64)
    if np.any(np.sum((result * SCALE)**2, axis=1) > RADIUS_UM**2):
        raise AssertionError('Projection escaped fixed physical bound')
    return result


def project_graph(final, detector_reference):
    """Use positions from the pre-repair ILP graph, keyed by stable node IDs."""
    result = copy.deepcopy(final)
    reference = detector_reference['nodes']
    ids = [i for i in final['nodes'] if i in reference]
    origins, targets = [], []
    for ident in ids:
        a, b = reference[ident], final['nodes'][ident]
        if a['node_id'] != b['node_id'] or a['t'] != b['t']:
            raise ValueError('Detector reference identity/time drift')
        origins.append([a[k] for k in ('z','y','x')])
        targets.append([b[k] for k in ('z','y','x')])
    if not ids:
        raise ValueError('No retained genuine detections')
    origins, targets = np.asarray(origins), np.asarray(targets)
    if not np.isfinite(origins).all() or not np.equal(origins, np.rint(origins)).all():
        raise ValueError('Integer genuine detector reference required')
    projected = origins.astype(np.int64) + project_offsets(targets - origins)
    changed = 0
    for ident, before, position in zip(ids, targets, projected):
        changed += int(np.any(before != position))
        for axis, value, bound in zip(('z','y','x'), position, (64,256,256)):
            if not 0 <= value < bound:
                raise ValueError('Projected coordinate outside movie')
            result['nodes'][ident][axis] = int(value)
    if result['edges'] != final['edges'] or result['nodes'].keys() != final['nodes'].keys():
        raise AssertionError('Projection changed topology or membership')
    return result, dict(radius_um=RADIUS_UM, eligible_nodes=len(ids), changed_nodes=changed,
                        inserted_nodes_untouched=len(final['nodes'])-len(ids),
                        node_ids_unchanged=True, edges_unchanged=True,
                        ground_truth_used=False, authorized_for_submission=False)
