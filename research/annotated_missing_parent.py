"""Null supervision only for annotated children with absent known parents.

A failed greedy GT match alone is insufficient: every source detection must
be farther than seven physical microns from the annotated parent's position.
Unannotated child columns and annotated births stay unsupervised.
"""
import numpy as np

SCALE = np.asarray([1.625, .40625, .40625])


def missing_parent_columns(gt_edges, child_matches, source_detections, source_truth, radius_um=7.):
    gt = np.asarray(gt_edges)
    matches = np.asarray(child_matches)
    detections = np.asarray(source_detections, dtype=float).reshape(-1, 3)
    truth = np.asarray(source_truth, dtype=float).reshape(-1, 3)
    if (gt.ndim != 2 or len(truth) != gt.shape[0] or matches.ndim != 1
            or not np.isfinite(gt).all() or not ((gt == 0) | (gt == 1)).all()
            or np.any(gt.sum(0) > 1) or not np.isfinite(matches).all()
            or np.any(matches != np.floor(matches)) or np.any(matches < -1)
            or np.any(matches >= gt.shape[1]) or not np.isfinite(detections).all()
            or not np.isfinite(truth).all() or not np.isfinite(radius_um) or radius_um <= 0):
        raise ValueError('Finite valid sparse GT edges, matches and coordinates required')
    result = np.zeros(len(matches), dtype=bool)
    for column, child in enumerate(matches.astype(int)):
        if child < 0:
            continue
        parents = np.flatnonzero(gt[:, child])
        if len(parents) != 1:
            continue
        delta = (detections - truth[parents[0]]) * SCALE
        result[column] = len(detections) == 0 or np.all(np.linalg.norm(delta, axis=1) > radius_um)
    return result


def batch_missing_parent_masks(source, target, gt_edges, gt_source_coords, ds_scale):
    masks = []
    for b in range(gt_edges.shape[0]):
        count = int(source[2][b].sum())
        masks.append(missing_parent_columns(
            gt_edges[b].detach().cpu().numpy(), target[3][b].detach().cpu().numpy(),
            (source[0][b, :count] * ds_scale).detach().cpu().numpy(),
            gt_source_coords[b].detach().cpu().numpy()))
    return masks


def patch_epoch(source):
    needle = 'frame_det[i][2], frame_det[i + 1][2],\n            ))'
    if source.count(needle) != 1:
        raise ValueError('Organizer batch-loss call changed')
    return source.replace(needle, '''frame_det[i][2], frame_det[i + 1][2],
                known_null=batch_missing_parent_masks(frame_det[i], frame_det[i + 1],
                    targets[:, i], coords[:, i] * ds_scale, ds_scale),
            ))''')
