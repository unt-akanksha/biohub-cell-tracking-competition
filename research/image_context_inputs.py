"""Shared training/inference image-context adapter; no neighbor-graph features."""
import math
import numpy as np
from research.image_division_context import image_context

SCALE = np.asarray((1.625, .40625, .40625), dtype=np.float32)


def anchor_tokens(nodes, parent_id, first_child_id, second_child_id):
    ids = (int(parent_id), int(first_child_id), int(second_child_id))
    if len(set(ids)) != 3:
        raise ValueError('Three distinct anchors required')
    # Deliberately access only the three anchors, never other graph nodes/edges.
    rows = [nodes[i] for i in ids]
    positions = [np.asarray([r[k] for k in ('z', 'y', 'x')], dtype=np.float32) * SCALE for r in rows]
    if any(r['t'] != rows[0]['t']+1 for r in rows[1:]) or not np.isfinite(positions).all():
        raise ValueError('Invalid anchor time/position')
    result = np.zeros((3, 8), dtype=np.float32)
    result[0, 5] = 1.
    daughters = sorted(zip(positions[1:], ids[1:]), key=lambda p: (tuple(p[0]), p[1]))
    for index, (position, _) in enumerate(daughters, 1):
        delta = position - positions[0]
        distance = float(np.linalg.norm(delta))
        result[index] = (.5, *np.clip(delta/20., -2., 2.), math.log1p(distance)/math.log1p(30.), 0., 1., 0.)
    return result


def contexts_for_patches(patches, anchors):
    # The training archive is float16 after physical sampling. Match that
    # quantization before computing nonlinear morphology at inference as well.
    values = np.asarray(patches, dtype=np.float16)
    anchors = np.asarray(anchors, dtype=np.float32)
    if values.ndim != 6 or values.shape[1:] != (3, 3, 17, 17, 17):
        raise ValueError('Expected N x three centers x image triplets')
    if anchors.shape != (len(values), 3, 8):
        raise ValueError('Candidate anchor batch misaligned')
    rows = [image_context(p[0], a) for p, a in zip(values, anchors)]
    if not rows:
        return np.empty((0, 43, 8), np.float16), np.empty((0, 43), bool)
    return np.stack([r[0] for r in rows]).astype(np.float16), np.stack([r[1] for r in rows])
