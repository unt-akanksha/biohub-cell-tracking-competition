"""Invertible, image-space 3D correspondence supervision without GT negatives."""
import numpy as np


def forward_points(points, parameters, size=64):
    """Triangular volume-preserving shear; coordinates are z,y,x voxels."""
    p = np.asarray(points, dtype=np.float64)
    dz, dy, dx, ay, ax = np.asarray(parameters, dtype=np.float64)
    z = p[..., 0] + dz
    y = p[..., 1] + dy + ay * np.sin(2 * np.pi * z / size)
    x = p[..., 2] + dx + ax * np.sin(2 * np.pi * y / size)
    return np.stack((z, y, x), -1)


def inverse_points(points, parameters, size=64):
    p = np.asarray(points, dtype=np.float64)
    dz, dy, dx, ay, ax = np.asarray(parameters, dtype=np.float64)
    x = p[..., 2] - dx - ax * np.sin(2 * np.pi * p[..., 1] / size)
    y = p[..., 1] - dy - ay * np.sin(2 * np.pi * p[..., 0] / size)
    z = p[..., 0] - dz
    return np.stack((z, y, x), -1)


def inverse_grid(parameters, size=64):
    axis = np.arange(size, dtype=np.float64)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), -1)
    source = inverse_points(grid, parameters, size)
    return (2 * source[..., ::-1] / (size - 1) - 1).astype(np.float32).copy()


def known_pairs(source_points, parameters, permutation, margin=2, size=64):
    """Use transform identities, never nearest-neighbor pseudo-labels."""
    source = np.asarray(source_points, np.float32)
    order = np.asarray(permutation, np.int64)
    if source.ndim != 2 or source.shape[1] != 3 or not np.isfinite(source).all():
        raise ValueError('Finite N by 3 source points required')
    if not np.array_equal(np.sort(order), np.arange(len(source))):
        raise ValueError('Target ordering must be a permutation')
    target = forward_points(source, parameters, size)
    valid = ((target >= margin) & (target <= size-1-margin)).all(-1)
    ids = order[valid[order]]
    return target[ids].astype(np.float32), ids
