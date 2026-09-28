"""Physical flow at exact raw centroids, with explicit trailing-border extension.

Strided image reads cover coordinates 0,4,...,252 for a 256-pixel axis.
Raw centroids may legitimately lie in (252,255]. Extend the last flow value
through that trailing strip; never change a centroid or discard a node.
This is a separate declared adapter, not a change to native-node inference.
"""
import numpy as np


def sample_raw_centroid_flow(field, points, image_shape, downsample=(1, 4, 4)):
    field = np.asarray(field)
    points = np.asarray(points, dtype=np.float64)
    shape = np.asarray(image_shape)
    stride = np.asarray(downsample)
    if (shape.shape != (3,) or stride.shape != (3,)
        or not np.isfinite(shape).all() or not np.isfinite(stride).all()
        or (shape < 2).any() or (shape != np.floor(shape)).any()
        or (stride < 1).any() or (stride != np.floor(stride)).any()):
        raise ValueError('Integer positive image shape and strides required')
    expected = tuple(((shape.astype(int) - 1) // stride.astype(int) + 1).tolist())
    if (field.shape != (3, *expected) or not np.isfinite(field).all()
        or min(expected) < 2):
        raise ValueError('Finite ZYX-micron flow on the exact strided grid required')
    if (points.ndim != 2 or points.shape[1] != 3
        or not np.isfinite(points).all() or (points < 0).any()
        or (points > shape - 1).any()):
        raise ValueError('Finite centroids inside the original voxel-center bounds required')
    grid = points / stride
    last = np.asarray(expected) - 1
    extended = np.any(grid > last, axis=1)
    # Only the sample location is extended to the final available field value.
    # The caller retains the original centroid array unchanged.
    bounded = np.minimum(grid, last)
    lower = np.floor(bounded).astype(int)
    upper = np.minimum(lower + 1, last)
    fraction = bounded - lower
    sampled = np.zeros((len(points), 3), dtype=np.float64)
    for bits in range(8):
        indices = [upper[:, axis] if bits & (1 << axis) else lower[:, axis]
                   for axis in range(3)]
        weight = np.prod(np.stack([
            fraction[:, axis] if bits & (1 << axis) else 1 - fraction[:, axis]
            for axis in range(3)], axis=1), axis=1)
        sampled += field[:, indices[0], indices[1], indices[2]].T * weight[:, None]
    return sampled.astype(np.float32), dict(
        sampled_nodes=len(points), trailing_border_extended_nodes=int(extended.sum()),
        policy='constant flow extension through trailing unsampled voxel centers',
        coordinates_modified=False, nodes_deleted=False)
