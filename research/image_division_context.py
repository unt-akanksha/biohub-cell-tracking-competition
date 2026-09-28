"""Division context derived from images, never from sparse annotation neighbors.

Input patches use the existing 17-cubed, one-micron grid and per-channel
standardization. Half-max volumes are morphology descriptors on that processed
grid, NOT calibrated raw nuclear volumes or conserved fluorescence measurements.
"""
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter, label

FAMILY = 'image_peak_volume_division_context_v1'
TOKEN_SHAPE = (43, 8)


def image_context(parent_temporal_patch, anchor_tokens):
    patch = np.asarray(parent_temporal_patch, dtype=np.float64)
    anchors = np.asarray(anchor_tokens, dtype=np.float32)
    if patch.shape != (3, 17, 17, 17) or anchors.shape != (3, 8):
        raise ValueError('Require image triplet and exactly three anchor tokens')
    if not np.isfinite(patch).all() or not np.isfinite(anchors).all():
        raise ValueError('Context inputs must be finite')
    if (int((anchors[:, 5] > .5).sum()) != 1 or
            int((anchors[:, 6] > .5).sum()) != 2):
        raise ValueError('Exactly one parent and two daughter anchors required')
    tokens = np.zeros(TOKEN_SHAPE, dtype=np.float32)
    mask = np.zeros(TOKEN_SHAPE[0], dtype=bool)
    tokens[:3] = anchors
    mask[:3] = True
    for channel, frame in enumerate(patch):
        smooth = gaussian_filter(frame, sigma=1., mode='reflect')
        background = float(np.quantile(smooth, .1))
        high = float(np.quantile(smooth, .99))
        amplitude = high - background
        if amplitude <= np.finfo(float).eps * max(1., abs(high), abs(background)):
            continue
        maxima = (smooth == maximum_filter(smooth, size=3, mode='reflect')) & (smooth > background)
        components, count = label(maxima, structure=np.ones((3, 3, 3)))
        peaks = []
        for component in range(1, count + 1):
            indices = np.argwhere(components == component)
            position = indices.mean(axis=0)
            index = indices[0]
            peak = float(smooth[tuple(index)])
            peaks.append((peak, tuple(position), index))
        peaks.sort(key=lambda x: (-x[0], x[1]))
        accepted = []
        for peak, position, index in peaks:
            xyz = np.asarray(position)
            if any(np.linalg.norm(xyz - old) < 3. for old in accepted):
                continue
            bounds = [slice(max(0, int(v) - 4), min(17, int(v) + 5)) for v in index]
            box = smooth[tuple(bounds)]
            above = box >= background + .5 * (peak - background)
            regions, _ = label(above, structure=np.ones((3, 3, 3)))
            local_index = tuple(int(index[d]) - bounds[d].start for d in range(3))
            region = int(regions[local_index])
            if not region:
                raise ValueError('Peak absent from its own half-max component')
            volume = int(np.count_nonzero(regions == region))
            slot = 3 + channel * 8 + len(accepted)
            tokens[slot] = ((channel - 1) / 2., *((xyz - 8.) / 20.),
                np.log1p(volume) / np.log1p(17**3), 0., 0.,
                np.clip((peak - background) / amplitude, 0., 4.))
            mask[slot] = True
            accepted.append(xyz)
            if len(accepted) == 8:
                break
    if not np.isfinite(tokens).all():
        raise ValueError('Nonfinite image context')
    return tokens, mask
