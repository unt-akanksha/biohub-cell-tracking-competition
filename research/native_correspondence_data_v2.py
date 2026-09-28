"""Native-image association preparation with image-only candidate placement."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import gaussian_filter, maximum_filter

VOXEL = np.array([1.625, .40625, .40625], np.float32)
SCALES = (.8125, 1.625, 3.25)


def proposals(pooled):
    if pooled.shape != (64, 64, 64) or not np.isfinite(pooled).all():
        raise ValueError('Invalid pooled image')
    response = gaussian_filter(pooled, .7) - gaussian_filter(pooled, 1.5)
    points = np.argwhere((response == maximum_filter(response, 3)) & (response > .025)).astype(np.float32)
    if len(points) > 4096:
        raise ValueError('Proposal cap exceeded; never truncate')
    points[:, 1:] = points[:, 1:] * 4 + 1.5
    return points * VOXEL


def match_queries(truth_um, image_um):
    """Unique image proposals; dummy matches keep unmatched truth explicit."""
    if not len(truth_um) or not len(image_um):
        return {}
    distances = np.linalg.norm(truth_um[:, None] - image_um[None], axis=-1)
    cost = np.concatenate((np.where(distances <= 3.25, distances, 1e6),
                           np.full((len(truth_um), len(truth_um)), 3.2501)), axis=1)
    rows, cols = linear_sum_assignment(cost)
    return {int(r): int(c) for r, c in zip(rows, cols) if c < len(image_um) and distances[r, c] <= 3.25}


def groups(parent_um, child_um, parent_truth, child_truth):
    matched = match_queries(child_truth, child_um)
    records = []
    counts = dict(annotated_edges=len(child_truth), matched_queries=len(matched),
                  positive=0, null=0, ambiguous_omitted=0, eligible=0)
    for row, child in sorted(matched.items()):
        # Ground truth cannot influence the image-generated top-k candidate set.
        distances = np.linalg.norm(parent_um - child_um[child], axis=1)
        candidates = np.flatnonzero(distances <= 20.)
        candidates = candidates[np.argsort(distances[candidates], kind='stable')[:16]]
        true_distance = np.linalg.norm(parent_um[candidates] - parent_truth[row], axis=1)
        nearest = int(np.argmin(true_distance)) if len(candidates) else -1
        safe = true_distance > 7.
        if nearest >= 0 and true_distance[nearest] <= 3.25:
            target = nearest; safe[nearest] = True; counts['positive'] += 1
        elif np.any(true_distance <= 7.):
            counts['ambiguous_omitted'] += 1
            continue
        else:
            target = 16; counts['null'] += 1
        records.append((child, candidates, safe, target))
    counts['eligible'] = len(records)
    return records, counts


def pack_groups(records, parent_um, child_um):
    parents = sorted({int(p) for _, candidates, _, _ in records for p in candidates})
    children = sorted({int(child) for child, _, _, _ in records})
    pmap = {p: i for i, p in enumerate(parents)}
    cmap = {c: i + len(parents) for i, c in enumerate(children)}
    ids = np.zeros((len(records), 17), np.int64)
    valid = np.zeros_like(ids, bool); mask = np.zeros((len(records), 16), bool)
    target = np.empty(len(records), np.int64)
    coordinates = np.zeros((len(records), 17, 3), np.float32)
    for row, (child, candidates, safe, label) in enumerate(records):
        n = len(candidates)
        ids[row, :n+1] = [cmap[child]] + [pmap[int(p)] for p in candidates]
        valid[row, :n+1] = True; mask[row, :n] = safe; target[row] = label
        coordinates[row, :n+1] = np.concatenate((child_um[child:child+1], parent_um[candidates]))
    return dict(ids=ids, valid=valid, loss_mask=mask, targets=target, coords=coordinates), parent_um[parents], child_um[children]


def normalize_gpu(raw, torch):
    if raw.shape != (64, 256, 256) or raw.dtype != np.uint16:
        raise ValueError('Native uint16 image required')
    image = torch.as_tensor(raw.astype(np.float32), device='cuda')
    low, high = torch.quantile(image.flatten(), torch.tensor([.01, .995], device='cuda'))
    image = ((image-low)/(high-low).clamp_min(1.)).clamp(0., 2.)
    pooled = image.reshape(64, 64, 4, 64, 4).mean((2, 4))
    return image, pooled.cpu().numpy()


def patches_gpu(image, positions_um, torch):
    """One volume, bounded grids: no replication of the full native input."""
    result = []
    axis = torch.arange(-7, 8, device=image.device, dtype=torch.float32)
    base = torch.stack(torch.meshgrid(axis, axis, axis, indexing='ij'), -1)
    voxel = torch.tensor(VOXEL, device=image.device)
    extent = torch.tensor([63., 255., 255.], device=image.device)
    scales = torch.tensor(SCALES, device=image.device).view(1, 3, 1, 1, 1, 1)
    for start in range(0, len(positions_um), 64):
        points = torch.as_tensor(positions_um[start:start+64], device=image.device)
        grid = (points[:, None, None, None, None, :] + base * scales) / voxel
        grid = (2.*grid/extent - 1.).flip(-1)
        sampled = torch.nn.functional.grid_sample(image[None, None], grid.reshape(1, -1, 15, 15, 3),
                        mode='bilinear', padding_mode='border', align_corners=True)
        result.append(sampled.reshape(len(points), 3, 15, 15, 15).half().cpu().numpy())
    return np.concatenate(result) if result else np.empty((0, 3, 15, 15, 15), np.float16)
