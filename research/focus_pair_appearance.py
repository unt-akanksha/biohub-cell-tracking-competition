"""Complete pair appearance descriptors and training-only Fisher statistics."""
import numpy as np

DIM = 64
MAX_BYTES = 2 * 1024 ** 3
CUTOFF = 1e-8


def descriptors(packet, columns=None):
    source = np.asarray(packet['source_features'], float)
    target = np.asarray(packet['target_features'], float)
    if (source.ndim != 2 or target.ndim != 2 or source.shape[1] != 32 or target.shape[1] != 32
            or max(len(source), len(target)) > 2048
            or not np.isfinite(source).all() or not np.isfinite(target).all()):
        raise ValueError('Complete finite bounded 32-channel image features required')
    if columns is None:
        columns = np.arange(len(target), dtype=np.int64)
    columns = np.asarray(columns)
    if (columns.ndim != 1 or columns.dtype != np.int64 or len(set(columns)) != len(columns)
            or (columns < 0).any() or (columns >= len(target)).any()):
        raise ValueError('Unique original target columns required')
    n, ns = len(columns), len(source)
    if n * (ns + 1) * DIM * 8 > MAX_BYTES:
        raise ValueError('Bounded complete appearance allocation required')
    a = source / np.maximum(np.linalg.norm(source, axis=1, keepdims=True), 1e-12)
    b = target[columns] / np.maximum(np.linalg.norm(target[columns], axis=1, keepdims=True), 1e-12)
    result = np.zeros((n, ns + 1, DIM), np.float32)
    result[:, :ns, :32] = np.abs(b[:, None] - a[None])
    result[:, :ns, 32:] = b[:, None] * a[None]
    return result.reshape(-1, DIM)


def empty_stats():
    return dict(count=np.zeros(2, np.int64), sums=np.zeros((2, DIM)), seconds=np.zeros((2, DIM, DIM)))


def accumulate(stats, appearance, packed, role):
    if role != 'fitting':
        raise ValueError('Only fitting pairs may enter appearance statistics')
    from research.focus_candidate_ranker import validate
    validate(packed)
    x = np.asarray(appearance)
    if (x.dtype != np.float32 or x.shape != (len(packed['offset']), DIM)
            or not np.isfinite(x).all() or np.any(x[packed['null_rows']] != 0)):
        raise ValueError('Complete aligned finite appearance and zero null rows required')
    label = np.zeros(len(x), np.int64)
    label[packed['chosen'][packed['present'] == 1]] = 1
    label[packed['null_rows']] = -1
    for cls in (0, 1):
        block = x[label == cls].astype(float)
        stats['count'][cls] += len(block)
        stats['sums'][cls] += block.sum(axis=0)
        stats['seconds'][cls] += block.T @ block


def merge_stats(parts, role):
    if role != 'fitting':
        raise ValueError('Only fitting moments may be merged')
    if not parts:
        raise ValueError('At least one fitting moment set required')
    result = empty_stats()
    for part in parts:
        for key in result:
            value = np.asarray(part[key])
            if value.shape != result[key].shape or not np.isfinite(value).all():
                raise ValueError('Exact finite moment shapes required')
            result[key] += value
    return result


def projector(stats, role):
    if role != 'fitting':
        raise ValueError('Only fitting statistics may define the projection')
    stats = merge_stats([stats], role)
    n, sums, seconds = (stats[k] for k in ('count', 'sums', 'seconds'))
    if n.dtype != np.int64 or (n < 2).any():
        raise ValueError('At least two real pairs in each fitting class required')
    total = int(n.sum())
    mean = sums.sum(axis=0) / total
    covariance = seconds.sum(axis=0) / total - np.outer(mean, mean)
    if np.min(np.diag(covariance)) < -1e-9:
        raise ValueError('Invalid negative descriptor variance')
    std = np.sqrt(np.maximum(np.diag(covariance), 0.))
    std = np.where(std < 1e-8, 1., std)
    means = sums / n[:, None]
    within = sum(seconds[c] - np.outer(sums[c], sums[c]) / n[c] for c in (0, 1)) / (total - 2)
    within /= std[:, None] * std[None, :]
    within = (within + within.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(within)
    if eigenvalues[-1] <= 0 or eigenvalues[0] < -1e-7 * eigenvalues[-1]:
        raise ValueError('Positive semidefinite nonzero within-class covariance required')
    keep = eigenvalues > CUTOFF * eigenvalues[-1]
    difference = (means[1] - means[0]) / std
    vector = eigenvectors[:, keep] @ ((eigenvectors[:, keep].T @ difference) / eigenvalues[keep])
    correlation = covariance / (std[:, None] * std[None, :])
    projected_variance = float(vector @ correlation @ vector)
    if not np.isfinite(projected_variance) or projected_variance <= 1e-16:
        raise ValueError('Nonzero finite discriminant required; no fallback')
    vector /= np.sqrt(projected_variance)
    return dict(role=role, feature_dimension=DIM, real_pair_counts=n.tolist(), mean=mean.tolist(), std=std.tolist(),
                lda_vector=vector.tolist(), retained_rank=int(keep.sum()), eigenvalue_cutoff=CUTOFF,
                minimum_eigenvalue=float(eigenvalues[0]), maximum_eigenvalue=float(eigenvalues[-1]),
                projection_rule='pooled_within_class_fisher_pseudoinverse', population_projection_variance=1.)


def transform(appearance, null_rows, model, arm):
    if (arm not in ('full', 'lda') or model['role'] != 'fitting' or model['feature_dimension'] != DIM
            or model['eigenvalue_cutoff'] != CUTOFF):
        raise ValueError('Exact fitted appearance transform and declared arm required')
    x = np.asarray(appearance)
    mean, std, vector = (np.asarray(model[k], float) for k in ('mean', 'std', 'lda_vector'))
    null_rows = np.asarray(null_rows)
    if (x.ndim != 2 or x.shape[1] != DIM or x.nbytes + x.size * 16 > MAX_BYTES
            or any(v.shape != (DIM,) or not np.isfinite(v).all() for v in (mean, std, vector))
            or (std <= 0).any() or not np.isfinite(x).all() or null_rows.dtype != np.int64
            or null_rows.ndim != 1 or (null_rows < 0).any() or (null_rows >= len(x)).any()
            or len(set(null_rows)) != len(null_rows) or np.any(x[null_rows] != 0)):
        raise ValueError('Finite aligned appearance and valid fitted scaler required')
    transformed = (x.astype(float) - mean) / std
    if arm == 'lda':
        transformed = (transformed @ vector)[:, None]
    transformed[null_rows] = 0.
    return transformed
