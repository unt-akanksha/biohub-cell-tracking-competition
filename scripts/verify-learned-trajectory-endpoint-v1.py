"""Independently replay source fits with weighted least squares, not solve()."""
import json
from pathlib import Path
import sys
import math
import numpy as np
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.public_d4_full_movie import sha, STEMS


def main():
    root = ROOT / '.biohub/cache/learned-trajectory-endpoint-v1-models'
    output = ROOT / 'reports/experiments/learned-trajectory-endpoint-v1-verification.json'
    if output.exists():
        raise ValueError('Preserve completed verification')
    manifest_path = root / 'manifest.json'
    if sha(manifest_path) != '57236f97756e7287204c1db32b084e0bbbe48315012b3117af444ddfa7296aa3':
        raise ValueError('Frozen model manifest changed')
    manifest = json.loads(manifest_path.read_text()); receipts = {}
    voxel = np.array([1.625, .40625, .40625])
    for target, record in manifest['models'].items():
        if record['source_embryo'] == target:
            raise ValueError('Source must exclude target embryo')
        for key in ('path', 'data_path'):
            digest = record['sha256' if key == 'path' else 'data_sha256']
            if sha(root / record[key]) != digest:
                raise ValueError('Frozen artifact changed')
        with np.load(root / record['path'], allow_pickle=False) as file:
            model = {k: file[k] for k in file.files}
        with np.load(root / record['data_path'], allow_pickle=False) as file:
            train = file['optimization']; calibration = file['calibration']
            tm, cm = file['optimization_movies'], file['calibration_movies']
        if (set(tm) & set(cm) or set(tm) != set(record['optimization_stems'])
                or set(cm) != set(record['calibration_stems'])
                or any(s in manifest['excluded_stems'] or not s.startswith(record['source_embryo'] + '_')
                       for s in set(tm) | set(cm))):
            raise ValueError('Source movie roles or exclusion violated')
        delta = train[:, 1:] - train[:, :-1]
        first = np.hstack((delta[:, 0], delta[:, 1]))
        second = np.hstack((-delta[:, 4], -delta[:, 3]))
        xraw = np.vstack((first, second)); y = np.vstack((delta[:, 2], -delta[:, 2]))
        count = {s: int((tm == s).sum()) for s in set(tm)}
        pw = np.array([1 / count[s] for s in tm]); pw /= pw.sum()
        base = np.concatenate((pw, pw)) / 2
        scale = np.sqrt(np.sum(base[:, None] * xraw ** 2, axis=0)).clip(min=.001)
        x = np.column_stack((xraw / scale, np.ones(len(xraw))))
        ridge = np.diag(np.sqrt([.01] * 6 + [1e-10]))
        robust = np.ones(len(x)); max_difference = 0.
        for _ in range(10):
            w = base * robust; w /= w.sum()
            augmented = np.vstack((np.sqrt(w[:, None]) * x, ridge))
            response = np.vstack((np.sqrt(w[:, None]) * y, np.zeros((7, 3))))
            coefficient = np.linalg.lstsq(augmented, response, rcond=None)[0]
            residual = y - x @ coefficient
            covariance = (residual * w[:, None]).T @ residual + np.diag(voxel ** 2 / 6)
            solved = np.linalg.solve(covariance, residual.T).T
            distance = np.sum(residual * solved, axis=1)
            robust = np.minimum(1, np.sqrt(chi2.ppf(.95, 3) / np.maximum(distance, 1e-12)))
        np.testing.assert_allclose(scale, model['scale'], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(coefficient, model['coef'], rtol=1e-9, atol=1e-10)
        def joint(windows):
            d = np.diff(windows, axis=1)
            left = np.column_stack((np.hstack((d[:, 0], d[:, 1])) / scale, np.ones(len(d))))
            right = np.column_stack((np.hstack((-d[:, 4], -d[:, 3])) / scale, np.ones(len(d))))
            return np.hstack((d[:, 2] - left @ coefficient, -d[:, 2] - right @ coefficient))
        jr = joint(train)
        jw = pw * np.sqrt(robust[:len(train)] * robust[len(train):]); jw /= jw.sum()
        mean = np.sum(jw[:, None] * jr, axis=0); centered = jr - mean
        covariance = (centered * jw[:, None]).T @ centered + np.diag(np.tile(voxel ** 2 / 6, 2))
        np.testing.assert_allclose(mean, model['residual_mean'], rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(covariance, model['covariance'], rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(model['precision'] @ covariance, np.eye(6), rtol=1e-8, atol=1e-8)
        errors = joint(calibration) - mean
        values = np.sum(errors * np.linalg.solve(covariance, errors.T).T, axis=1)
        index = min(len(values), math.ceil(.99 * (len(values) + 1))) - 1
        threshold = float(np.sort(values)[index])
        np.testing.assert_allclose(threshold, record['threshold'], rtol=1e-9, atol=1e-8)
        receipts[target] = dict(source_embryo=record['source_embryo'],
            optimization_windows=len(train), calibration_windows=len(calibration),
            calibration_threshold=threshold, minimum_covariance_eigenvalue=float(np.linalg.eigvalsh(covariance).min()),
            weighted_least_squares_replay=True, source_exclusion_verified=True)
    result = dict(status='verified_source_models', manifest_sha256=sha(manifest_path),
                  verifier_sha256=sha(Path(__file__)), models=receipts,
                  candidate_quality_evaluated=False, authorized_for_submission=False)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=result['status'], result_sha256=sha(output))), flush=True)


if __name__ == '__main__':
    main()
