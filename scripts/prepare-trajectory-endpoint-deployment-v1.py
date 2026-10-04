"""Single ordinary final refit, with no test-embryo name lookup."""
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.public_d4_full_movie import sha
from research.learned_trajectory_endpoint_v1 import fit, calibrate


def main():
    start = time.monotonic()
    source = ROOT / '.biohub/cache/learned-trajectory-endpoint-v1-models'
    output = ROOT / '.biohub/cache/trajectory-endpoint-deployment-v1'
    if output.exists():
        raise ValueError('Preserve deployment fit')
    mp = source / 'manifest.json'
    if sha(mp) != '57236f97756e7287204c1db32b084e0bbbe48315012b3117af444ddfa7296aa3':
        raise ValueError('Frozen source manifest changed')
    manifest = json.loads(mp.read_text()); train, calibration, tm, cm = [], [], [], []
    for _, record in sorted(manifest['models'].items()):
        path = source / record['data_path']
        if sha(path) != record['data_sha256']:
            raise ValueError('Frozen source bank changed')
        with np.load(path, allow_pickle=False) as data:
            train.append(data['optimization']); calibration.append(data['calibration'])
            tm.extend(data['optimization_movies'].tolist()); cm.extend(data['calibration_movies'].tolist())
    if set(tm) & set(cm) or (set(tm) | set(cm)) & set(manifest['excluded_stems']):
        raise ValueError('Data role leakage')
    model = fit(np.concatenate(train), np.asarray(tm))
    threshold = calibrate(model, np.concatenate(calibration))
    output.mkdir(parents=True)
    path = output / 'deployment-model.npz'
    np.savez_compressed(path, **model, threshold=np.asarray(threshold))
    result = dict(status='frozen', source_manifest_sha256=sha(mp),
                  generator_sha256=sha(Path(__file__)),
                  code_sha256=sha(ROOT / 'research/learned_trajectory_endpoint_v1.py'),
                  design_sha256=sha(ROOT / 'reports/experiments/trajectory-endpoint-deployment-v1-design.md'),
                  model_path=path.name, model_sha256=sha(path), threshold=threshold,
                  optimization_windows=len(tm), calibration_windows=len(cm),
                  optimization_stems=sorted(set(tm)), calibration_stems=sorted(set(cm)),
                  excluded_stems=manifest['excluded_stems'], movie_name_used_for_inference=False,
                  authorized_for_submission=False, gpu_hours=0, elapsed_seconds=time.monotonic()-start)
    (output / 'manifest.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if not k.endswith('_stems')}), flush=True)


if __name__ == '__main__':
    main()
