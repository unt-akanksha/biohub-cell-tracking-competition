"""Prepare all original15D fitting choices, with independent feature replay."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.focus_pair_history_head import pack, validate
from research.focus_candidate_ranker import combine, SCALE

RUN = 'focus-pair-history-data-v1'
CACHE = ROOT/'.biohub/cache'/RUN
SOURCES = ['scripts/prepare-focus-pair-history-data.py', 'research/focus_pair_history_head.py',
    'research/focus_pair_history.py', 'research/focus_candidate_ranker.py',
    f'reports/experiments/{RUN}-design.md']


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def inventory():
    history_path = ROOT/'reports/experiments/focus-pair-history-v1-audit.json'
    smoke_path = ROOT/'reports/experiments/focus-pair-history-head-v1-smoke.json'
    original_path = ROOT/'reports/experiments/focus-candidate-ranker-v1-data-smoke.json'
    spec_path = ROOT/'.biohub/cache/kernel-outputs/focus-parent-dropout-training-v1/focus_parent_dropout_training/runtime/training_spec.json'
    expected = {history_path:'799102689148f65daeb557a1aaf8fc002f2524fa5ccb2c2efaf527c4ee81432d',
        smoke_path:'5e3e9be994606a02ea99332b005103af6398c7014b7c8ebc713ccb179c594720',
        original_path:'c0376562166ca50ffdb12e9f79aff7bc867a217c5f69104454e723e45c3b8ccf',
        spec_path:'811dc9187cb587cbc803a0514c92cadb47e41704ef473c5532c269cb5790a36e'}
    if any(sha(p) != v for p, v in expected.items()):
        raise ValueError('Original data, completed history audit and real optimizer smoke required')
    history, smoke, original, spec = [json.loads(p.read_text()) for p in expected]
    for record in (history, smoke, original):
        if any(sha(ROOT/p) != v for p, v in record['source_hashes'].items()):
            raise ValueError('Frozen successful upstream sources required')
    prior = {r['stem']: r for r in original['records']}
    roots = [ROOT/'.biohub/cache/kernel-outputs/focus-adaptation-features-v1/focus_adaptation_features/outputs',
             ROOT/'.biohub/cache/kernel-outputs/focus-extra-fit-features-v1/focus_extra_fit_features/outputs']
    rows = []
    for group, root in zip(spec['feature_groups'], roots):
        for record in group['feature_records']:
            if record['role'] != 'fitting':
                continue
            stem = record['stem']
            if stem not in prior or record['manifest_sha256'] != prior[stem]['manifest_sha256']:
                raise ValueError('Only original fitting inventory permitted')
            path = root/stem/'manifest.json'
            if sha(path) != record['manifest_sha256']:
                raise ValueError('Original fitting manifest changed')
            manifest = json.loads(path.read_text())
            if (manifest['role'] != 'fitting' or manifest['stem'] != stem
                    or [p['file'] for p in manifest['pairs']] != [f'{t:03d}.npz' for t in range(99)]):
                raise ValueError('Exactly99 original fitting transitions required')
            rows.append((root, record, manifest))
    if [r['stem'] for _, r, _ in rows] != spec['contract']['fitting_stems'] or len(rows) != 12:
        raise ValueError('Exact twelve-movie fitting order required')
    return spec, prior, rows, {str(p.relative_to(ROOT)).replace('\\','/'):v for p,v in expected.items()}


def independent_history(packet, previous, parameters):
    ns = len(packet['source_coords'])
    columns = np.flatnonzero(packet['labels'] >= 0)
    result = np.zeros((len(columns), ns+1, 7))
    if previous is None:
        if int(packet['source_frame']) != 0:
            raise ValueError('Independent replay requires complete preceding history')
        return result.reshape(-1, 7)
    mapping = {int(node): i for i, node in enumerate(previous['target_indices'])}
    flow = np.asarray([previous['backward_um'][mapping[int(node)]]
                       for node in packet['source_indices']], float).reshape(ns, 3)
    for i, column in enumerate(columns):
        residual = (np.asarray(packet['source_coords'], float)*SCALE
            -np.asarray(packet['target_coords'][column], float)*SCALE-flow
            -np.asarray(parameters['mean_um']))/np.sqrt(parameters['variance_um2'])
        result[i, :ns, 0] = 1.
        result[i, :ns, 1:4] = residual
        result[i, :ns, 4:7] = residual*residual
    return result.reshape(-1, 7)


def main():
    started = time.monotonic()
    output = ROOT/f'reports/experiments/{RUN}-result.json'
    if CACHE.exists() or output.exists():
        raise ValueError('Never overwrite prepared or partial history arrays')
    if shutil.disk_usage(ROOT).free < 2*1024**3:
        raise ValueError('At least2GiB free local disk required')
    frozen = {p: sha(ROOT/p) for p in SOURCES}
    spec, old, rows, evidence = inventory()
    CACHE.mkdir()
    records, maximum_error = [], 0.
    for root, record, manifest in rows:
        stem = record['stem']
        base_path = ROOT/'.biohub/cache/focus-candidate-ranker-v1'/(stem+'.npz')
        if sha(base_path) != old[stem]['sha256']:
            raise ValueError('Exact original complete base choices required')
        with np.load(base_path, allow_pickle=False) as saved:
            base = {k:saved[k].copy() for k in saved.files}
        parts, previous = [], None
        for pair in manifest['pairs']:
            path = root/stem/pair['file']
            if sha(path) != pair['sha256']:
                raise ValueError('Original fitting packet changed')
            with np.load(path, allow_pickle=False) as saved:
                packet = {k:saved[k].copy() for k in saved.files}
            data = pack(packet, previous, spec['motion_parameters'], 'fitting')
            expected = independent_history(packet, previous, spec['motion_parameters'])
            if expected.size:
                maximum_error = max(maximum_error, float(np.max(np.abs(expected-data['features'][:,8:]))))
                if not np.allclose(expected, data['features'][:,8:], atol=1e-8, rtol=1e-12):
                    raise ValueError('Every history training feature must independently replay')
                parts.append(data)
            previous = packet
            if time.monotonic()-started > 560:
                raise RuntimeError('History preparation approaching600second cap')
        combined = combine(parts)
        validate(combined)
        for key in base:
            actual = combined[key][:, :8] if key == 'features' else combined[key]
            if not np.array_equal(actual, base[key]):
                raise ValueError('Every original base feature, group, target and label must replay')
        path = CACHE/(stem+'.npz')
        np.savez(path, **combined)
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != set(combined) or any(not np.array_equal(saved[k], v) for k,v in combined.items()):
                raise ValueError('Exact serialized15D arrays required')
        records.append(dict(stem=stem, file=path.name, sha256=sha(path), base_sha256=old[stem]['sha256'],
            manifest_sha256=record['manifest_sha256'], groups=len(combined['starts']), choices=len(combined['offset']),
            parents=int(combined['present'].sum()), absent=int((combined['present'] == 0).sum()),
            array_bytes=sum(v.nbytes for v in combined.values()), file_bytes=path.stat().st_size))
        print(json.dumps(records[-1]), flush=True)
        del combined, parts, base
    if (sum(r['groups'] for r in records), sum(r['choices'] for r in records), sum(r['parents'] for r in records),
            sum(r['absent'] for r in records)) != (10915, 4691320, 10754, 161):
        raise ValueError('All original supervised choices must be preserved')
    if any(sha(ROOT/p) != v for p,v in frozen.items()):
        raise ValueError('Frozen data preparation changed')
    result = dict(status='complete_independently_replayed_history_fitting_arrays', source_hashes=frozen,
        evidence=evidence, records=records, total_groups=10915, total_choices=4691320,
        total_array_bytes=sum(r['array_bytes'] for r in records),
        maximum_independent_feature_error=maximum_error, optimizer_steps=0, labels_created=0,
        diagnostic_movies_opened=0, source_movies_opened=0, new_target_movies_opened=0, gpu_seconds=0,
        authorized_for_submission=False, elapsed_seconds=time.monotonic()-started)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','source_hashes','evidence')}, indent=2), flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['--worker']:
        main()
    elif not sys.argv[1:]:
        packages = str(Path(sys.prefix)/'Lib/site-packages')
        source = str(Path(__file__).resolve())
        code = (f'import sys,runpy; sys.path.insert(0,{packages!r}); '
                f'sys.argv=[{source!r},"--worker"]; runpy.run_path(sys.argv[0],run_name="__main__")')
        subprocess.run([sys._base_executable, '-S', '-c', code], cwd=ROOT, timeout=600, check=True)
    else:
        raise ValueError('Unsupported arguments')
