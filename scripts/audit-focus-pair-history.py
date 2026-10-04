"""Verify previous-image-flow availability on the original fitting cache only."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.focus_cached_pair import validate_pair
from research.focus_pair_history import FEATURES, candidate_history, previous_source_motion
from research.focus_candidate_ranker import SCALE

RUN = 'focus-pair-history-v1'
SOURCES = ['scripts/audit-focus-pair-history.py', 'research/focus_pair_history.py',
    'research/focus_cached_pair.py', 'research/focus_candidate_ranker.py',
    'tests/test_focus_pair_history.py', f'reports/experiments/{RUN}-design.md']


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def packet(folder, entry, *, labels=True):
    path = folder/entry['file']
    if sha(path) != entry['sha256']:
        raise ValueError('Exact original cached packet required')
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k].copy() for k in data.files if labels or k != 'labels'}


def main():
    started = time.monotonic()
    output = ROOT/f'reports/experiments/{RUN}-audit.json'
    if output.exists():
        raise ValueError('Never overwrite a completed history audit')
    frozen = {p: sha(ROOT/p) for p in SOURCES}
    spec_path = ROOT/'.biohub/cache/kernel-outputs/focus-parent-dropout-training-v1/focus_parent_dropout_training/runtime/training_spec.json'
    prior_path = ROOT/'reports/experiments/focus-candidate-ranker-v1-data-smoke.json'
    if (sha(spec_path) != '811dc9187cb587cbc803a0514c92cadb47e41704ef473c5532c269cb5790a36e'
            or sha(prior_path) != 'c0376562166ca50ffdb12e9f79aff7bc867a217c5f69104454e723e45c3b8ccf'):
        raise ValueError('Original approved12 fitting data receipts required')
    spec = json.loads(spec_path.read_text())
    old = {r['stem']: r for r in json.loads(prior_path.read_text())['records']}
    roots = [ROOT/'.biohub/cache/kernel-outputs/focus-adaptation-features-v1/focus_adaptation_features/outputs',
             ROOT/'.biohub/cache/kernel-outputs/focus-extra-fit-features-v1/focus_extra_fit_features/outputs']
    inventory = []
    for group, root in zip(spec['feature_groups'], roots):
        for entry in group['feature_records']:
            if entry['role'] != 'fitting':
                continue
            stem = entry['stem']
            if stem not in old or entry['manifest_sha256'] != old[stem]['manifest_sha256']:
                raise ValueError('Only the original fitting inventory may be read')
            path = root/stem/'manifest.json'
            if sha(path) != entry['manifest_sha256']:
                raise ValueError('Original fitting manifest changed')
            manifest = json.loads(path.read_text())
            if (manifest['role'] != 'fitting' or manifest['stem'] != stem
                    or [p['file'] for p in manifest['pairs']] != [f'{t:03d}.npz' for t in range(99)]):
                raise ValueError('Exactly99 fitting transitions required')
            inventory.append((root, entry, manifest))
    if [r['stem'] for _, r, _ in inventory] != spec['contract']['fitting_stems'] or len(inventory) != 12:
        raise ValueError('Exact ordered12 fitting movies required')
    parameters = spec['motion_parameters']
    root, entry, manifest = next(row for row in inventory if row[1]['stem'] == '6bba_57b7cc1e')
    # Original fixed stress pair; all targets, with labels omitted from loading.
    current = packet(root/entry['stem'], manifest['pairs'][31], labels=False)
    previous = packet(root/entry['stem'], manifest['pairs'][30], labels=False)
    initial = [{k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in p.items()} for p in (current, previous)]
    ns, nt = len(current['source_coords']), len(current['target_coords'])
    native_hashes, maximum_error, maximum_bound = [], 0., 0
    tick = time.monotonic()
    for block_size in (32, 17):
        digest = hashlib.sha256()
        for first in range(0, nt, block_size):
            columns = np.arange(first, min(first+block_size, nt), dtype=np.int64)
            out = candidate_history(current, previous, parameters, columns)
            digest.update(out['features'].tobytes())
            maximum_bound = max(maximum_bound, out['conservative_working_bytes'])
            if block_size == 32:
                # Independent formula and ID lookup, not the production join.
                mapping = {int(node): i for i, node in enumerate(previous['target_indices'])}
                flow = np.array([previous['backward_um'][mapping[int(i)]] for i in current['source_indices']], float)
                actual = out['features'].reshape(len(columns), ns+1, 7)
                for j, column in enumerate(columns):
                    residual = (np.asarray(current['source_coords'], float)*SCALE
                        -np.asarray(current['target_coords'][column], float)*SCALE-flow
                        -np.asarray(parameters['mean_um']))/np.sqrt(parameters['variance_um2'])
                    reference = np.zeros((ns+1, 7))
                    reference[:ns, 0] = 1
                    reference[:ns, 1:4] = residual
                    reference[:ns, 4:7] = residual*residual
                    maximum_error = max(maximum_error, float(np.max(np.abs(reference-actual[j]))))
                    if not np.allclose(reference, actual[j], atol=1e-8, rtol=1e-12):
                        raise ValueError('Independent real history formula differs')
            if time.monotonic()-started > 280:
                raise RuntimeError('History audit approaching300second cap')
        native_hashes.append(digest.hexdigest())
    if native_hashes[0] != native_hashes[1]:
        raise ValueError('All original target feature rows must be block-size invariant')
    for before, after in zip(initial, (current, previous)):
        if any(hashlib.sha256(after[k].tobytes()).hexdigest() != v for k, v in before.items()):
            raise ValueError('Feature computation modified an original input')
    smoke = dict(stem=entry['stem'], source_frame=31, sources=ns, targets=nt,
        choices=nt*(ns+1), current_packet_sha256=manifest['pairs'][31]['sha256'],
        previous_packet_sha256=manifest['pairs'][30]['sha256'],
        features_sha256=native_hashes[0], exact_block_equivalence=True,
        maximum_independent_formula_error=maximum_error, maximum_conservative_block_bytes=maximum_bound,
        elapsed_seconds=time.monotonic()-tick, labels_loaded=False, quality_evaluated=False)
    print(json.dumps(dict(stage='complete_real_smoke', **smoke)), flush=True)
    rows = []
    for root, entry, manifest in inventory:
        stem = entry['stem']
        raw_path = root/'raw_detections'/(stem+'.npz')
        if sha(raw_path) != entry['raw_sha256'] or entry['raw_sha256'] != old[stem]['raw_sha256']:
            raise ValueError('Exact original fitting geometry required')
        with np.load(raw_path, allow_pickle=False) as data:
            coords = data['coords'].copy()
        previous = None
        counts = dict(known_parent=0, known_absent=0, unknown=0, known_parent_with_history=0,
            known_absent_with_history=0, unknown_with_history=0, boundary_known_parent=0,
            boundary_known_absent=0, source_nodes_with_history=0, transitions_with_history=0)
        digest = hashlib.sha256()
        for t, pair in enumerate(manifest['pairs']):
            current = packet(root/stem, pair)
            actual = validate_pair(current, coords)
            if any(actual[k] != pair[k] for k in actual):
                raise ValueError('Original complete pair labels/counts must replay')
            flow, available = previous_source_motion(current, previous)
            if not np.array_equal(available, np.full(len(current['source_coords']), t > 0)):
                raise ValueError('History must exist for every source except the boundary')
            for key in ('known_parent', 'known_absent', 'unknown'):
                counts[key] += actual[key]
                if t > 0:
                    counts[key+'_with_history'] += actual[key]
                elif key != 'unknown':
                    counts['boundary_'+key] += actual[key]
            counts['source_nodes_with_history'] += int(available.sum())
            counts['transitions_with_history'] += int(t > 0)
            digest.update(flow.tobytes())
            previous = current
            if time.monotonic()-started > 280:
                raise RuntimeError('History audit approaching300second cap')
        if counts['known_parent']+counts['known_absent'] != old[stem]['groups']:
            raise ValueError('Original full known-group coverage changed')
        rows.append(dict(stem=stem, manifest_sha256=entry['manifest_sha256'],
            raw_sha256=entry['raw_sha256'], transitions=99, aligned_flow_sha256=digest.hexdigest(), **counts))
        print(json.dumps(dict(stage='history_inventory', **rows[-1])), flush=True)
    totals = {k: sum(row[k] for row in rows) for k in counts}
    if (totals['known_parent'], totals['known_absent'], totals['unknown'], totals['transitions_with_history']) != (10754, 161, 451160, 1176):
        raise ValueError('Original complete fitting coverage differs')
    if any(sha(ROOT/p) != v for p, v in frozen.items()):
        raise ValueError('Frozen history method changed during audit')
    result = dict(status='verified_original_fitting_history_data_and_real_smoke', source_hashes=frozen,
        training_spec_sha256=sha(spec_path), original_data_sha256=sha(prior_path), feature_names=list(FEATURES),
        smoke=smoke, rows=rows, totals=totals, gpu_seconds=0, optimizer_steps=0, labels_created=0,
        quality_evaluated=False, diagnostic_movies_opened=0, source_movies_opened=0, new_target_movies_opened=0,
        authorized_for_submission=False, elapsed_seconds=time.monotonic()-started, actual_worker_pid=os.getpid())
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({k: v for k, v in result.items() if k not in ('rows', 'source_hashes')}, indent=2), flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['--worker']:
        main()
    elif not sys.argv[1:]:
        # Direct base interpreter avoids Windows venv redirector descendants
        # surviving a subprocess timeout; keep only this isolated package path.
        packages = str(Path(sys.prefix)/'Lib/site-packages')
        source = str(Path(__file__).resolve())
        code = (f'import sys,runpy; sys.path.insert(0,{packages!r}); '
                f'sys.argv=[{source!r},"--worker"]; runpy.run_path(sys.argv[0],run_name="__main__")')
        subprocess.run([sys._base_executable, '-S', '-c', code], cwd=ROOT, timeout=300, check=True)
    else:
        raise ValueError('Unsupported arguments')
