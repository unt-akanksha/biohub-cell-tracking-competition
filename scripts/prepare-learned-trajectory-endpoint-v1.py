"""Fit two frozen opposite-embryo motion models; never read diagnostic graphs."""
import json
from pathlib import Path
import runpy
import sys
import time
from collections import defaultdict, Counter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.learned_trajectory_endpoint_v1 import fit, calibrate, scores, SCALE
from research.public_d4_full_movie import sha, STEMS


def main():
    start = time.monotonic()
    output = ROOT / '.biohub/cache/learned-trajectory-endpoint-v1-models'
    if output.exists():
        raise ValueError('Preserve previous model preparation')
    ip = ROOT / '.biohub/staging/biohub-relational-division-inventory-v3/relational_division_inventory_v3.json'
    cache = ROOT / '.biohub/cache/competition-train-geffs-packed-v1'
    mp = cache / 'train_geff_cache_manifest.json'
    if sha(ip) != '94150632f5a80b2ef48a39743a425cbe1b8e57b1c131c19ef0bde3d97d1c783e' or sha(mp) != '744f06f75388a7c9199179e5a90bfae4fb3aac835d4e5181893f3745601a9cb9':
        raise ValueError('Original data/role inventory changed')
    inventory, manifest = json.loads(ip.read_text()), json.loads(mp.read_text())
    excluded = set(inventory['final_probe_stems']) | set(STEMS)
    roles = {s: 'optimization' for s in manifest['stems'] if s not in excluded}
    for row in inventory['examples']:
        if row['stem'] not in excluded:
            roles[row['stem']] = row['role']
    selected = {s: r for s, r in roles.items() if r in ('optimization', 'selection')}
    pins = {}
    for row in manifest['files']:
        stem = row['relative_path'].split('.geff/', 1)[0]
        if stem not in selected:
            continue
        path = cache / 'train' / row['relative_path']
        if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise ValueError('Source annotations changed')
        pins[row['relative_path']] = row['sha256']
    graph_helper = ROOT / 'scripts/audit-comoving-division-geometry-v1.py'
    graph = runpy.run_path(str(graph_helper))['graph']
    windows = {(e, r): [] for e in ('44b6', '6bba') for r in ('optimization', 'selection')}
    movies = {k: [] for k in windows}
    counts = {}
    for index, (stem, role) in enumerate(sorted(selected.items())):
        nodes, incoming = graph(cache / 'train' / (stem + '.geff'))
        outgoing = defaultdict(list)
        for child, parents in incoming.items():
            for parent in parents:
                if parent not in nodes or child not in nodes or nodes[child][0] != nodes[parent][0] + 1:
                    raise ValueError('Invalid source annotation temporal edge')
                outgoing[parent].append(child)
        trajectories = []
        for first in sorted(nodes):
            chain = [first]
            for _ in range(5):
                choices = outgoing[chain[-1]]
                if len(choices) != 1 or len(incoming[choices[0]]) != 1:
                    break
                chain.append(choices[0])
            if len(chain) == 6:
                trajectories.append(np.asarray([nodes[i][1:] for i in chain]) * SCALE)
        key = (stem.split('_')[0], role)
        windows[key].extend(trajectories); movies[key].extend([stem] * len(trajectories))
        counts[stem] = dict(role=role, windows=len(trajectories))
        if (index + 1) % 25 == 0:
            print(json.dumps(dict(event='movies_prepared', complete=index + 1, total=len(selected))), flush=True)
    output.mkdir(parents=True)
    result = dict(run_id='learned-trajectory-endpoint-v1-models', status='frozen',
                  source_inventory_sha256=sha(ip), truth_manifest_sha256=sha(mp),
                  code_sha256=sha(ROOT / 'research/learned_trajectory_endpoint_v1.py'),
                  design_sha256=sha(ROOT / 'reports/experiments/learned-trajectory-endpoint-v1-design.md'),
                  generator_sha256=sha(Path(__file__)), graph_loader_sha256=sha(graph_helper),
                  excluded_stems=sorted(excluded), per_movie=counts, source_graph_hashes=pins,
                  models={}, gpu_hours=0, authorized_for_submission=False)
    for source, target in (('44b6', '6bba'), ('6bba', '44b6')):
        train = np.asarray(windows[source, 'optimization'])
        calibration = np.asarray(windows[source, 'selection'])
        if len(train) < 100 or len(calibration) < 20:
            result['status'] = 'rejected_insufficient_data'
            break
        model = fit(train, movies[source, 'optimization'])
        threshold = calibrate(model, calibration)
        path = output / f'source-{source}-target-{target}.npz'
        np.savez_compressed(path, **model, threshold=np.asarray(threshold))
        features_path = output / f'source-{source}-trajectories.npz'
        np.savez_compressed(features_path, optimization=train, calibration=calibration,
                            optimization_movies=np.asarray(movies[source, 'optimization']),
                            calibration_movies=np.asarray(movies[source, 'selection']))
        row = dict(source_embryo=source, target_embryo=target, path=path.name, sha256=sha(path),
                   data_path=features_path.name, data_sha256=sha(features_path),
                   threshold=threshold, optimization_windows=len(train), calibration_windows=len(calibration),
                   calibration_acceptance=float((scores(model, calibration) <= threshold).mean()),
                   optimization_stems=sorted(set(movies[source, 'optimization'])),
                   calibration_stems=sorted(set(movies[source, 'selection'])))
        if any(not s.startswith(source + '_') or s in excluded for s in row['optimization_stems'] + row['calibration_stems']):
            raise ValueError('Source/target leakage')
        result['models'][target] = row
        print(json.dumps({k: v for k, v in row.items() if not k.endswith('_stems')}), flush=True)
    result['elapsed_seconds'] = time.monotonic() - start
    (output / 'manifest.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=result['status'], elapsed_seconds=result['elapsed_seconds'],
                         manifest_sha256=sha(output / 'manifest.json'))), flush=True)


if __name__ == '__main__':
    main()
