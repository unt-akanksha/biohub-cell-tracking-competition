"""Freeze broader time coverage using the SAME previously declared movie roles."""
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import zarr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.visual_correspondence_data_v1 import validate_roles


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    start = time.monotonic()
    previous = ROOT/'.biohub/cache/visual-correspondence-ensemble-v1-r2-bundle/BUNDLE.json'
    if sha(previous) != 'a3a59450791c4d6aef7e75f8d7e2a40c5a41f9dc1a4a14d0440f0a1f0e1712d0':
        raise ValueError('Prior roles changed')
    old = json.loads(previous.read_text()); roles = validate_roles(old['records'])
    geffs = ROOT/'.biohub/cache/competition-train-geffs-packed-v1'
    manifest_path = geffs/'train_geff_cache_manifest.json'
    if sha(manifest_path) != '744f06f75388a7c9199179e5a90bfae4fb3aac835d4e5181893f3745601a9cb9':
        raise ValueError('Official-label cache manifest changed')
    manifests = json.loads(manifest_path.read_text())['files']
    result, totals = [], {}
    for stem, role in sorted(roles.items()):
        checks = [r for r in manifests if r['relative_path'].startswith(stem+'.geff/')]
        if len(checks) != 21:
            raise ValueError('Incomplete exact GEFF file set')
        for r in checks:
            path = geffs/'train'/r['relative_path']
            if path.stat().st_size != r['bytes'] or sha(path) != r['sha256']:
                raise ValueError('GEFF data changed')
        group = zarr.open_group(str(geffs/'train'/(stem+'.geff')), mode='r')
        ids = np.asarray(group['nodes/ids']).astype(np.int64)
        times = np.asarray(group['nodes/props/t/values']).astype(np.int64)
        positions = np.stack([np.asarray(group[f'nodes/props/{a}/values']) for a in ('z','y','x')], 1)
        edges = np.asarray(group['edges/ids']).astype(np.int64)
        id_to_row = {int(node): i for i, node in enumerate(ids)}
        if len(id_to_row) != len(ids) or len({int(b) for a,b in edges}) != len(edges):
            raise ValueError('Node or unique-parent identity violation')
        transitions = {}
        for parent, child in edges:
            a, b = id_to_row[int(parent)], id_to_row[int(child)]
            t = int(times[a])
            if times[b] != t+1 or not 0 <= t < 99:
                raise ValueError('Unexpected temporal edge')
            transitions.setdefault(t, []).append([int(parent), int(child)])
        available = sorted(transitions)
        selected = available if role == 'selection' else [available[i] for i in np.unique(np.rint(np.linspace(0, len(available)-1, min(16, len(available)))).astype(int))]
        selected_edges = [e for t in selected for e in transitions[t]]
        frames = sorted({f for t in selected for f in (t,t+1)})
        keep = np.isin(times, frames)
        nodes = [[int(node), int(t), *map(float,p)] for node,t,p in zip(ids[keep], times[keep], positions[keep])]
        record = dict(stem=stem, embryo=stem.split('_')[0], role=role,
                      available_annotated_transitions=len(available), selected_transitions=selected,
                      image_frames=frames, nodes=nodes, edges=selected_edges, geff_files=checks)
        result.append(record)
        key = record['embryo']+'-'+role
        total = totals.setdefault(key, dict(movies=0, annotated_correspondences=0, frames=0, transitions=0))
        total['movies'] += 1; total['annotated_correspondences'] += len(selected_edges)
        total['frames'] += len(frames); total['transitions'] += len(selected)
    output = ROOT/'.biohub/cache/native-correspondence-v2-plan'
    output.mkdir(parents=True, exist_ok=False)
    payload = dict(run_id='native-correspondence-v2', status='frozen_before_images_or_model_scores', movies=result,
        previous_bundle_sha256=sha(previous), geff_manifest_sha256=sha(manifest_path), totals=totals,
        role_assignment='unchanged v1 movie roles, exclusions and sealed-audit boundary',
        optimization_sampling='up to16 uniformly spaced annotated transitions per movie; every known edge at those transitions',
        selection_sampling='ALL annotated transitions of each existing selection movie',
        competition_test_data_read=False, sealed_audit_opened=False, authorized_for_submission=False,
        patch_scales_um=[.8125, 1.625, 3.25], patch_storage_size=15, model_crop_size=11,
        data_hypothesis='Remove XY decimation aliasing; add finer raw-image detail; greatly increase temporal coverage and source44 examples',
        query_hypothesis='Train on one-to-one image-proposal matched child positions, not GT-centered image queries',
        elapsed_seconds=time.monotonic()-start)
    (output/'MOVIES.json').write_text(json.dumps(payload, indent=2)+'\n')
    print(json.dumps(dict(status=payload['status'], totals=totals, movie_plan_sha256=sha(output/'MOVIES.json'), seconds=payload['elapsed_seconds'])))


if __name__ == '__main__':
    main()
