"""Source-only candidate coverage; no fitter, prediction changes or holdout reads."""
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.public_d4_full_movie import sha
from research.trajectory_candidate_inventory_v1 import candidates, label


def main():
    started = time.monotonic()
    target = ROOT/'.biohub/cache/trajectory-candidate-supervision-v1'
    assert not target.exists()
    plan = ROOT/'.biohub/cache/trajectory-disagreement-source-v1-plan/MOVIES.json'
    assert sha(plan) == '5c4b64793068565598537db6bb3af439513bde3d46e357ffeab59164bda020bf'
    scope = json.loads(plan.read_text())
    assert all(m['role'] == 'optimization' for m in scope['movies'])
    stems = [m['stem'] for m in scope['movies']]
    base = ROOT/'.biohub/cache/trajectory-disagreement-source-v1-full-output'
    receipt = json.loads((ROOT/'reports/experiments/trajectory-disagreement-source-v1-full-harvest.json').read_text())
    assert receipt['status'] == 'verified_backup'
    for r in receipt['records']:
        assert sha(base/r['path']) == r['sha256']
    terminal = json.loads((base/'result.json').read_text())
    assert terminal['status'] == 'complete_prelabel_predictions' and set(terminal['movies']) == set(stems)
    assert terminal['contract_sha256'] == 'ba27bfeb00ab800507275e0d9e5864e53033cb4350505a41331f574e7f9ab80c'
    target.mkdir()
    frozen = {}
    for stem in stems:
        folder = base/(stem+'-original')
        initial = json.loads((folder/'pre-postprocess.json').read_text())
        final = json.loads((folder/'repaired-prediction.json').read_text())
        groups = candidates(initial, final)
        path = target/(stem+'-candidates.npz')
        np.savez_compressed(path, **groups)
        frozen[stem] = sha(path)
        print(json.dumps(dict(stem=stem, stage='candidates_frozen', groups=len(groups['children']), edges=len(groups['parents']))), flush=True)
    (target/'PRELABEL.json').write_text(json.dumps(frozen, indent=2)+'\n')
    truth_root = ROOT/'.biohub/cache/competition-train-geffs-packed-v1'
    manifest = truth_root/'train_geff_cache_manifest.json'
    assert sha(manifest) == '744f06f75388a7c9199179e5a90bfae4fb3aac835d4e5181893f3745601a9cb9'
    files = json.loads(manifest.read_text())['files']
    import zarr
    rows, label_hashes = {}, {}
    for stem in stems:
        selected = [r for r in files if r['relative_path'].startswith(stem+'.geff/')]
        assert len(selected) == 21
        for r in selected:
            assert sha(truth_root/'train'/r['relative_path']) == r['sha256']
        gt = zarr.open_group(str(truth_root/'train'/(stem+'.geff')), mode='r')
        ids = np.asarray(gt['nodes/ids'])
        arrays = {a: np.asarray(gt[f'nodes/props/{a}/values']) for a in ('t','z','y','x')}
        nodes = {int(i): dict(t=int(arrays['t'][j]), **{a:float(arrays[a][j]) for a in ('z','y','x')}) for j,i in enumerate(ids)}
        edges = np.asarray(gt['edges/ids']).astype(np.int64).tolist()
        assert len(nodes) == len(ids) and len({b for a,b in edges}) == len(edges)
        final = json.loads((base/(stem+'-original')/'repaired-prediction.json').read_text())
        path = target/(stem+'-candidates.npz')
        assert sha(path) == frozen[stem]
        with np.load(path, allow_pickle=False) as packed:
            groups = dict(packed)
        labels, safe, rows[stem] = label(groups, final, nodes, edges)
        label_path = target/(stem+'-labels.npz')
        np.savez_compressed(label_path, target=labels, safe=safe)
        label_hashes[stem] = sha(label_path)
        print(json.dumps(dict(stem=stem, stage='labels_attached', **rows[stem])), flush=True)
    result = dict(status='source_candidate_inventory_complete', per_movie=rows,
                  candidate_sha256=frozen, label_sha256=label_hashes,
                  source_scope_sha256=sha(plan), source_sha256=sha(Path(__file__)),
                  helper_sha256=sha(ROOT/'research/trajectory_candidate_inventory_v1.py'),
                  source_only=True, selection_or_validation_opened=False, model_fitted=False,
                  predictions_changed=False, authorized_for_submission=False,
                  seconds=time.monotonic()-started)
    (target/'RESULT.json').write_text(json.dumps(result, indent=2)+'\n')
    (ROOT/'reports/experiments/trajectory-candidate-supervision-v1.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(status=result['status'], seconds=result['seconds'])))


if __name__ == '__main__':
    main()
