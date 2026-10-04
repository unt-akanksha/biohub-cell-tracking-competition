"""Pinned full-movie score of hash-bound image motion on unchanged detections."""
import hashlib
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
BASE = runpy.run_path(str(ROOT/'scripts/score-independent-selection.py'))


def validate_manifest(manifest,terminal,codex,reference,reference_sha):
    if (manifest['status'] != 'completed' or terminal['status'] != 'completed'
        or manifest['run_id'] != 'backward-flow-selection-v1' or terminal['run_id'] != manifest['run_id']
        or terminal['declared_budget_seconds'] != 3600 or not 0 < terminal['elapsed_seconds'] <= 3600
        or terminal['submission_performed'] is not False
        or manifest['checkpoint_sha256'] != codex['checkpoint_sha256']
        or manifest['detector_checkpoint_sha256'] != reference['checkpoint_sha256']
        or manifest['split_sha256'] != reference['manifest_sha256']
        or manifest['reference_manifest_sha256'] != reference_sha
        or any(manifest[k] is not False for k in ('ground_truth_opened','target_audit_opened','authorized_for_submission'))
        or [r['stem'] for r in manifest['records']] != [r['stem'] for r in reference['records']]):
        raise ValueError('Exact completed flow and native-node manifests required')
    for r,p in zip(manifest['records'],reference['records']):
        if (r['image_shape'] != p['image_shape'] or r['processed_frames'] != p['processed_frames']
            or r['processed_frames'] != 100 or r['processed_pairs'] != 99
            or r['predicted_nodes'] != p['predicted_nodes'] or r['all_nodes_covered'] is not True
            or r['native_graph_sha256'] != p['graph_sha256']):
            raise ValueError('Incomplete flow or changed detection coverage')


def score(root,notebook,truth_root,flow_root,flow_notebook):
    import numpy as np
    from research.backward_flow_linking import link_backward_flow
    originals,reference = BASE['prepare'](root,notebook)
    manifest = json.loads((flow_root/'outputs/flow_manifest.json').read_text())
    terminal = json.loads((flow_root/'launcher_terminal.json').read_text())
    codex = json.loads(flow_notebook.read_text())['metadata']['codex']
    validate_manifest(manifest,terminal,codex,reference,
                      hashlib.sha256((root/'outputs/selection_manifest.json').read_bytes()).hexdigest())
    bundles = BASE['PILOT']['embedded_sources'](flow_notebook)
    for name,folder,filename in [('sources','repo','source_hashes.json'),('runtime_sources','runtime','runtime_hashes.json')]:
        expected = {p:hashlib.sha256(s.encode()).hexdigest() for p,s in bundles[name].items()}
        if json.loads((flow_root/filename).read_text()) != expected:
            raise ValueError('Flow source receipt mismatch')
        for p,digest in expected.items():
            if hashlib.sha256((flow_root/folder/p).read_bytes()).hexdigest() != digest:
                raise ValueError('Flow runtime source changed')
    cached = []
    for r in manifest['records']:
        path = flow_root/'outputs'/(r['stem']+'.npz')
        if hashlib.sha256(path.read_bytes()).hexdigest() != r['file_sha256']:
            raise ValueError('Motion cache checksum mismatch')
        coords = originals[r['stem']].node_attrs().select(['t','z','y','x']).to_numpy().astype(float)
        with np.load(path,allow_pickle=False) as data:
            if set(data.files) != {'coords','backward_um'} or not np.array_equal(data['coords'],coords):
                raise ValueError('Motion cache changed native detections or order')
            flow = data['backward_um'].copy()
        if flow.shape != (len(coords),3) or not np.isfinite(flow).all() or np.any(flow[coords[:,0] == 0] != 0):
            raise ValueError('Invalid aligned physical flow')
        cached.append((coords,flow))
    pending = iter(cached)
    def linker(coords):
        native,flow = next(pending)
        if not np.array_equal(coords,native):
            raise ValueError('Paired scorer changed cached node order')
        return link_backward_flow(coords,flow)
    result = runpy.run_path(str(ROOT/'scripts/score-independent-motion-prior.py'))['score'](
        root,notebook,truth_root,linker=linker)
    if next(pending,None) is not None:
        raise ValueError('Incomplete paired scoring')
    result.update(run_id='backward-flow-scoring-v1',flow_checkpoint_sha256=manifest['checkpoint_sha256'],
        flow_manifest_sha256=hashlib.sha256((flow_root/'outputs/flow_manifest.json').read_bytes()).hexdigest(),
        scope='Eight complete source-selection movies; fixed native nodes, unchanged variance/null/topology; image-derived backward flow')
    return result
