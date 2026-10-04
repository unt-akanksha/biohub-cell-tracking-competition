"""Package only image-derived inference functions and the fixed 18 learned weights."""
import ast
import hashlib
import json
from pathlib import Path
import runpy
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.public_d4_full_movie import sha


def selected(path,names):
    source=path.read_text(encoding='utf-8');tree=ast.parse(source);parts={}
    for node in tree.body:
        name=node.name if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) else None
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):name=node.targets[0].id
        if name in names:parts[name]=ast.get_source_segment(source,node)
    assert set(parts)==set(names)
    return '\n\n'.join(parts[n] for n in names)+'\n'


def main():
    started=time.monotonic();target=ROOT/'.biohub/cache/trajectory-structured-portable-v1';assert not target.exists()
    quality_path=ROOT/'reports/experiments/trajectory-structured-eight-v1-result.json'
    quality=json.loads(quality_path.read_text());assert quality['quality_pass'] and all(quality['gates'].values())
    model=ROOT/'.biohub/cache/trajectory-structured-loss-v1-full/weights.npz'
    assert sha(model)=='e33fe1b79291ed89697db7a5ee6a839bc2ef34e23f504b27f1f29e44290107ba'
    with np.load(model,allow_pickle=False) as f:weights=f['6bba']
    sources=[('trajectory_disagreement_data_v1.py',('SCALE','positions','adjacency')),
             ('trajectory_candidate_inventory_v1.py',('candidates',)),
             ('trajectory_candidate_ranker_v1.py',('FEATURES','features')),
             ('trajectory_joint_assignment_v1.py',('apply',))]
    text='"""Prediction-only structured trajectory assignment; no fitting or annotations."""\nfrom collections import Counter,defaultdict\nfrom copy import deepcopy\nimport numpy as np\nfrom scipy.spatial import cKDTree\nfrom scipy.optimize import linear_sum_assignment\n\n'
    source_hashes={}
    for name,names in sources:
        path=ROOT/'research'/name;source_hashes[name]=sha(path);text+=selected(path,names)+'\n'
    text+='''def refine(initial, final, raw_coords, raw_edges, weights):
    for ident,node in initial['nodes'].items():
        ident=int(ident)
        if not 0<=ident<len(raw_coords) or not np.array_equal(raw_coords[ident],[node[k] for k in ('t','z','y','x')]):
            raise ValueError('Raw detector identity changed')
    groups=candidates(initial,final)
    matrix=features(initial,final,groups,raw_edges)
    return apply(final,groups,matrix,np.asarray(weights,np.float64))
'''
    ast.parse(text)
    target.mkdir();module=target/'structured-trajectory.py';module.write_bytes(text.encode())
    env=runpy.run_path(str(module));assert np.shape(weights)==(len(env['FEATURES']),)
    payload=dict(schema='structured-trajectory-v1',features=env['FEATURES'],weights=weights.tolist(),
                 training_weights_sha256=sha(model),inference_source_sha256=source_hashes,
                 quality_reference_sha256=sha(quality_path),training_data_included=False)
    model_path=target/'structured-trajectory-model.json';model_path.write_bytes((json.dumps(payload,indent=2)+'\n').encode())
    cached=ROOT/'.biohub/cache/trajectory-overlap-cache-kaggle-v1-output'
    expected=ROOT/'.biohub/cache/trajectory-structured-eight-v1';tests=[]
    for stem,record in quality['records'].items():
        matches=list((cached/'trajectory-complete').glob('shard-*/'+stem+'-original'));assert len(matches)==1
        folder=matches[0]
        for filename,key in (('pre-postprocess.json','initial_sha256'),('raw-candidates.npz','raw_sha256'),('repaired-prediction.json','baseline_sha256')):
            assert sha(folder/filename)==record[key]
        initial=json.loads((folder/'pre-postprocess.json').read_text());final=json.loads((folder/'repaired-prediction.json').read_text())
        before=time.monotonic()
        with np.load(folder/'raw-candidates.npz',allow_pickle=False) as raw:
            result,details=env['refine'](initial,final,raw['coords'],raw['edges'],payload['weights'])
        reference=expected/(stem+'-prediction.json');assert sha(reference)==record['prediction_sha256']
        assert result==json.loads(reference.read_text())
        tests.append(dict(stem=stem,exact_graph_identity=True,seconds=time.monotonic()-before,changed_edges=details['changed_edges']))
    report=dict(status='portable_inference_verified',code_sha256=sha(module),model_sha256=sha(model_path),
                quality_sha256=sha(quality_path),source_hashes=source_hashes,records=tests,
                model_parameters=len(weights),annotations_or_training_code_packaged=False,
                kaggle_runtime_acceptance_required=True,authorized_for_submission=False,seconds=time.monotonic()-started)
    (target/'RESULT.json').write_text(json.dumps(report,indent=2)+'\n')
    (ROOT/'reports/experiments/trajectory-structured-portable-v1.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
