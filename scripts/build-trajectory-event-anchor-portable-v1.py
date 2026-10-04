"""Package a fixed diagnostic candidate; no automatic release-gate waiver."""
import ast
import json
from pathlib import Path
import runpy
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.trajectory_runtime_v1 import sha,validate_graph


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def select(path,names,renames=None):
    tree=ast.parse(path.read_text(encoding='utf-8'));found={}
    for node in tree.body:
        name=node.name if isinstance(node,(ast.FunctionDef,ast.ClassDef)) else None
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            name=node.targets[0].id
        if name in names:found[name]=node
    assert set(found)==set(names),(path,names)
    class Rename(ast.NodeTransformer):
        def visit_Name(self,node):
            node.id=(renames or {}).get(node.id,node.id);return node
    result=[]
    for name in names:
        node=Rename().visit(found[name])
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)):node.name=(renames or {}).get(node.name,node.name)
        result.append(ast.unparse(ast.fix_missing_locations(node)))
    return '\n\n'.join(result)+'\n\n'


def main():
    started=time.monotonic();name='trajectory-event-anchor-portable-v1'
    target=ROOT/'.biohub/cache'/name;receipt=ROOT/'reports/experiments'/(name+'.json')
    assert not target.exists() and not receipt.exists()
    quality_path=ROOT/'reports/experiments/trajectory-event-anchor-eight-v1.json'
    assert sha(quality_path)=='f56568612879bf5d7f604a459b17ca2a7f4f6185920f51af8e0cdccfa3d465b1'
    quality=read(quality_path)
    assert quality['status']=='fixed_anchor_eight_diagnostic_complete'
    assert quality['quality_checks']['pooled_score_improves'] and quality['quality_checks']['every_embryo_nonregressing']
    assert quality['selection_failure_preserved'] and not quality['overall_release_gates_pass']
    artifact=ROOT/'.biohub/cache/trajectory-event-learning-smoke-v1/epoch-3.npz'
    assert sha(artifact)==quality['model_sha256'] and quality['parameter_key']=='anchor'
    with np.load(artifact,allow_pickle=False) as data:weights=data['anchor']
    old_model=read(ROOT/'.biohub/cache/trajectory-structured-portable-v1/structured-trajectory-model.json')
    for source,digest in old_model['inference_source_sha256'].items():assert sha(ROOT/'research'/source)==digest
    modules=[
        ('trajectory_disagreement_data_v1.py',('SCALE','positions','adjacency'),{}),
        ('trajectory_candidate_inventory_v1.py',('candidates',),{}),
        ('trajectory_candidate_ranker_v1.py',('FEATURES','features'),{'FEATURES':'EDGE_FEATURES','features':'edge_features'}),
        ('trajectory_event_features_v1.py',('FEATURES',),{}),
        ('trajectory_event_assignment_v1.py',('EventSolveError','problem','validate_choice','solve'),{}),
        ('trajectory_event_candidates_v1.py',('frames',),{}),
        ('trajectory_event_fast_features_v1.py',('FeatureContext',),{}),
        ('trajectory_event_dominance_v1.py',('allowed_options','infer'),{}),
        ('trajectory_event_pruned_inference_v1.py',('refine',),{'refine':'refine_prepared'})]
    expected={
        'trajectory_event_features_v1.py':'6091c871aabc55a5ee314a67d9be39f54616b641628f023e3088a41abf9acd19',
        'trajectory_event_assignment_v1.py':'f554fc5ff2e3b4e4d39693244e96fa226db5295e74274965cf8a771f447fcded',
        'trajectory_event_candidates_v1.py':'b0140970917fe44c9d5f5599560406441c79a4488647eaf589c3a77495ac732d',
        'trajectory_event_fast_features_v1.py':'2ddbfadf8e94680ca265f6b5db0b6adc549de1c1516d3b7c693c0f45fec5296f',
        'trajectory_event_pruned_inference_v1.py':quality['inference_sha256']}
    for source,digest in expected.items():assert sha(ROOT/'research'/source)==digest
    code='''"""Prediction-only joint trajectory inference. No training data or movie router."""
from collections import Counter, defaultdict
from itertools import combinations
import time
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

'''
    source_hashes={}
    for source,names,renames in modules:
        path=ROOT/'research'/source;source_hashes[source]=sha(path);code+=select(path,names,renames)
    code+='''def refine(initial, final, raw_coords, raw_edges, weights, *, per_frame_seconds=2., max_seconds=120.):
    for ident, node in initial['nodes'].items():
        ident=int(ident)
        if not 0 <= ident < len(raw_coords) or not np.array_equal(raw_coords[ident], [node[k] for k in ('t','z','y','x')]):
            raise ValueError('Raw detector identity changed')
    groups=candidates(initial, final)
    matrix=edge_features(initial, final, groups, raw_edges)
    return refine_prepared(initial, final, groups, matrix, weights,
                           per_frame_seconds=per_frame_seconds, max_seconds=max_seconds)
'''
    tree=ast.parse(code)
    assert not any(isinstance(n,ast.ImportFrom) and (n.module or '').startswith('research') for n in ast.walk(tree))
    assert not any(isinstance(n,ast.FunctionDef) and n.name in ('fit','prepare','label','train') for n in ast.walk(tree))
    target.mkdir();module=target/'event-trajectory.py';module.write_bytes(code.encode('utf-8'))
    env=runpy.run_path(str(module));assert weights.shape==(len(env['FEATURES']),)==(30,)
    model=dict(schema='event-trajectory-anchor-v1',features=env['FEATURES'],weights=weights.tolist(),
        source_artifact_sha256=sha(artifact),parameter_key='anchor',inference_source_sha256=source_hashes,
        quality_reference_sha256=sha(quality_path),training_data_included=False,selection_failure_preserved=True,
        validation_movie_failure_preserved=True,authorized_for_submission=False)
    model_path=target/'event-trajectory-model.json'
    model_path.write_bytes((json.dumps(model,indent=2,allow_nan=False)+'\n').encode('utf-8'))
    report=dict(status='running',code_sha256=sha(module),model_sha256=sha(model_path),
        source_sha256=sha(Path(__file__)),quality_reference_sha256=sha(quality_path),records=[],
        training_or_annotations_included=False,movie_id_routing=False,selection_failure_preserved=True,
        validation_movie_failure_preserved=True,authorized_for_submission=False,blas_thread_limit=1)
    def persist():
        report['seconds']=time.monotonic()-started
        text=json.dumps(report,indent=2,allow_nan=False)+'\n'
        (target/'RESULT.json').write_text(text,encoding='utf-8');receipt.write_text(text,encoding='utf-8')
    persist()
    try:
        inputs_root=ROOT/'.biohub/cache/trajectory-event-eight-v1-features';inputs=read(inputs_root/'RESULT.json')
        assert sha(inputs_root/'RESULT.json')==quality['input_receipt_sha256']
        # Small complete-movie smoke precedes the other seven deployment replays.
        stems=['6bba_23af9eeb']+[s for s in inputs['per_movie'] if s!='6bba_23af9eeb']
        for stem in stems:
            frozen=inputs['per_movie'][stem];ip=ROOT/frozen['initial_path'];rp=ip.parent/'raw-candidates.npz'
            bp=inputs_root/(stem+'-prediction.json')
            for path,key in ((ip,'initial'),(rp,'raw'),(bp,'prediction')):assert sha(path)==frozen[key+'_sha256']
            base=read(bp);initial=read(ip);before=time.monotonic()
            with np.load(rp,allow_pickle=False) as raw:
                graph,details=env['refine'](initial,base,raw['coords'],raw['edges'],model['weights'])
            validate_graph(graph,100);assert graph['nodes']==base['nodes']
            reference=ROOT/'.biohub/cache/trajectory-event-anchor-eight-v1'/(stem+'-prediction.json')
            assert sha(reference)==quality['inference'][stem]['prediction_sha256']
            assert graph==read(reference),'Portable replay differs: '+stem
            record=dict(stem=stem,exact_graph_identity=True,seconds=time.monotonic()-before,
                processed_frames=details['processed_frames'],budget_exhausted=details['budget_exhausted'],
                solver_fallbacks=details['solver_fallbacks'])
            report['records'].append(record);persist();print(json.dumps(record),flush=True)
        report['status']='portable_inference_verified';persist();print(json.dumps(dict(status=report['status'],seconds=report['seconds'])),flush=True)
    except BaseException as error:
        report.update(status='failed',error=repr(error));persist();raise


if __name__=='__main__':
    with threadpool_limits(limits=1,user_api='blas'):main()
