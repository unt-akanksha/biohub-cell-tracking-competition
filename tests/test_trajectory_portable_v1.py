import ast
import copy
import csv
import json
from pathlib import Path

import pytest

from research.trajectory_portable_adapter_v1 import portable_source
from research.trajectory_runtime_v1 import (assemble_csv, graph_rows, image_metadata,
                                           validate_graph, validate_movie_ids, verify_bundle, sha)

ROOT = Path(__file__).resolve().parents[1]


def tiny_graph():
    return dict(nodes={str(t):dict(node_id=t,t=t,z=1,y=2,x=3) for t in range(8)},
                edges=[dict(source_id=t,target_id=t+1) for t in range(7)])


def test_adapter_preserves_expensive_inference_and_repair():
    original = (ROOT/'scripts/run-trajectory-division-full-movie-v1.py').read_text()
    adapted = portable_source(original)
    a = original.index('            cfg = env[\'PredictConfig\']')
    b = original.index('    if any(helper.sha',a)
    block = original[a:b].replace("args.images / 'train'", 'args.images')
    assert block in adapted
    assert "stems = args.movies" in adapted and 'helper.STEMS' not in adapted
    assert 'verify_idle_gpu_query' not in adapted
    assert 'td_ilp.Solver = ScipSolver' in adapted
    ast.parse(adapted)


def test_model_initialization_unchanged():
    original = (ROOT/'scripts/run-trajectory-division-full-movie-v1.py').read_text()
    start = original.index('    temporal = load_module')
    end = original.index('    result.update(device=',start)
    assert original[start:end] in portable_source(original)


def test_csv_roundtrip_and_exact_unused_fields(tmp_path):
    graph = tiny_graph()
    paths = {}
    for m in ('new_embryo_a','new_embryo_b'):
        paths[m] = tmp_path/(m+'.json'); paths[m].write_text(json.dumps(graph))
    result = assemble_csv(paths,{m:8 for m in paths},tmp_path/'submission.csv')
    assert result['rows'] == 30
    with (tmp_path/'submission.csv').open() as f:
        rows = list(csv.DictReader(f))
    for m in paths:
        sub = [r for r in rows if r['dataset']==m]
        nodes = {r['node_id']:{k:int(r[k]) for k in ('node_id','t','z','y','x')}
                 for r in sub if r['row_type']=='node'}
        edges = [{k:int(r[k]) for k in ('source_id','target_id')}
                 for r in sub if r['row_type']=='edge']
        assert dict(nodes=nodes,edges=edges) == graph
        assert all(r['source_id']=='-1' and r['target_id']=='-1' for r in sub if r['row_type']=='node')
        assert all(r['node_id']=='-1' and r['t']=='-1' for r in sub if r['row_type']=='edge')


@pytest.mark.parametrize('kind',['missing_frame','duplicate_edge','dangling','merge','nan','float_id'])
def test_fail_closed_bad_graph(kind):
    graph = tiny_graph()
    if kind=='missing_frame': del graph['nodes']['7']
    if kind=='duplicate_edge': graph['edges'].append(graph['edges'][0].copy())
    if kind=='dangling': graph['edges'][0]['target_id']=999
    if kind=='merge':
        graph['nodes']['99']=dict(node_id=99,t=0,z=1,y=1,x=1)
        graph['edges'].append(dict(source_id=99,target_id=1))
    if kind=='nan': graph['nodes']['0']['x']=float('nan')
    if kind=='float_id': graph['edges'][0]['target_id']=1.0
    with pytest.raises(ValueError): validate_graph(graph,8)


def test_no_final_csv_on_incomplete_coverage(tmp_path):
    with pytest.raises(ValueError): assemble_csv({}, {'movie':8}, tmp_path/'submission.csv')
    assert not (tmp_path/'submission.csv').exists()


def test_input_movie_ids_not_embryo_routed():
    validate_movie_ids(['never_seen_123','other-embryo'])
    for bad in (['../train'], ['x','x'], []):
        with pytest.raises(ValueError): validate_movie_ids(bad)


def test_manifest_tampering_rejected(tmp_path):
    p=tmp_path/'runtime.py'; p.write_text('pass')
    c=tmp_path/'CONTRACT.json'
    c.write_text(json.dumps(dict(ground_truth_included=False,
        public_prediction_tables_included=False,bundle_sha256={p.name:sha(p)})))
    verify_bundle(tmp_path,sha(c))
    p.write_text('changed')
    with pytest.raises(ValueError): verify_bundle(tmp_path,sha(c))


def test_generic_time_count_preserves_spatial_scale(tmp_path):
    (tmp_path/'0').mkdir()
    (tmp_path/'zarr.json').write_text(json.dumps(dict(attributes=dict(
        multiscales=[dict(datasets=[dict(coordinateTransformations=[dict(scale=[1,1.625,.40625,.40625])])])],
        image_statistics=dict(quantiles=[1,2])))))
    (tmp_path/'0/zarr.json').write_text(json.dumps(dict(shape=[120,64,256,256],data_type='uint16')))
    result=image_metadata(tmp_path)
    assert result.image_shape==(120,64,64,64)
    assert result.scale==(1.625,.40625,.40625)


@pytest.mark.parametrize('layout',['flat','namespaced'])
def test_bootstrap_expanded_dataset_without_recursive_scan(tmp_path,layout):
    input_root=tmp_path/'input'; working=tmp_path/'working';working.mkdir()
    prefix=input_root if layout=='flat' else input_root/'datasets/indarkarhana'
    runtime=prefix/'biohub-trajectory-motion-runtime-v1';runtime.mkdir(parents=True)
    (runtime/'tiny.py').write_text('pass\n')
    contract=dict(ground_truth_included=False,public_prediction_tables_included=False,
                  bundle_sha256={'tiny.py':sha(runtime/'tiny.py')})
    (runtime/'CONTRACT.json').write_text(json.dumps(contract))
    code=(ROOT/'scripts/trajectory-kaggle-bootstrap-v1.py').read_text()
    code=code[:code.index('sys.path.insert(0, str(bundle))')]
    assert "glob('**/" not in code
    code=code.replace("Path('/kaggle/input')",f'Path({str(input_root)!r})')
    code=code.replace("Path('/kaggle/working')",f'Path({str(working)!r})')
    code=code.replace('__RUN_MODE__','acceptance').replace('__CONTRACT_SHA256__',sha(runtime/'CONTRACT.json'))
    exec(compile(code,'tested-bootstrap','exec'),{})
    assert (working/'trajectory-runtime-v1/tiny.py').read_text()=='pass\n'
