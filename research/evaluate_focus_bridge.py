"""Notebook-stage paired evaluation; executed after base inference."""
import hashlib
import inspect
import json
from pathlib import Path
import sys

# The CLI worker sets PYTHONPATH=src, but the notebook process does not.
_official_source_root = REPO_DIR / 'src'
if not (_official_source_root / 'biohub_tracking/metrics.py').is_file():
    raise FileNotFoundError('Materialized official scorer source is missing')
sys.path.insert(0, str(_official_source_root))

import numpy as np
import pandas as pd
import polars as pl
import tracksdata as td
import zarr
import biohub_tracking.metrics as official
import biohub_tracking.division_metrics as division_official
from biohub_tracking.io import open_dataset
from geff import GeffMetadata


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_hash(root):
    h = hashlib.sha256()
    for path in sorted(root.rglob('*')):
        if path.is_file():
            h.update(path.relative_to(root).as_posix().encode())
            h.update(b'\0'); h.update(path.read_bytes()); h.update(b'\0')
    return h.hexdigest()


assert file_hash(inspect.getfile(official)) == '31baf45b54c78f68bab4f65dd8f4b38bca702abb644171c6df7c46cdeef55d83'
assert file_hash(inspect.getfile(division_official)) == 'd1cf1e0a43009d02174f1699ce2aa28458a2220ac4b521731d3bcf31cf8c76be'
_proposal_receipts = [Path('/kaggle/input') / prefix / 'focus3d_bridge_proposals_terminal.json'
                     for prefix in ('biohub-focus3d-bridge-proposals-v1',
                         'notebooks/indarkarhana/biohub-focus3d-bridge-proposals-v1',
                         'kernels/indarkarhana/biohub-focus3d-bridge-proposals-v1')]
receipts = [p for p in _proposal_receipts if p.is_file()
            and file_hash(p) == '288eab6b9e2c568f38587fa51c0af0c2dd140443df8a890a05f8cf2d290ee985']
assert len(receipts) == 1, receipts
receipt = json.loads(receipts[0].read_text())
assert receipt['status'] == 'completed' and receipt['ground_truth_opened'] is False
assert set(receipt['frozen_stems']) == set(test_stems)
control_csv = pd.read_csv(SUBMISSION_PATH)
assert DEEPCENTER_VETO_DETECTOR is not None
output = Path('/kaggle/working/paired_graphs'); output.mkdir(exist_ok=True)
prepared = {}

for stem in test_stems:
    proposal_path = receipts[0].parent / 'my_predict' / f'{stem}.geff'
    assert tree_hash(proposal_path) == receipt['proposal_graph_sha256'][stem]
    graph = td.graph.IndexedRXGraph.from_geff(str(proposal_path))[0]
    shape = tuple(zarr.open_group(str(TEST_DIR / f'{stem}.zarr'), mode='r')['0'].shape)
    focus_nodes = {int(row['node_id']): {k: row[k] for k in ('node_id', 't', 'z', 'y', 'x')}
                   for row in graph.node_attrs().iter_rows(named=True)}
    focus_edges = [{'source_id': int(row['source_id']), 'target_id': int(row['target_id'])}
                   for row in graph.edge_attrs().iter_rows(named=True)]
    frame = control_csv[control_csv.dataset == stem]
    nodes = {int(row.node_id): {'node_id': int(row.node_id), 't': int(row.t),
             'z': float(row.z), 'y': float(row.y), 'x': float(row.x)}
             for row in frame[frame.row_type == 'node'].itertuples()}
    edges = [{'source_id': int(row.source_id), 'target_id': int(row.target_id)}
             for row in frame[frame.row_type == 'edge'].itertuples()]
    probabilities = {}; frame_cache = {}; heatmap_cache = {}
    for node_id, node in sorted(focus_nodes.items(), key=lambda item: (item[1]['t'], item[0])):
        # Recovery revision: keep proposal geometry/topology intact, but do
        # not confirm smoothed points outside the actual image extent.
        if not all(0 <= node[k] < limit for k, limit in zip(('t','z','y','x'), shape)):
            continue
        value = deepcenter_score_point(stem, int(node['t']), tuple(node[k] for k in ('z','y','x')),
                                      DEEPCENTER_VETO_DETECTOR, frame_cache, heatmap_cache)
        if value is not None:
            probabilities[node_id] = value
    candidate_nodes, candidate_edges, stats = apply_focus_bridge(nodes, edges, focus_nodes, focus_edges, probabilities)
    assert all(candidate_nodes[key] == node for key, node in nodes.items())
    assert candidate_edges[:len(edges)] == edges
    for key in set(candidate_nodes) - set(nodes):
        node = candidate_nodes[key]
        for axis, limit in zip(('z','y','x'), shape[1:]):
            node[axis] = max(0, int(round(node[axis])))
            assert node[axis] < limit, (stem, key, axis)
    paths = {}
    for arm, arm_nodes, arm_edges in [('control', nodes, edges), ('bridge', candidate_nodes, candidate_edges)]:
        path = output / f'{stem}-{arm}.json'
        path.write_text(json.dumps({'nodes': arm_nodes, 'edges': arm_edges}, sort_keys=True))
        paths[arm] = {'path': str(path), 'sha256': file_hash(path)}
    prepared[stem] = {'graphs': paths, 'stats': stats}

Path('/kaggle/working/prelabel_graph_manifest.json').write_text(json.dumps(prepared, indent=2, sort_keys=True))


def load_fresh_graph(path):
    payload = json.loads(Path(path).read_text()); graph = td.graph.InMemoryGraph()
    for axis in ('z', 'y', 'x'):
        graph.add_node_attr_key(axis, pl.Float64, 0.0)
    original = sorted(payload['nodes'], key=int)
    new_ids = graph.bulk_add_nodes([{k: payload['nodes'][key][k] for k in ('t','z','y','x')} for key in original])
    mapping = dict(zip(map(int, original), new_ids))
    if payload['edges']:
        graph.bulk_add_edges([{'source_id': mapping[e['source_id']], 'target_id': mapping[e['target_id']]}
                              for e in payload['edges']])
    return graph


# Ground truth is first loaded here, after both arms for every movie are persisted.
rows = {'control': [], 'bridge': []}
for stem in test_stems:
    for arm in rows:
        record = prepared[stem]['graphs'][arm]
        assert file_hash(record['path']) == record['sha256']
        graph = load_fresh_graph(record['path'])
        ds = open_dataset(str(COMP_DIR / 'train' / f'{stem}.zarr'), normalize=False, load_image=False, require_tracks=True)
        er = official.evaluate(graph, ds.tracks, scale=ds.scale, max_distance=7.0)
        meta = GeffMetadata.read(str(COMP_DIR / 'train' / f'{stem}.geff'))
        row = official.per_sample_metrics(er, float(meta.extra['estimated_number_of_nodes']), official.node_recall(graph, ds.tracks))
        rows[arm].append({**row, 'stem': stem, 'embryo': stem.split('_')[0]})
summaries = {arm: official.summarise(values) for arm, values in rows.items()}
deltas = {c['stem']: b['adj_edge_jaccard'] - c['adj_edge_jaccard'] for c,b in zip(rows['control'],rows['bridge'])}
tp_gain = sum(r['edge_tp'] for r in rows['bridge']) - sum(r['edge_tp'] for r in rows['control'])
div_ok = (summaries['bridge']['division_jaccard'] >= summaries['control']['division_jaccard']
          or all(np.isnan(summaries[a]['division_jaccard']) for a in summaries))
passed = bool(summaries['bridge']['score'] > summaries['control']['score'] and min(deltas.values()) >= 0 and tp_gain > 0 and div_ok)
result = {'run_id': 'focus-bridge-official-paired-v2', 'status': 'completed', 'summaries': summaries,
          'per_movie': rows, 'per_movie_adjusted_delta': deltas, 'edge_tp_gain': tp_gain,
          'by_embryo': {arm: {e: official.summarise([r for r in values if r['embryo']==e]) for e in ('44b6','6bba')} for arm,values in rows.items()},
          'paired_gate_passed': passed, 'authorized_for_submission': False,
          'requires_production_validation': True, 'public_predictions_copied': False,
          'metric_hack_used': False, 'prepared': prepared}
Path('/kaggle/working/bridge_paired_result.json').write_text(json.dumps(result, indent=2, sort_keys=True))
print(json.dumps(result, indent=2, sort_keys=True))
_LC_FINISHED = True
_LC_TIMER.cancel()
_lc_write_terminal('completed')
