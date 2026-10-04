"""CPU-only fixed-division LAP test on the verified complete source cache."""
import hashlib
import json
from pathlib import Path
import runpy
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.division_preserving_flow_assignment import link, forks, validate_edges
from research.focus_source_flow_comparison import compare
from research.backward_flow_linking import link_backward_flow

RUN = 'focus-division-preserving-assignment-v1'
REFERENCE_SHA = '896d3de86fa30fcd860f2c16073d47718118d94e1a43fe70710626c7b32ea169'
NOTEBOOK_SHA = 'd36fd6ef5fb460ae4964429aded2832b0d44db91a65836476ecea64639f30375'
SOURCE = runpy.run_path(str(ROOT / 'scripts/score-focus-source-flow.py'))
FILES = ['research/division_preserving_flow_assignment.py',
         'research/backward_flow_linking.py', 'research/independent_motion_prior.py',
         'research/focus_source_flow_comparison.py',
         'scripts/score-focus-division-preserving-assignment.py',
         'scripts/score-focus-source-flow.py',
         'reports/experiments/focus-division-preserving-assignment-v1-design.md']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def comparison(rows, summaries, reference):
    stems = reference['contract']['source_stems']
    if rows['control'] != reference['per_movie']['candidate'] or summaries['control'] != reference['summaries']['candidate']:
        raise ValueError('Exact original FOCUS-flow complete-score replay required')
    if rows['parent'] != reference['per_movie']['parent'] or summaries['parent'] != reference['summaries']['parent']:
        raise ValueError('Exact original parent replay required')
    source = compare(dict(parent=rows['parent'], control=reference['per_movie']['control'], candidate=rows['candidate']),
                     dict(parent=summaries['parent'], control=reference['summaries']['control'], candidate=summaries['candidate']), stems)
    deltas = {k: summaries['candidate'][k]-summaries['control'][k] for k in ('score', 'edge_jaccard', 'node_recall')}
    per_movie = [dict(stem=a['stem'], adjusted_edge_delta=a['adj_edge_jaccard']-b['adj_edge_jaccard'])
                 for a, b in zip(rows['candidate'], rows['control'])]
    conditions = dict(original_source_gate=source['source_gate_passed'],
        score_gain_over_focus_flow=deltas['score'] > 0,
        raw_edge_gain_over_focus_flow=deltas['edge_jaccard'] > 0,
        recall_identical=deltas['node_recall'] == 0,
        per_movie_loss_bounded=min(r['adjusted_edge_delta'] for r in per_movie) >= -.02-1e-12,
        true_divisions_preserved=summaries['candidate']['division_tp'] >= summaries['control']['division_tp'])
    return dict(source_comparison=source, focus_flow_deltas=deltas, focus_flow_per_movie_deltas=per_movie,
                conditions=conditions, source_gate_passed=all(conditions.values()), authorized_for_submission=False)


def graph_from_arrays(coords, edges):
    import polars as pl
    import tracksdata as td
    graph = td.graph.InMemoryGraph()
    for axis in ('z', 'y', 'x'):
        graph.add_node_attr_key(axis, pl.Float64, 0.)
    ids = graph.bulk_add_nodes([dict(t=int(t), z=float(z), y=float(y), x=float(x)) for t,z,y,x in coords])
    if len(edges):
        graph.bulk_add_edges([dict(source_id=ids[int(s)], target_id=ids[int(d)]) for s,d in edges])
    return graph


def main():
    import numpy as np
    import tracksdata as td
    from geff import GeffMetadata
    started = time.monotonic()
    target = ROOT / f'reports/experiments/{RUN}-result.json'
    cache = ROOT / '.biohub/cache' / RUN
    if target.exists() or cache.exists():
        raise ValueError('Existing run artifacts: do not overwrite or duplicate')
    frozen = {name: sha(ROOT/name) for name in FILES}
    reference_path = ROOT / 'reports/experiments/focus-source-flow-v1-result.json'
    if sha(reference_path) != REFERENCE_SHA:
        raise ValueError('Frozen source result changed')
    reference = json.loads(reference_path.read_text())
    folder = ROOT / '.biohub/cache/kernel-outputs/focus-source-flow-v1/focus_source_flow'
    notebook = ROOT / 'kaggle/biohub-focus-source-flow-v1/biohub-focus-source-flow-v1.ipynb'
    prepared, parent_graphs, policy = SOURCE['prepare'](folder, notebook, NOTEBOOK_SHA)
    cache.mkdir()
    records = []
    # Persist every prediction before opening the already-exposed source labels.
    for stem in policy['source_stems']:
        sample = folder / 'outputs' / stem / 'sampled_flow.npz'
        with np.load(sample, allow_pickle=False) as data:
            coords, flow = data['coords'].copy(), data['backward_um'].copy()
        edges, receipt = link(coords, flow)
        candidate = cache / (stem+'.npz')
        np.savez_compressed(candidate, coords=coords, edges=np.asarray(edges, dtype=np.int64).reshape(-1,2))
        records.append(dict(stem=stem, sample_sha256=sha(sample), candidate_sha256=sha(candidate),
                            nodes=len(coords), frames=100, **receipt))
    manifest = dict(run_id=RUN, source_hashes=frozen, reference_sha256=REFERENCE_SHA,
                    notebook_sha256=NOTEBOOK_SHA, records=records, all_predictions_saved_before_gt=True,
                    previously_exposed_source_movies=True, new_target_movies_opened=0)
    manifest_path = cache / 'prelabel_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2))
    arrays = {}
    for record in records:
        stem = record['stem']
        path = cache/(stem+'.npz')
        if sha(path) != record['candidate_sha256']:
            raise ValueError('Persisted prediction identity changed')
        with np.load(path, allow_pickle=False) as data:
            coords, edges = data['coords'].copy(), data['edges'].copy()
        with np.load(folder/'outputs'/stem/'sampled_flow.npz', allow_pickle=False) as data:
            if not np.array_equal(coords, data['coords']):
                raise ValueError('Candidate deleted or moved an actual detector node')
            original = [(s,d) for s,d,_ in link_backward_flow(coords, data['backward_um'])]
        validate_edges(coords, list(map(tuple, edges)), forks(original))
        graph = graph_from_arrays(coords, edges)
        SOURCE['verify_graph'](graph, coords, [(int(s),int(d),0.) for s,d in edges])
        arrays[stem] = (coords, edges)
    if frozen != {name: sha(ROOT/name) for name in FILES}:
        raise ValueError('Experiment source changed during inference')
    metric = SOURCE['SCORER']['load_scorer'](ROOT/'.biohub/vendor/kaggle-cell-tracking-competition/src/tracking_cellmot')
    rows = {a: [] for a in ('parent','control','candidate')}
    for stem in policy['source_stems']:
        gt_path = ROOT/'.biohub/cache/competition-train-geffs-packed-v1/train'/(stem+'.geff')
        total = float(GeffMetadata.read(str(gt_path)).extra['estimated_number_of_nodes'])
        for arm in rows:
            if arm == 'parent':
                graph = parent_graphs[stem]
            elif arm == 'control':
                graph = td.graph.IndexedRXGraph.from_geff(str(prepared[stem]['candidate']))[0]
            else:
                graph = graph_from_arrays(*arrays[stem])
            truth = td.graph.IndexedRXGraph.from_geff(str(gt_path))[0]
            er = metric.evaluate(graph, truth, scale=(1.625,.40625,.40625), max_distance=7.)
            row = dict(metric.per_sample_metrics(er, total, SOURCE['SCORER']['diagnostic_node_recall'](metric,graph,truth)),
                       stem=stem, embryo=stem.split('_')[0])
            rows[arm].append(row)
            print(json.dumps(dict(arm=arm, **row)), flush=True)
    summaries = {a: metric.summarise(r) for a,r in rows.items()}
    result = dict(status='completed_fixed_division_source_assignment',run_id=RUN,per_movie=rows,summaries=summaries,
        by_embryo={a:{'6bba':metric.summarise(r)} for a,r in rows.items()},
        comparison=comparison(rows,summaries,reference), records=records,
        source_hashes=frozen, prelabel_manifest_sha256=sha(manifest_path), reference_sha256=REFERENCE_SHA,
        authoritative_scorer_commit='075fc5f5a52d11077f9dc2b074644618f26939e2',
        elapsed_seconds=time.monotonic()-started, gpu_seconds=0, new_target_movies_opened=0,
        authorized_for_submission=False, all_prediction_nodes_and_forks_preserved=True,
        caveat='Previously exposed eight source movies; FOCUS pretraining overlap unverified. Not leaderboard evidence.')
    if frozen != {name: sha(ROOT/name) for name in FILES}:
        raise ValueError('Experiment source changed during scoring')
    target.write_text(json.dumps(SOURCE['FULL']['finite_json'](result),indent=2,allow_nan=False))
    print(json.dumps(result['comparison']),flush=True)


if __name__ == '__main__':
    main()
