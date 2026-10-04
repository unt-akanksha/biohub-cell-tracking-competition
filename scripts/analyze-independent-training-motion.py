"""Motion diagnostics on only the four recorded fitting movies; no model edits."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCALE = np.asarray([1.625,.40625,.40625])


def summarize(displacements):
    values = np.asarray(displacements,dtype=float).reshape(-1,3)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Finite training displacements required')
    lengths = np.linalg.norm(values,axis=1)
    return dict(n=len(values),median_delta_um=np.median(values,axis=0).tolist(),
        axis_std_um=values.std(axis=0).tolist(),
        distance_quantiles_um=dict(zip(('p50','p90','p95','p99'),
            np.quantile(lengths,[.5,.9,.95,.99]).tolist())))


def run():
    import tracksdata as td
    split = json.loads((ROOT/'research/independent_real_baseline_v1_split.json').read_text())
    stems = split['folds'][0]['train'][:4]
    assert not set(stems)&set(split['folds'][0]['selection']+split['folds'][0]['audit_order'])
    per_movie,pooled = {},[]
    for stem in stems:
        path = ROOT/'.biohub/cache/competition-train-geffs-packed-v1/train'/(stem+'.geff')
        graph = td.graph.IndexedRXGraph.from_geff(str(path))[0]
        nodes = {int(r['node_id']):r for r in graph.node_attrs().iter_rows(named=True)}
        edges = list(graph.edge_attrs().iter_rows(named=True))
        degree = {}
        for edge in edges:
            source = int(edge['source_id'])
            degree[source] = degree.get(source,0)+1
        deltas = []
        for edge in edges:
            source,target = nodes[int(edge['source_id'])],nodes[int(edge['target_id'])]
            if degree[int(edge['source_id'])] != 1 or target['t'] != source['t']+1:
                continue
            deltas.append([(target[k]-source[k])*s for k,s in zip(('z','y','x'),SCALE)])
        per_movie[stem] = summarize(deltas)
        per_movie[stem]['annotated_division_parents'] = sum(value == 2 for value in degree.values())
        per_movie[stem]['annotated_single_child_parents'] = sum(value == 1 for value in degree.values())
        per_movie[stem]['annotated_nodes'] = len(nodes)
        pooled.extend(deltas)
    return dict(scope='Four fitting movies only, annotated nondivision one-frame edges',
        per_movie=per_movie,pooled=summarize(pooled),selection_opened=False,
        division_events=sum(row['annotated_division_parents'] for row in per_movie.values()),
        single_child_events=sum(row['annotated_single_child_parents'] for row in per_movie.values()),
        target_audit_opened=False,model_changed=False,authorized_for_submission=False)


if __name__ == '__main__':
    print(json.dumps(run(),indent=2))
