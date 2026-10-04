"""Training-only label coverage for a possible broader fit; no model launch."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run():
    import tracksdata as td
    split_path = ROOT/'research/independent_real_baseline_v1_split.json'
    split = json.loads(split_path.read_text())
    fold = split['folds'][0]
    stems = fold['train']
    if set(stems)&set(fold['selection']+fold['audit_order']):
        raise ValueError('Training inventory intersects selection/audit')
    rows = []
    for stem in stems:
        if not stem.startswith(fold['training_embryo']+'_'):
            raise ValueError('Target embryo in training inventory')
        path = ROOT/'.biohub/cache/competition-train-geffs-packed-v1/train'/(stem+'.geff')
        graph = td.graph.IndexedRXGraph.from_geff(str(path))[0]
        nodes = {int(r['node_id']):r for r in graph.node_attrs().iter_rows(named=True)}
        degree = {}
        for edge in graph.edge_attrs().iter_rows(named=True):
            source,target = int(edge['source_id']),int(edge['target_id'])
            if nodes[target]['t'] != nodes[source]['t']+1:
                continue
            degree[source] = degree.get(source,0)+1
        rows.append(dict(stem=stem,annotated_nodes=len(nodes),
            divisions=sum(n == 2 for n in degree.values()),
            single_child=sum(n == 1 for n in degree.values())))
    return dict(scope='Frozen fold-zero train list only',movies=len(rows),per_movie=rows,
        division_events=sum(r['divisions'] for r in rows),
        movies_with_divisions=sum(r['divisions'] > 0 for r in rows),
        single_child_events=sum(r['single_child'] for r in rows),
        split_sha256=hashlib.sha256(split_path.read_bytes()).hexdigest(),
        selection_opened=False,target_audit_opened=False,model_changed=False)


if __name__ == '__main__':
    print(json.dumps(run(),indent=2))
