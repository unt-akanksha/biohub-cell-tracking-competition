"""Read-only, label-free audit of downloaded bridge recovery inputs."""
from pathlib import Path
import hashlib
import json
import sys

import pandas as pd
import zarr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.focus3d_bridge_rescue import _validate_graph

csv_path = ROOT / '.biohub/cache/kernel-outputs/focus-bridge-official-paired-v2/control_validation.csv'
assert hashlib.sha256(csv_path.read_bytes()).hexdigest() == 'f3766c0d9f4212020b99c59b26aa13f9a5fde245ba842cd5e38ea4604db5eed4'
table = pd.read_csv(csv_path)
assert table.id.is_unique
assert set(table.row_type) == {'node', 'edge'}
root = ROOT / '.biohub/cache/kernel-outputs/focus3d-bridge-proposals-v1/my_predict'
assert set(table.dataset) == {p.stem for p in root.glob('*.geff')}
for path in sorted(root.glob('*.geff')):
    group = zarr.open_group(str(path), mode='r')
    ids = group['nodes/ids'][:]
    props = {k: group[f'nodes/props/{k}/values'][:] for k in ('t', 'z', 'y', 'x')}
    nodes = {int(n): {'node_id': int(n), **{k: float(v[i]) for k, v in props.items()}}
             for i, n in enumerate(ids)}
    edges = [{'source_id': int(s), 'target_id': int(t)} for s, t in group['edges/ids'][:]]
    _validate_graph(nodes, edges, proposal_coordinates=True)
    frame = table[table.dataset == path.stem]
    control = {int(r.node_id): dict(node_id=int(r.node_id), t=int(r.t), z=r.z, y=r.y, x=r.x)
               for r in frame[frame.row_type == 'node'].itertuples()}
    control_edges = [dict(source_id=int(r.source_id), target_id=int(r.target_id))
                     for r in frame[frame.row_type == 'edge'].itertuples()]
    _validate_graph(control, control_edges)
    print(json.dumps(dict(stem=path.stem, focus_nodes=len(nodes), focus_edges=len(edges),
                          control_nodes=len(control), control_edges=len(control_edges),
                          graph_contract_passed=True)))
