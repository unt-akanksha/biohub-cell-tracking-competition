"""Apply the audited sparse-label rule with disjoint whole-movie roles."""
from collections import Counter
import numpy as np
from research.focus_predicted_training_labels import targets


def movie_labels(coords,index_to_gt,truth_nodes,truth_edges,*,movie_role):
    if movie_role not in ('fitting','diagnostic'):raise ValueError('Exact whole-movie adaptation role required')
    records=[];counts=Counter()
    for t in range(99):
        row=targets(coords,index_to_gt,truth_nodes,truth_edges,t)
        # This NEW cache uses four/four disjoint movie roles, not the earlier
        # two-movie temporal partition. Label semantics are unchanged.
        row['role']=movie_role
        counts.update(row['status'])
        records.append({k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in row.items()})
    return dict(rows=records,counts=dict(counts),role=movie_role,frames=list(range(100)))


def node_matches(coords,truth):
    """Same DistanceMatching as the pinned scorer, without its edgeless shortcut."""
    import polars as pl
    import tracksdata as td
    graph=td.graph.InMemoryGraph()
    for axis in ('z','y','x'):graph.add_node_attr_key(axis,pl.Float64,0.)
    graph.bulk_add_nodes([dict(t=int(t),z=float(z),y=float(y),x=float(x)) for t,z,y,x in coords])
    from tracksdata.metrics import DistanceMatching
    from tracksdata.options import get_options,set_options
    previous=get_options().show_progress
    set_options(show_progress=False)
    try:
        graph.match(truth,matching=DistanceMatching(max_distance=7.,scale=(1.625,.40625,.40625)))
    finally:
        set_options(show_progress=previous)
    keys=td.DEFAULT_ATTR_KEYS
    nodes=graph.node_attrs().sort(keys.NODE_ID)
    if not np.array_equal(nodes.select('t','z','y','x').to_numpy(),coords):raise ValueError('Node matching changed raw geometry/order')
    return {i:g for i,g in enumerate(nodes[keys.MATCHED_NODE_ID])}
