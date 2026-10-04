from pathlib import Path
import runpy
import numpy as np
import pytest
from research.focus_adaptation_labels import movie_labels,node_matches


def test_whole_movie_role_does_not_inherit_earlier_temporal_embargo():
    coords=np.array([[79,1,1,1],[80,1,1,1]],float)
    truth={10:coords[0],11:coords[1]}
    for role in ('fitting','diagnostic'):
        result=movie_labels(coords,{0:10,1:11},truth,[(10,11)],movie_role=role)
        assert len(result['rows'])==99 and {r['role'] for r in result['rows']}=={role}
        assert result['rows'][79]['labels']==[0]
        assert result['counts']['known_parent']==1
    with pytest.raises(ValueError):movie_labels(coords,{0:10,1:11},truth,[(10,11)],movie_role='embargo')


def test_real_patched_scorer_edgeless_node_matching():
    import polars as pl
    import tracksdata as td
    root=Path(__file__).resolve().parents[1]
    metric=runpy.run_path(str(root/'scripts/score-independent-selection.py'))['load_scorer'](root/'.biohub/vendor/kaggle-cell-tracking-competition/src/tracking_cellmot')
    truth=td.graph.InMemoryGraph()
    for axis in ('z','y','x'):truth.add_node_attr_key(axis,pl.Float64,0.)
    ids=truth.bulk_add_nodes([dict(t=0,z=1.,y=1.,x=1.),dict(t=1,z=1.,y=1.,x=1.)])
    truth.bulk_add_edges([dict(source_id=ids[0],target_id=ids[1])])
    coords=np.array([[0,1,1,1],[1,1,1,1],[1,50,50,50]],float)
    mapping=node_matches(coords,truth)
    assert mapping[0]==ids[0] and mapping[1]==ids[1] and mapping[2] in (None,-1)
    result=movie_labels(coords,mapping,{ids[0]:coords[0],ids[1]:coords[1]},[(ids[0],ids[1])],movie_role='fitting')
    assert result['rows'][0]['labels']==[0,-1]
    control=td.graph.InMemoryGraph()
    for axis in ('z','y','x'):control.add_node_attr_key(axis,pl.Float64,0.)
    control_ids=control.bulk_add_nodes([dict(t=int(t),z=z,y=y,x=x) for t,z,y,x in coords])
    control.bulk_add_edges([dict(source_id=control_ids[0],target_id=control_ids[1])])
    metric.evaluate(control,truth,scale=(1.625,.40625,.40625),max_distance=7.)
    expected=dict(enumerate(control.node_attrs().sort('node_id')[td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID]))
    assert mapping==expected
