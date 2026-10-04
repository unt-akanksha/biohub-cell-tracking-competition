from pathlib import Path
import runpy
import pytest

M=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/verify-focus-bridge-paired.py'))


def test_rejects_existing_node_mutation():
    c={'nodes':{'0':{'t':0,'x':1}},'edges':[]}
    b={'nodes':{'0':{'t':0,'x':2}},'edges':[]}
    with pytest.raises(ValueError,match='Existing nodes changed'):
        M['check_mutation'](c,b)


def test_bridge_and_edge_count_verified():
    c={'nodes':{'0':{'t':0},'1':{'t':2}},'edges':[]}
    b={'nodes':{**c['nodes'],'2':{'t':1,'x':1,'y':2,'z':3}},
       'edges':[{'source_id':0,'target_id':2},{'source_id':2,'target_id':1}]}
    assert M['check_mutation'](c,b)==1
    b['edges'].pop()
    with pytest.raises(ValueError,match='edge count'):
        M['check_mutation'](c,b)


def test_aggregate_handles_no_divisions_without_nan_score():
    row={'edge_tp':8,'edge_fp':1,'edge_fn':1,'adj_edge_jaccard':0.79,
         'division_tp':0,'division_fp':0,'division_fn':0}
    assert M['aggregate']([row])['score']==0.79
