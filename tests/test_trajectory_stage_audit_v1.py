import pytest
from research.trajectory_stage_audit_v1 import attribute


def test_provenance_and_missing_link_reasons_are_exhaustive():
    nodes = {0:dict(original=10,matched=1),1:dict(original=11,matched=2),2:dict(original=12,matched=3)}
    edges = [dict(source=0,target=1,matched=True,valid=True),dict(source=0,target=2,matched=False,valid=True)]
    result = attribute(nodes,edges,[(1,2),(2,3),(3,4)],[(10,11)])
    assert result['counts'] == dict(in_initial_graph_tp=1,introduced_after_initial_graph_fp=1,
                                   fn_both_endpoints_matched=1,fn_endpoint_unmatched=1)
    assert result['recovered_gt_edges'] == [(1,2)]


def test_duplicate_official_match_cannot_inflate_attribution():
    nodes={0:dict(original=0,matched=0),1:dict(original=1,matched=1)}
    edge=dict(source=0,target=1,matched=True,valid=True)
    with pytest.raises(ValueError):
        attribute(nodes,[edge,edge],[(0,1)],[(0,1)])
