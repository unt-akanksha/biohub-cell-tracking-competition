import copy
import numpy as np
import pytest
from research.public_localization_projection import project_offsets, project_graph, SCALE, RADIUS_UM


def test_projection_bound_symmetry_and_idempotence():
    rng = np.random.default_rng(1729)
    offsets = rng.integers(-80,81,(500,3))
    answer = project_offsets(offsets)
    assert np.all(np.sum((answer*SCALE)**2,axis=1) <= RADIUS_UM**2)
    np.testing.assert_array_equal(project_offsets(answer),answer)
    np.testing.assert_array_equal(project_offsets(-offsets),-answer)
    np.testing.assert_array_equal(project_offsets(offsets[:,[0,2,1]]),answer[:,[0,2,1]])
    small=np.array([[0,0,0],[1,0,0],[0,4,0],[0,2,-2]])
    np.testing.assert_array_equal(project_offsets(small),small)


def test_graph_only_changes_large_drift_of_existing_detections():
    reference={'nodes':{'0':dict(node_id=0,t=0,z=20,y=40,x=40)}}
    final={'nodes':{'0':dict(node_id=0,t=0,z=25,y=48,x=42),
                    '1':dict(node_id=1,t=1,z=24,y=50,x=50)},
           'edges':[dict(source_id=0,target_id=1)]}
    before=copy.deepcopy(final)
    result,receipt=project_graph(final,reference)
    assert final == before
    assert result['nodes']['1'] == before['nodes']['1']
    assert result['edges'] == before['edges']
    assert receipt['changed_nodes']==1 and receipt['inserted_nodes_untouched']==1
    assert receipt['authorized_for_submission'] is False


@pytest.mark.parametrize('offsets',[[[float('nan'),0,0]],[[.5,0,0]],[1,2]])
def test_invalid_offsets_rejected(offsets):
    with pytest.raises(ValueError): project_offsets(offsets)
