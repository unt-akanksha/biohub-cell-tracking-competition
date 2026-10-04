import numpy as np
import pytest
from research.focus_cached_motion import assemble_flow


def fixture():
    coords=np.column_stack([np.arange(100),np.ones((100,3))]).astype(float)
    packets=[dict(source_frame=np.array(t),source_indices=np.array([t]),target_indices=np.array([t+1]),
        source_coords=np.ones((1,3),np.float32),target_coords=np.ones((1,3),np.float32),backward_um=np.full((1,3),t,np.float32)) for t in range(99)]
    return coords,packets


def test_complete_recovery_preserves_order_and_zero_initial_frame():
    coords,packets=fixture();flow=assemble_flow(coords,packets)
    assert flow.dtype==np.float32 and flow.shape==(100,3)
    np.testing.assert_array_equal(flow[0],0)
    np.testing.assert_array_equal(flow[1:,0],np.arange(99))


@pytest.mark.parametrize('failure',['missing','reordered','coordinate','nan'])
def test_corruption_fails_closed(failure):
    coords,packets=fixture()
    if failure=='missing':packets.pop()
    if failure=='reordered':packets[0],packets[1]=packets[1],packets[0]
    if failure=='coordinate':packets[1]['target_coords'][0,0]=2
    if failure=='nan':packets[1]['backward_um'][0,0]=np.nan
    with pytest.raises(ValueError):assemble_flow(coords,packets)
