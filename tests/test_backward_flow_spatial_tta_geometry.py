"""CPU geometry oracle; deliberately does not import local Torch."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research'))
from backward_flow_spatial_tta import inverse_components


@pytest.mark.parametrize('rotation',range(4))
@pytest.mark.parametrize('reflected',[False,True])
def test_inverse_components_against_transformed_point_displacements(rotation,reflected):
    shape=np.array([3,4,7]); parent=np.array([1.,1.,5.]); child=np.array([2.,2.,3.])
    # Independent coordinate-map oracle for torch.rot90(Y,X), then flip X.
    def point(value):
        z,y,x=value; height,width=shape[1:]
        for _ in range(rotation):
            y,x=width-1-x,y
            height,width=width,height
        if reflected: x=width-1-x
        return np.array([z,y,x])
    transformed=point(parent)-point(child)
    actual=np.array([transformed[i]*sign for i,sign in inverse_components(rotation,reflected)])
    np.testing.assert_array_equal(actual,parent-child)


def test_invalid_d4_indices_rejected():
    for rotation,reflected in [(4,False),(-1,True),(True,False),(0,1)]:
        with pytest.raises(ValueError): inverse_components(rotation,reflected)
