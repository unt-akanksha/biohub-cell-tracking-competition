import numpy as np
import pytest
from research.blindspot_restoration import receptive_offsets,tiles,build_model


def test_symbolic_receptive_field_excludes_input_center():
    offsets = receptive_offsets()
    assert (0,0,0) not in offsets
    assert len(offsets) == 15**3-7**3
    assert all(any(v % 2 for v in p) for p in offsets)
    assert max(abs(v) for p in offsets for v in p) == 7


@pytest.mark.parametrize('shape,core',[((17,18,19),(5,7,8)),((1,2,3),(4,4,4)),((32,64,65),(16,32,32))])
def test_tiling_exactly_once_and_has_full_actual_halo(shape,core):
    coverage = np.zeros(shape,dtype=np.uint8)
    for tile in tiles(shape,core):
        coverage[tile['write']] += 1
        for n,read,write,crop in zip(shape,tile['read'],tile['write'],tile['crop']):
            assert read.start == max(0,write.start-7)
            assert read.stop == min(n,write.stop+7)
            assert crop.start+read.start == write.start
            assert crop.stop+read.start == write.stop
    assert np.all(coverage == 1)


@pytest.mark.parametrize('shape,core',[((0,2,3),(1,1,1)),((1,2),(1,1,1)),((2,2,2),(1,0,1)),((2,2.5,2),(1,1,1))])
def test_invalid_tile_geometry_fails(shape,core):
    with pytest.raises(ValueError):
        list(tiles(shape,core))


@pytest.mark.parametrize('width',[0,-1,2.5,True])
def test_invalid_model_width_rejected_without_importing_torch(width):
    with pytest.raises(ValueError):
        build_model(width)
