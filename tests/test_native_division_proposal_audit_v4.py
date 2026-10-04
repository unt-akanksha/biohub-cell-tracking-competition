import numpy as np
from research.native_correspondence_data_v2 import VOXEL
from research.native_division_proposal_audit_v4 import coverage,proposals_xy2


def movie():
    return dict(nodes=[[1,0,10,10,10],[2,1,10,30,10],[3,1,10,10,30]],edges=[[1,2],[1,3]])


def test_exact_and_missing_daughter():
    m=movie();p=np.array([m['nodes'][0][2:]],np.float32)*VOXEL;c=np.array([n[2:] for n in m['nodes'][1:]],np.float32)*VOXEL
    assert coverage(m,0,p,c)['counts']['eligible']==1
    result=coverage(m,0,p,c[:1]);assert result['counts']['missing_daughter_only']==1
    assert result['events'][0]['matched']==[True,True,False]


def test_displacement_not_reclassified_as_detection_failure():
    m=movie();m['nodes'][1][2]=40
    p=np.array([m['nodes'][0][2:]],np.float32)*VOXEL;c=np.array([n[2:] for n in m['nodes'][1:]],np.float32)*VOXEL
    assert coverage(m,0,p,c)['counts']['displacement_over_20um']==1


def test_xy2_image_only_geometry():
    image=np.zeros((64,128,128),np.float32);image[20,30,40]=10
    points=proposals_xy2(image)
    np.testing.assert_allclose(points,np.array([[20,60.5,80.5]])*VOXEL)
