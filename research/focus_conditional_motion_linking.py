"""Label-free conditional mean correction with the unchanged edge policy."""
import numpy as np
from research.focus_conditional_motion import predict
from research.focus_residual_calibration import link as gaussian_link


def link(coords,flow,model):
    points=np.asarray(coords,float);motion=np.asarray(flow,float)
    if points.ndim!=2 or points.shape[1]!=4 or motion.shape!=(len(points),3) or not np.isfinite(points).all() or not np.isfinite(motion).all():raise ValueError('Complete finite geometry and aligned flow required')
    correction=predict(model,np.column_stack([motion,points[:,1:]]))
    corrected=motion+correction;corrected[points[:,0]==0]=0.
    if not np.isfinite(corrected).all():raise ValueError('Nonfinite predicted parent displacement')
    return gaussian_link(points,corrected,dict(mean_um=[0.,0.,0.],variance_um2=model['variance_um2']))
