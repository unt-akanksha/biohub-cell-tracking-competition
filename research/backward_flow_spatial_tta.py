"""D4-average physical displacement fields, not scalar heatmaps.

XY grid transforms must also rotate/reflect the Y/X vector components.
Detections, graph rules and frozen flow weights are outside this module.
"""
from edge_feature_tta import D4_VIEWS,transform,invert


def inverse_components(rotation,reflected):
    """Native output component -> transformed input (index, sign), Z/Y/X."""
    if type(rotation) is not int or rotation not in range(4) or type(reflected) is not bool:
        raise ValueError('One of eight registered D4 views required')
    y=(1,1); x=(2,-1 if reflected else 1)
    for _ in range(rotation):
        y,x=x,(y[0],-y[1])
    return ((0,1),y,x)


def invert_vector_field(field,rotation,reflected):
    import torch
    if field.ndim!=5 or field.shape[1]!=3 or not field.is_floating_point():
        raise ValueError('Floating B,3,Z,Y,X physical vector field required')
    aligned=invert(field,rotation,reflected)
    return torch.stack([aligned[:,index]*sign for index,sign in inverse_components(rotation,reflected)],dim=1)


def install_backward_flow_spatial_tta(flow):
    import torch
    if flow.training or hasattr(flow,'_biohub_flow_tta'):
        raise ValueError('Native evaluation-only flow required')
    flow.requires_grad_(False)
    original=flow.forward
    receipt=dict(version=1,views=8,forward_calls=0,
        components='ZYX physical microns; inverse XY vector basis and grid',
        output_precision='FP32 arithmetic mean',maximum_mean_absolute_flow_delta_um=0.)

    def forward(images):
        if flow.training or images.ndim!=5 or images.shape[1]!=2:
            raise ValueError('Evaluation-only previous/current image pairs required')
        with torch.no_grad():
            native=original(images)
            if native.shape!=(images.shape[0],3,*images.shape[2:]):
                raise ValueError('Native flow grid mismatch')
            total=native.float().clone()
            for rotation,reflected in D4_VIEWS[1:]:
                predicted=original(transform(images,rotation,reflected))
                restored=invert_vector_field(predicted,rotation,reflected)
                if restored.shape!=native.shape or restored.device!=native.device:
                    raise ValueError('Inverse vector field grid/device mismatch')
                total.add_(restored.float())
            average=total/8
            if not torch.isfinite(average).all(): raise ValueError('Nonfinite averaged flow')
            delta=float((average-native.float()).abs().mean())
        receipt['forward_calls']+=1
        receipt['maximum_mean_absolute_flow_delta_um']=max(receipt['maximum_mean_absolute_flow_delta_um'],delta)
        return average

    flow.forward=forward
    flow._biohub_flow_tta=receipt
    return receipt
