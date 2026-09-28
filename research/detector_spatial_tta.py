"""Eight-view detector averaging, installed inside the native image-flow cache."""
from edge_feature_tta import D4_VIEWS,transform,invert


def install_detector_spatial_tta(model):
    import torch
    if model.training or hasattr(model,'_image_motion_mode') or hasattr(model,'_biohub_detector_tta'):
        raise ValueError('Install detector TTA in evaluation mode before the image-flow wrapper')
    original = model.encode
    receipt = dict(version=1,views=8,encode_calls=0,features='native unchanged',
        logits='inverse-aligned XY D4 arithmetic mean',maximum_mean_absolute_logit_delta=0.)

    def encode(images):
        if model.training:
            raise ValueError('Detector TTA is inference-only')
        with torch.no_grad():
            native,logits = original(images)
            totals = [v.float().clone() for v in logits]
            for rotation,reflected in D4_VIEWS[1:]:
                _,view_logits = original(transform(images,rotation,reflected))
                if len(view_logits)!=len(logits):
                    raise ValueError('Detector frame count changed')
                for total,value in zip(totals,view_logits):
                    restored = invert(value,rotation,reflected)
                    if restored.shape!=total.shape:
                        raise ValueError('Detector inverse transform shape mismatch')
                    total.add_(restored.float())
            averaged = [(total/len(D4_VIEWS)).to(value.dtype) for total,value in zip(totals,logits)]
            if not all(torch.isfinite(v).all() for v in averaged):
                raise ValueError('Nonfinite detector average')
            delta = max(float((a.float()-b.float()).abs().mean()) for a,b in zip(averaged,logits))
        receipt['encode_calls']+=1
        receipt['maximum_mean_absolute_logit_delta']=max(receipt['maximum_mean_absolute_logit_delta'],delta)
        return native,averaged

    model.encode=encode
    model._biohub_detector_tta=receipt
    return receipt
