"""Learned-linker integration of frozen image flow and optional calibration.

Install only on a model with the existing TRAINING-mode static residual.
The replacement changes the spatial prior, not the detector or neural scores.
"""
from backward_flow_ops import sample_backward_flow
from independent_motion_prior import SCALE,VARIANCE
from motion_residual import parent_probabilities

FLOW_CHECKPOINT_SHA = '3006ee0f904640b16dd988d6404a1b3ac4f933ca68ca912d69fe4598f96d4788'


def contract():
    return dict(version=1,flow_checkpoint_sha256=FLOW_CHECKPOINT_SHA,flow_frozen=True,
                displacement='child_to_parent_um_zyx',variance='unchanged_static_control',
                downsample=[1,4,4],neural_residual='preserved_then_trained')


def validate_flow_initialization(state,split):
    identity = state['identity']
    fold = split['folds'][0]
    diagnostic = fold['train'][::5]
    if (state['step'] != 1000 or identity['max_steps'] != 1000
        or identity['fitting_stems'] != [s for s in fold['train'] if s not in diagnostic]
        or identity['diagnostic_stems'] != diagnostic or identity['downsample'] != [1,4,4]
        or any(identity[k] is not False for k in ('public_checkpoint_loaded','selection_opened','target_audit_opened','authorized_for_submission'))):
        raise ValueError('Completed independently fitted flow model required')


def flow_hash(model):
    import hashlib
    digest = hashlib.sha256()
    for name,value in model.state_dict().items():
        digest.update(name.encode()+b'\0')
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def embedded_flow(state,device):
    from backward_flow_model import BackwardFlowNet
    if state['identity'].get('image_motion') != contract():
        raise ValueError('Embedded image-motion contract mismatch')
    flow = BackwardFlowNet()
    flow.load_state_dict(state['frozen_flow_model'],strict=True)
    if flow_hash(flow) != state['identity']['frozen_flow_sha256']:
        raise ValueError('Embedded frozen flow checksum mismatch')
    return flow.to(device).requires_grad_(False).eval()


def calibration_parameters(values):
    import math
    bounds = ((0.,1.),(.25,4.),(-12.,4.))
    if len(values)!=3 or any(not math.isfinite(v) or not lo<=v<=hi for v,(lo,hi) in zip(values,bounds)):
        raise ValueError('Three finite bounded calibration parameters required')
    return tuple(float(v) for v in values)


def install_image_motion_residual(model,flow_model,*,inference=False,calibration=None,skip_zero_neural=False):
    import torch
    if getattr(model,'_motion_residual_mode',None) != 'training':
        raise ValueError('Static training-logit residual must be installed first')
    if hasattr(model,'_image_motion_mode'):
        raise ValueError('Image motion already installed')
    calibrated = calibration_parameters(calibration) if calibration is not None else None
    if skip_zero_neural and (calibrated is None or calibrated[0]!=0. or not inference):
        raise ValueError('Zero-neural shortcut requires explicit zero-weight calibrated inference')
    if type(skip_zero_neural) is not bool:
        raise ValueError('Explicit boolean execution option required')
    execution = dict(skip_zero_neural=skip_zero_neural,neural_forward_calls=0,zero_weight_skips=0)
    if calibrated is not None and (not inference or not hasattr(model,'_motion_residual_raw_predict')):
        raise ValueError('Calibration requires inference with direct unmodified neural scores')
    flow_model.requires_grad_(False).eval()
    original_encode,original_predict = model.encode,model.predict_edges
    cache = {}

    def encode(images):
        cache.clear()
        if images.ndim != 5 or images.shape[1] != 2:
            raise ValueError('Exactly one previous/current frame pair per sample required')
        features,detector = original_encode(images)
        flow_model.eval()
        with torch.no_grad(),torch.amp.autocast(images.device.type,dtype=torch.float16,
                                               enabled=images.device.type == 'cuda'):
            # Official training inputs and standalone flow inference are
            # quantized to FP16 before encoding. Leave detector inputs intact.
            cache['flow'] = flow_model(images.half().float()).detach()
        return features,detector

    def predict(*values):
        field = cache.pop('flow',None)
        if field is None:
            raise ValueError('A fresh encoded image pair is required for edge prediction')
        source,target = values[2],values[3]
        if field.shape[0] != source.shape[0] or target.shape[0] != source.shape[0]:
            raise ValueError('Image and node batch mismatch')
        flow,valid = sample_backward_flow(field,target/target.new_tensor([1,4,4]))
        target_mask = values[7]
        if target_mask.dtype != torch.bool or target_mask.shape != valid.shape or (target_mask&~valid).any():
            raise ValueError('Real target outside flow grid; no clamping or deletion')
        flow = flow*target_mask[...,None]
        delta = (source.unsqueeze(-2)-target.unsqueeze(-3))*source.new_tensor(SCALE)
        static = -.5*(delta.square()/source.new_tensor(VARIANCE)).sum(-1)
        shifted = delta-flow.unsqueeze(-3)
        image_prior = -.5*(shifted.square()/source.new_tensor(VARIANCE)).sum(-1)
        if calibrated is not None:
            alpha,beta,null_logit = calibrated
            # Direct neural scores match the frozen calibration collector;
            # do not subtract large spatial logits to recover small features.
            if skip_zero_neural:
                from calibrated_motion_scores import calibrated_motion_scores
                scores = calibrated_motion_scores(image_prior,
                    lambda:model._motion_residual_raw_predict(*values),alpha,beta,skip_zero_neural=True)
                execution['zero_weight_skips']+=1
            else:
                scores = alpha*model._motion_residual_raw_predict(*values)+beta*image_prior
                execution['neural_forward_calls']+=1
            return torch.logit(parent_probabilities(scores,null_logit=null_logit).clamp(1e-7,1-1e-7))
        scores = original_predict(*values)+(image_prior-static)
        if inference:
            return torch.logit(parent_probabilities(scores).clamp(1e-7,1-1e-7))
        return scores

    model.encode,model.predict_edges = encode,predict
    model._image_motion_mode = 'inference' if inference else 'training'
    model._image_motion_execution = execution
    if calibrated is not None:
        model._image_motion_calibration = list(calibrated)
