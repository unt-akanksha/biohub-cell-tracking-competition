"""Learned residual over the frozen physical prior, with an explicit null parent."""
from independent_motion_prior import SCALE,VARIANCE,NULL_LOGIT


def contract():
    return dict(version=1,scale=SCALE.tolist(),variance_um2=VARIANCE.tolist(),
        null_logit=NULL_LOGIT,inference_activation='sigmoid_of_parent_posterior_log_odds')


def parent_probabilities(scores, *, null_logit=NULL_LOGIT):
    import torch
    null = torch.full_like(scores[...,:1,:],null_logit)
    return torch.softmax(torch.cat([scores,null],dim=-2),dim=-2)[...,:scores.shape[-2],:]


def add_null_target(scores,target):
    import torch
    return (torch.cat([scores,torch.full_like(scores[:1],NULL_LOGIT)],dim=0),
            torch.cat([target,torch.zeros_like(target[:1])],dim=0))


def install_motion_residual(model, *, inference=False):
    import torch
    if hasattr(model,'_motion_residual_mode'):
        raise ValueError('Motion residual already installed')
    original = model.predict_edges
    def predict(*values):
        raw = original(*values)
        source,target = values[2],values[3]
        delta = (source.unsqueeze(-2)-target.unsqueeze(-3))*source.new_tensor(SCALE)
        prior = -.5*(delta.square()/source.new_tensor(VARIANCE)).sum(-1)
        scores = raw+prior
        if not inference:
            return scores
        # The organizer predictor already supports sigmoid edge activation.
        # Encode the null-aware parent posterior as log-odds, with no extra
        # nodes and no modification of the organizer inference implementation.
        return torch.logit(parent_probabilities(scores).clamp(1e-7,1-1e-7))
    model.predict_edges = predict
    model._motion_residual_raw_predict = original
    model._motion_residual_mode = 'inference' if inference else 'training'
