"""Fixed equal-probability detector ensemble; parent features stay native."""


def probability_mixture_logit(a, b):
    import torch
    import torch.nn.functional as F
    if a.shape != b.shape or a.device != b.device:
        raise ValueError('Ensemble shape/device mismatch')
    if not a.is_floating_point() or not b.is_floating_point():
        raise ValueError('Floating logits required')
    a, b = a.float(), b.float()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError('Nonfinite component logits')
    return (torch.logaddexp(-F.softplus(-a), -F.softplus(-b))
            - torch.logaddexp(-F.softplus(a), -F.softplus(b)))


def install_owned_detector_ensemble(parent, secondary):
    import torch
    from detector_spatial_tta import install_detector_spatial_tta
    if parent is secondary or any(m.training or hasattr(m, '_image_motion_mode')
                                 or hasattr(m, '_biohub_detector_tta')
                                 or hasattr(m, '_biohub_detector_ensemble')
                                 for m in (parent, secondary)):
        raise ValueError('Distinct frozen native eval models required before flow wrapper')
    for model in (parent, secondary):
        model.requires_grad_(False)
    first = install_detector_spatial_tta(parent)
    second = install_detector_spatial_tta(secondary)
    original = parent.encode
    receipt = dict(version=1, weights=[.5, .5], probability_mixture=True,
                   features='parent native unchanged', parent_d4=first,
                   secondary_d4=second, encode_calls=0,
                   maximum_mean_absolute_logit_delta=0.)

    def encode(images):
        if parent.training or secondary.training:
            raise ValueError('Ensemble is inference-only')
        with torch.no_grad():
            features, a = original(images)
            _, b = secondary.encode(images)
            if not a or len(a) != len(b):
                raise ValueError('Ensemble frame count mismatch')
            logits = [probability_mixture_logit(x, y) for x, y in zip(a, b)]
            delta = max(float((x-y.float()).abs().mean()) for x, y in zip(logits, a))
        receipt['encode_calls'] += 1
        receipt['maximum_mean_absolute_logit_delta'] = max(
            receipt['maximum_mean_absolute_logit_delta'], delta)
        return features, logits

    parent.encode = encode
    parent._biohub_detector_ensemble = receipt
    return receipt
