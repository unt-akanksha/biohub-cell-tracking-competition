"""Inference-only spatial feature averaging; native detector logits are retained.

Eight unique XY dihedral views, each inverted into the native spatial grid.
No model weights, detection logits, thresholds or coordinate arrays change.
An experimental linker input, not a promoted inference policy.
"""
D4_VIEWS = tuple((rotation, reflected) for rotation in range(4) for reflected in (False, True))


def transform(value, rotation, reflected):
    import torch
    rotated = torch.rot90(value, rotation, dims=(-2,-1))
    return rotated.flip(-1) if reflected else rotated


def invert(value, rotation, reflected):
    import torch
    value = value.flip(-1) if reflected else value
    return torch.rot90(value, -rotation, dims=(-2,-1))


def install_edge_feature_tta(model):
    import torch
    if model.training:
        raise ValueError('Feature TTA requires evaluation mode')
    if hasattr(model, '_biohub_edge_feature_tta'):
        raise ValueError('Feature TTA already installed')
    original = model.encode
    receipt = dict(version=1, views=8, encode_calls=0, detector_logits='native unchanged',
                   maximum_mean_absolute_feature_delta=0.)

    def encode(images):
        if model.training:
            raise ValueError('Feature TTA cannot run in training mode')
        with torch.no_grad():
            native_features, native_logits = original(images)
            total = native_features.float().clone()
            for rotation, reflected in D4_VIEWS[1:]:
                features, _ = original(transform(images, rotation, reflected))
                restored = invert(features, rotation, reflected)
                if restored.shape != native_features.shape:
                    raise ValueError('Feature TTA spatial shape mismatch')
                total.add_(restored.float())
            mean = (total / len(D4_VIEWS)).to(native_features.dtype)
            if not torch.isfinite(mean).all():
                raise ValueError('Nonfinite feature TTA output')
            delta = float((mean.float()-native_features.float()).abs().mean())
        receipt['encode_calls'] += 1
        receipt['maximum_mean_absolute_feature_delta'] = max(receipt['maximum_mean_absolute_feature_delta'],delta)
        # Do not run a detection head on the averaged features.
        return mean, native_logits

    model.encode = encode
    model._biohub_edge_feature_tta = receipt
    return receipt
