"""Reuse the public eighth encode, which receives exactly the earlier flip-X view.

Preserves the original eight contributions, inverse transforms and addition
order. This is NOT the previously rejected D4 correction.
"""
from research.trajectory_portable_adapter_v1 import replace_once


def cached_predictor(source):
    source=replace_once(source,
        '                _u_flip, det_flip = model.encode(imgs_flip)\n',
        '                _u_flip, det_flip = model.encode(imgs_flip)\n'
        '                if dims == (-1,):\n'
        '                    _cache_x_image, _cache_x_features, _cache_x_logits = imgs_flip, _u_flip, det_flip\n')
    source=replace_once(source,
        '            _u_at, det_at = model.encode(imgs_at)\n',
        '            if not torch.equal(imgs_at, _cache_x_image):\n'
        '                raise RuntimeError("Exact duplicate-view input identity failed")\n'
        '            _u_at, det_at = _cache_x_features, _cache_x_logits\n')
    source=replace_once(source,
        '            del imgs_at, det_at, _u_at\n',
        '            del imgs_at, det_at, _u_at\n'
        '            del _cache_x_image, _cache_x_features, _cache_x_logits\n')
    source=replace_once(source,
        '                        _secondary_u_flip, secondary_det_flip = secondary_model.encode(\n'
        '                            secondary_imgs_flip\n'
        '                        )\n',
        '                        _secondary_u_flip, secondary_det_flip = secondary_model.encode(\n'
        '                            secondary_imgs_flip\n'
        '                        )\n'
        '                        if dims == (-1,):\n'
        '                            _secondary_cache_x = (secondary_imgs_flip, _secondary_u_flip, secondary_det_flip)\n')
    source=replace_once(source,
        '                    _secondary_u_at, secondary_det_at = secondary_model.encode(\n'
        '                        secondary_imgs_at\n'
        '                    )\n',
        '                    if not torch.equal(secondary_imgs_at, _secondary_cache_x[0]):\n'
        '                        raise RuntimeError("Secondary duplicate-view input identity failed")\n'
        '                    _secondary_u_at, secondary_det_at = _secondary_cache_x[1:]\n')
    source=replace_once(source,
        '                    del secondary_imgs_at, secondary_det_at, _secondary_u_at\n',
        '                    del secondary_imgs_at, secondary_det_at, _secondary_u_at\n'
        '                    del _secondary_cache_x\n')
    return source
