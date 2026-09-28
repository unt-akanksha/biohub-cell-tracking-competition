"""Re-estimate BatchNorm buffers without changing any learned parameter."""
import hashlib


def recalibrate_batchnorm(model,batches,encode):
    import torch
    modules = dict(model.named_modules())
    bn = {name:module for name,module in modules.items()
          if isinstance(module,torch.nn.modules.batchnorm._BatchNorm) and module.track_running_stats}
    if not bn:
        raise ValueError('No tracked BatchNorm layers')
    allowed = {f'{name}.{key}'.lstrip('.') for name in bn
               for key in ('running_mean','running_var','num_batches_tracked')}
    before = {key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
    modes = {module:module.training for module in model.modules()}
    momenta = {module:module.momentum for module in bn.values()}
    def parameter_hash():
        digest = hashlib.sha256()
        for name,value in model.named_parameters():
            digest.update(name.encode()+b'\0'+value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()
    initial_hash = parameter_hash()
    count = 0
    try:
        model.eval()
        for module in bn.values():
            module.reset_running_stats()
            module.train()
            module.momentum = None
        with torch.no_grad():
            for batch in batches:
                encode(batch)
                count += 1
        if not count:
            raise ValueError('Empty calibration stream')
    finally:
        for module,momentum in momenta.items():
            module.momentum = momentum
        for module,training in modes.items():
            module.training = training
    changed = []
    for key,value in model.state_dict().items():
        if not torch.equal(before[key],value.detach().cpu()):
            if key not in allowed:
                raise ValueError('Non-BatchNorm state changed: '+key)
            changed.append(key)
        if key in allowed and not torch.isfinite(value).all():
            raise ValueError('Nonfinite BatchNorm buffer')
    final_hash = parameter_hash()
    if final_hash != initial_hash or not changed:
        raise ValueError('Weights changed or calibration did not change buffers')
    for module in bn.values():
        if int(module.num_batches_tracked) != count or (module.running_var < 0).any():
            raise ValueError('Incomplete or invalid BatchNorm statistics')
    return dict(batches=count,bn_layers=len(bn),changed_buffers=changed,
                initial_batches_tracked={name:int(before[f'{name}.num_batches_tracked'.lstrip('.')]) for name in bn},
                parameter_sha256=final_hash,parameters_unchanged=True)
