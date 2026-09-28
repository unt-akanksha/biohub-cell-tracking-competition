"""Bounded same-example AMP backoff, never updating on nonfinite gradients."""
from focus_joint_probe import SETTINGS as ORIGINAL

SETTINGS=dict(ORIGINAL,amp_initial_scale=65536.,max_amp_attempts=17)


def joint_update(model,optimizer,scaler,sample,forward,loss_fn,null_weight,tensor_hash):
    import torch
    cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all()
    before=tensor_hash(model.state_dict());trainable=list(model.unet.parameters())+list(model.transformer.parameters())
    skipped=[]
    for attempt in range(1,SETTINGS['max_amp_attempts']+1):
        if attempt>1:torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state_all(cuda_rng)
        model.eval();model.transformer.train();optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda',dtype=torch.float16):scores,labels=forward(sample)
        loss=loss_fn(scores.float(),labels,null_weight)
        if not torch.isfinite(loss):raise ValueError('Nonfinite forward loss is not a loss-scaling recovery case')
        scale_before=float(scaler.get_scale());scaler.scale(loss).backward();scaler.unscale_(optimizer)
        if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in trainable):
            # GradScaler recorded found_inf during unscale; step must skip the optimizer.
            scaler.step(optimizer);scaler.update();scale_after=float(scaler.get_scale())
            if tensor_hash(model.state_dict())!=before or not scale_after<scale_before:raise ValueError('Overflow backoff changed parameters or did not reduce scale')
            skipped.append(dict(attempt=attempt,scale_before=scale_before,scale_after=scale_after,optimizer_skipped=True,parameters_unchanged=True))
            continue
        gradients={name:any(p.grad is not None and bool((p.grad!=0).any()) for p in module.parameters()) for name,module in [('encoder',model.unet),('head',model.transformer)]}
        if not all(gradients.values()):raise ValueError('Both trainable groups need nonzero gradients')
        norm=torch.nn.utils.clip_grad_norm_(trainable,SETTINGS['gradient_clip'],error_if_nonfinite=True)
        scaler.step(optimizer);scaler.update()
        return dict(loss=float(loss.detach()),gradient_norm=float(norm),nonzero_gradients=gradients,attempts=attempt,
            successful_scale=scale_before,overflow_retries=skipped)
    raise ValueError('Bounded AMP retry budget exhausted; no larger training authorized')
