"""Separate architectural center exclusion from global input sensitivity.

A dead local ReLU has zero derivatives without implying a constant model.
Require center exclusion at every probe and an actual output response to
changing the full input, not a nonzero derivative at each arbitrary location.
"""
import math


def assess(records,input_response_max):
    if len(records)<5 or not math.isfinite(input_response_max) or input_response_max<0:
        raise ValueError('Multiple finite functional probes required')
    if any(not math.isfinite(r[k]) or r[k]<0 for r in records
           for k in ('center_gradient_abs','center_output_delta','gradient_l1')):
        raise ValueError('Finite nonnegative sensitivity observations required')
    conditions=dict(center_gradients_zero=all(r['center_gradient_abs']==0 for r in records),
                    center_perturbations_zero=all(r['center_output_delta']==0 for r in records),
                    actual_input_response=input_response_max>0)
    return dict(conditions=conditions,passed=all(conditions.values()),records=records,
                input_response_max=input_response_max,
                observed_gradient_l1_max=max(r['gradient_l1'] for r in records),
                inactive_local_probes=sum(r['gradient_l1']==0 for r in records))


def audit(model,samples):
    import torch
    if samples.ndim!=5 or samples.shape[1]!=1 or samples.shape[0]!=3:
        raise ValueError('Three predetermined training-image samples required')
    shape=samples.shape[-3:]
    points=[(0,0,0),(0,shape[1]//2,shape[2]//2),tuple(n//2 for n in shape),
            tuple(n-1 for n in shape),(7,7,7)]
    with torch.no_grad():
        response=float((model(samples)-model(torch.zeros_like(samples))).abs().max())
    records=[]
    with torch.enable_grad():
        for i in range(3):
            for point in points:
                sample=samples[i:i+1].detach().clone().requires_grad_(True)
                value=model(sample)[(0,0)+point]
                gradient,=torch.autograd.grad(value,sample)
                with torch.no_grad():
                    changed=sample.detach().clone()
                    changed[(0,0)+point]+=17.
                    delta=float((model(changed)[(0,0)+point]-value.detach()).abs())
                records.append(dict(sample=i,point=point,center_gradient_abs=float(gradient[(0,0)+point].abs()),
                                    center_output_delta=delta,gradient_l1=float(gradient.abs().sum())))
    return assess(records,response)
