"""Synthetic CPU functionality only; no images, labels, GPU, or model download."""
import hashlib
import io
import json
from pathlib import Path
import time

from blindspot_restoration import build_model, predict_tiled, receptive_offsets


def main():
    import torch
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.manual_seed(1729)
    torch.use_deterministic_algorithms(True)
    if (0,0,0) in receptive_offsets():
        raise ValueError('Symbolic center leakage')
    model = build_model(4).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(.01)
    volume = torch.rand(1,1,17,18,19)
    points = [(0,0,0),(0,8,9),(8,9,9),(16,17,18),(7,7,7)]
    receipts = []
    for point in points:
        x = volume.clone().requires_grad_(True)
        scalar = model(x)[(0,0)+point]
        gradient, = torch.autograd.grad(scalar,x)
        center_gradient = float(gradient[(0,0)+point])
        neighbor_gradient = float(gradient.abs().sum())
        with torch.no_grad():
            changed = volume.clone()
            changed[(0,0)+point] += 17.
            difference = float((model(changed)[(0,0)+point]-scalar.detach()).abs())
        if center_gradient != 0 or difference != 0 or neighbor_gradient <= 0:
            raise ValueError('Blind spot or nontrivial neighboring dependence failed')
        receipts.append(dict(point=point, center_gradient=center_gradient,
                             center_perturbation_output_delta=difference, other_gradient_l1=neighbor_gradient))
    with torch.no_grad():
        whole = model(volume)
        tiled = predict_tiled(model,volume,core=(5,7,8))
        tile_error = float((whole-tiled).abs().max())
    if tile_error > 2e-5:
        raise ValueError('Tiled inference differs from whole-volume inference')
    # Only an optimizer/serialization smoke, not evidence of biological quality.
    trained = build_model(4)
    optimizer = torch.optim.Adam(trained.parameters(),lr=.01)
    noisy_constant = .5 + .05*torch.randn(1,1,12,12,12)
    losses = []
    for step in range(32):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(trained(noisy_constant),noisy_constant)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite training smoke')
        loss.backward()
        if torch.count_nonzero(trained.first.weight.grad[:,:,1,1,1]):
            raise ValueError('Masked central weights received gradient')
        optimizer.step()
        losses.append(float(loss.detach()))
    if losses[-1] >= losses[0]:
        raise ValueError('Training smoke did not lower its synthetic loss')
    trained.eval()
    after = noisy_constant.clone().requires_grad_(True)
    grad, = torch.autograd.grad(trained(after)[0,0,6,6,6],after)
    if float(grad[0,0,6,6,6]) != 0:
        raise ValueError('Training created input-center leakage')
    buffer = io.BytesIO()
    torch.save(trained.state_dict(),buffer)
    checkpoint_sha = hashlib.sha256(buffer.getvalue()).hexdigest()
    buffer.seek(0)
    restored = build_model(4)
    restored.load_state_dict(torch.load(buffer,map_location='cpu',weights_only=True),strict=True)
    with torch.no_grad():
        if not torch.equal(trained(noisy_constant),restored(noisy_constant)):
            raise ValueError('Strict checkpoint replay changed prediction')
    result = dict(status='passed_synthetic_blindspot_functionality',torch_version=torch.__version__,
        device='cpu',gpu_hours=0,competition_data_read=False,ground_truth_read=False,
        seed=1729,synthetic_training_steps=32,initial_synthetic_loss=losses[0],final_synthetic_loss=losses[-1],
        blindspot_receipts=receipts,post_training_center_gradient=0.,tile_max_abs_error=tile_error,
        tile_tolerance=2e-5,checkpoint_roundtrip_exact=True,synthetic_checkpoint_sha256=checkpoint_sha,
        prototype_parameters=sum(p.numel() for p in model.parameters()),
        proposed_width16_parameters=sum(p.numel() for p in build_model(16).parameters()),
        source_sha256={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                       for name in ('blindspot_restoration.py','probe-blindspot-restoration.py')},
        elapsed_seconds=time.monotonic()-started,authorized_for_submission=False,
        caveat='Synthetic functionality only; no microscope denoising or detection gain established.')
    Path('/kaggle/working/blindspot_cpu_probe.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
