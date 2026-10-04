"""Small standalone Torch contract tests, run on the existing Antelume venv."""
import argparse
import json
from pathlib import Path
import sys
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--module-root', type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.module_root))
from visual_correspondence_models_v1 import VisualCorrespondence, prepare_patches, mask_scores

torch.set_num_threads(2)
torch.manual_seed(52)
# CPU synthetic tests do not compete with subsequent GPU training.
stored = torch.randn(2, 4, 2, 15, 15, 15)
coords = torch.tensor([[[0., 0, 0], [1., 2, 3], [2., 4, 2], [3., 2, 1]]]).expand(2, -1, -1).clone()
valid = torch.ones((2, 4), dtype=torch.bool)
x, c = prepare_patches(stored, coords, valid)
assert torch.allclose(x, stored[..., 2:13, 2:13, 2:13], atol=1e-5)
assert torch.equal(c, coords)
assert torch.equal(mask_scores(torch.ones(2, 4), torch.zeros(2, 3, dtype=torch.bool))[:, -1], torch.ones(2))
counts = {}
for family in ('resnet3d', 'token_transformer3d'):
    model = VisualCorrespondence(family).eval()
    counts[family] = sum(p.numel() for p in model.parameters())
    y = model(x, c, valid)
    prior = torch.cat((-2*((c[:, 1:]-c[:, :1])/10).square().sum(-1), torch.full((2, 1), -8.)), -1)
    assert torch.allclose(y, prior, atol=1e-6)
    # Test architecture equivariance with a NONZERO learned residual head.
    torch.nn.init.normal_(model.edge_head[-1].weight, std=.01)
    y = model(x, c, valid)
    order = torch.tensor([0, 3, 1, 2]); output_order = torch.tensor([2, 0, 1, 3])
    yp = model(x[:, order], c[:, order], valid[:, order])
    assert torch.allclose(yp, y[:, output_order], atol=2e-5)
    empty = valid.clone(); empty[:, 1:] = False
    ye = model(x, c, empty)
    assert torch.isfinite(ye).all() and torch.equal(ye.argmax(-1), torch.tensor([3, 3]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad(); loss = torch.nn.functional.cross_entropy(y, torch.tensor([0, 1])); loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert any(p.grad is not None and p.grad.abs().max() > 0 for p in model.encoder.parameters())
    optimizer.step()
print(json.dumps(dict(status='passed', checks=['center_crop', 'null_mask', 'zero_residual_prior',
      'candidate_permutation', 'empty_parent_set', 'finite_backward', 'visual_encoder_gradient'], parameters=counts)))
