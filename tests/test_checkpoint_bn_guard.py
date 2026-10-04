import copy
from pathlib import Path
import sys

import pytest
import torch
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'research'))
from checkpoint_bn_guard import checkpoint_once_batchnorm, recompute_batchnorm_buffers


@pytest.mark.parametrize('momentum', [.1, None])
def test_outputs_gradients_and_single_update_match_uncheckpointed(momentum):
    torch.manual_seed(91)
    reference = torch.nn.Sequential(
        torch.nn.Conv3d(2, 3, 3, padding=1), torch.nn.BatchNorm3d(3, momentum=momentum),
        torch.nn.ReLU(), torch.nn.Conv3d(3, 3, 3, padding=1),
        torch.nn.BatchNorm3d(3, momentum=momentum), torch.nn.ReLU()).train()
    ordinary, guarded = copy.deepcopy(reference), copy.deepcopy(reference)
    for _ in range(3):
        x = torch.randn(2, 2, 4, 5, 5)
        outputs, gradients = [], []
        for model, mode in ((reference, 'plain'), (ordinary, 'ordinary'), (guarded, 'guarded')):
            model.zero_grad(set_to_none=True)
            inp = x.clone().requires_grad_()
            out = model(inp) if mode == 'plain' else (
                checkpoint(model, inp, use_reentrant=False) if mode == 'ordinary'
                else checkpoint_once_batchnorm(model, inp))
            out.square().mean().backward()
            outputs.append(out.detach())
            gradients.append([inp.grad.clone()] + [p.grad.clone() for p in model.parameters()])
        for actual in outputs[1:]:
            torch.testing.assert_close(actual, outputs[0])
        for actual in gradients[1:]:
            for a, b in zip(actual, gradients[0]):
                torch.testing.assert_close(a, b)
        for name, value in reference.state_dict().items():
            torch.testing.assert_close(guarded.state_dict()[name], value)
    for index in (1, 4):
        assert reference[index].num_batches_tracked.item() == 3
        assert ordinary[index].num_batches_tracked.item() == 6
        assert guarded[index].num_batches_tracked.item() == 3


def test_restores_original_buffers_after_exception():
    bn = torch.nn.BatchNorm1d(2).train()
    originals = dict(bn.named_buffers())
    with pytest.raises(RuntimeError, match='deliberate'):
        with recompute_batchnorm_buffers(bn):
            bn(torch.randn(3, 2))
            raise RuntimeError('deliberate')
    assert bn.training
    for name, value in bn.named_buffers():
        assert value is originals[name]
    assert bn.num_batches_tracked.item() == 0


def test_no_running_statistics_supported():
    bn = torch.nn.BatchNorm1d(2, track_running_stats=False)
    x = torch.randn(3, 2, requires_grad=True)
    checkpoint_once_batchnorm(bn, x).square().sum().backward()
    assert torch.isfinite(x.grad).all()


def test_installed_method_uses_replica_state_and_keeps_eval_unchanged():
    from checkpoint_bn_guard import install_checkpoint_batchnorm_guard
    from torch.utils.checkpoint import checkpoint as _grad_ckpt

    class Encoder(torch.nn.Module):
        def _run(self, block, x):
            if self.gradient_checkpointing and self.training:
                return _grad_ckpt(block, x, use_reentrant=False)
            return block(x)

    encoder = Encoder()
    encoder.gradient_checkpointing = True
    install_checkpoint_batchnorm_guard(encoder)
    install_checkpoint_batchnorm_guard(encoder)
    replica = copy.deepcopy(encoder).eval()
    block = torch.nn.Sequential(torch.nn.BatchNorm1d(2), torch.nn.ReLU()).train()
    x = torch.randn(3, 2, requires_grad=True)
    encoder._run(block, x).sum().backward()
    assert block[0].num_batches_tracked.item() == 1
    # Replica's own eval flag must bypass checkpointing (no root-bound closure).
    block.eval()
    torch.testing.assert_close(replica._run(block, x), block(x))
    assert block[0].num_batches_tracked.item() == 1
