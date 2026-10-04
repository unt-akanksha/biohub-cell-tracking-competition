import importlib.util
from pathlib import Path
from types import SimpleNamespace
import torch
from research.independent_real_baseline import install_empty_attention_guard

torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('official_attention_test',
    ROOT/'.biohub/vendor/kaggle-cell-tracking-competition/src/tracking_cellmot/models/simple_node_transformer.py')
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def fixture():
    torch.manual_seed(19)
    block = M.CrossAttentionBlock(hidden_dim=8, n_heads=2, dropout=0)
    q = torch.randn(2, 2, 8, requires_grad=True)
    kv = torch.randn(2, 2, 8, requires_grad=True)
    mask = torch.tensor([[True, True], [False, False]])
    return block, q, kv, mask


def test_original_empty_keys_reproduce_nonfinite_gradients():
    block, q, kv, mask = fixture()
    output = block(q, kv, mask)
    output[0].sum().backward()
    assert not torch.isfinite(output[1]).all()
    assert any(p.grad is not None and not torch.isfinite(p.grad).all() for p in block.parameters())


def test_guard_keeps_real_attention_and_backward_finite():
    block, q, kv, mask = fixture()
    expected = block(q[:1], kv[:1], mask[:1]).detach()
    holder = SimpleNamespace(transformer=SimpleNamespace(blocks=[block]))
    keys = list(block.state_dict())
    install_empty_attention_guard(holder)
    install_empty_attention_guard(holder)
    output = block(q, kv, mask)
    torch.testing.assert_close(output[:1], expected)
    output[0].sum().backward()
    assert torch.isfinite(output).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in block.parameters())
    assert torch.isfinite(q.grad).all() and torch.isfinite(kv.grad).all()
    assert list(block.state_dict()) == keys


def test_guard_with_all_empty_keys_is_identity_and_nonempty_path_unchanged():
    block, q, kv, mask = fixture()
    expected = block(q, kv, torch.ones_like(mask)).detach()
    install_empty_attention_guard(SimpleNamespace(transformer=SimpleNamespace(blocks=[block])))
    torch.testing.assert_close(block(q, kv, torch.ones_like(mask)), expected)
    torch.testing.assert_close(block(q, kv, torch.zeros_like(mask)), q)
