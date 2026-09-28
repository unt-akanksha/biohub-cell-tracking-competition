"""Discard duplicate BatchNorm running-stat updates during recomputation only.

Eager, non-reentrant checkpointing only. Training outputs still use batch
statistics. This context does not change parameters, modes, or state-dict keys.
Concurrent forwards through the same module instance are not supported.
"""
from contextlib import contextmanager, nullcontext


@contextmanager
def recompute_batchnorm_buffers(block):
    from torch.nn.modules.batchnorm import _BatchNorm

    saved = []
    try:
        for module in block.modules():
            if isinstance(module, _BatchNorm) and module.track_running_stats:
                for name in ('running_mean', 'running_var', 'num_batches_tracked'):
                    value = getattr(module, name)
                    if value is not None:
                        saved.append((module, name, value))
                        setattr(module, name, value.clone())
        yield
    finally:
        for module, name, value in reversed(saved):
            setattr(module, name, value)


def checkpoint_once_batchnorm(block, *args):
    from torch.utils.checkpoint import checkpoint

    return checkpoint(block, *args, use_reentrant=False,
                      context_fn=lambda: (nullcontext(), recompute_batchnorm_buffers(block)))


def install_checkpoint_batchnorm_guard(unet):
    """Patch the pinned class, not a bound instance method copied by DataParallel."""
    import ast
    import inspect
    import textwrap

    cls = type(unet)
    if getattr(cls._run, '_biohub_bn_once', False):
        return
    original = ast.parse(textwrap.dedent(inspect.getsource(cls._run))).body[0]
    expected = ast.parse('''def _run(self, block, x):
    if self.gradient_checkpointing and self.training:
        return _grad_ckpt(block, x, use_reentrant=False)
    return block(x)
''').body[0]
    if [ast.dump(n) for n in original.body] != [ast.dump(n) for n in expected.body]:
        raise ValueError('Checkpoint _run source drift')

    def guarded_run(self, block, x):
        if self.gradient_checkpointing and self.training:
            return checkpoint_once_batchnorm(block, x)
        return block(x)

    guarded_run._biohub_bn_once = True
    cls._run = guarded_run
