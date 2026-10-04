import ast
from pathlib import Path
import runpy

ROOT=Path(__file__).resolve().parents[1]
BUILD=runpy.run_path(str(ROOT/'scripts/build-focus-joint-smoke.py'))


def test_worker_four_steps_live_encoder_gradients_and_frozen_external_components():
    source=BUILD['worker']();ast.parse(source)
    assert 'model.unet.requires_grad_(True)' in source
    assert 'model.encode(torch.from_numpy(images.get(sample))' in source
    assert "if k.startswith('detect_head.')" in source
    assert "torch.amp.GradScaler('cuda')" in source and 'scaler.unscale_(optimizer)' in source
    assert 'enumerate(selected,1)' in source and 'SETTINGS[\'steps\']+1' not in source
    assert 'def evaluate(' not in source
    assert source.index('all_four_original_image_replays_passed')<source.index('optimizer=torch.optim.AdamW')
    assert 'Exact real-image checkpoint reload failed' in source


def test_notebook_preserves_offline_environment_inputs_and_budget():
    nb,meta=BUILD['build']({'contract':{}})
    assert meta['is_private'] and meta['enable_gpu'] and not meta['enable_internet'] and not meta['enable_tpu']
    assert len(meta['kernel_sources'])==3 and nb['metadata']['codex']['declared_budget_seconds']==3600
    assert '3480-' in ''.join(nb['cells'][-1]['source'])
    for cell in nb['cells']:ast.parse(''.join(cell['source']))
