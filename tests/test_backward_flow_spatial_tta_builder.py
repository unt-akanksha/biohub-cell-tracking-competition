import ast
import hashlib
from pathlib import Path
import runpy

ROOT=Path(__file__).resolve().parents[1]


def test_motion_only_probe_staging_does_not_mutate_live_ensemble():
    frozen=ROOT/'kaggle/biohub-owned-detector-ensemble-selection-v1/biohub-owned-detector-ensemble-selection-v1.ipynb'
    before=hashlib.sha256(frozen.read_bytes()).hexdigest()
    nb,meta=runpy.run_path(str(ROOT/'scripts/build-backward-flow-spatial-tta-probe.py'))['build']()
    assert hashlib.sha256(frozen.read_bytes()).hexdigest()==before
    assert nb['metadata']['codex']['declared_budget_seconds']==3600
    assert nb['metadata']['codex']['flow_views']==8
    assert not nb['metadata']['codex']['target_audit_opened']
    assert meta['enable_gpu'] and not meta['enable_internet'] and not meta['enable_tpu']
    assert meta['kernel_sources']==['indarkarhana/biohub-image-motion-linker-v1/2']
    source=''.join(nb['cells'][1]['source'])
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='runtime_sources')
    runtime=ast.literal_eval(node.value)
    assert 'flow_patch_receipt=flow_receipt' in runtime['real_checkpoint_gpu_smoke.py']
    assert 'detections_identical=True' in runtime['run_pilot.py']
    assert 'test_backward_flow_spatial_tta.py' in ' '.join(runtime)
    for name,code in runtime.items():
        if name.endswith('.py'): ast.parse(code)


def test_three_arm_probe_embeds_zero_weight_regression_and_keeps_v1_frozen():
    path=ROOT/'kaggle/biohub-backward-flow-spatial-tta-probe-v1/biohub-backward-flow-spatial-tta-probe-v1.ipynb'
    before=path.read_bytes()
    nb,meta=runpy.run_path(str(ROOT/'scripts/build-backward-flow-spatial-tta-probe-v2.py'))['build']()
    assert path.read_bytes()==before
    assert meta['id'].endswith('-v2')
    assert nb['metadata']['codex']['zero_weight_graph_parity_required']
    source=''.join(nb['cells'][1]['source'])
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='runtime_sources')
    runtime=ast.literal_eval(node.value)
    assert runtime['calibrated_motion_scores.py']==(ROOT/'research/calibrated_motion_scores.py').read_text()
    assert 'test_optional_zero_weight_shortcut_preserves_scores_and_skips_neural' in runtime['tests/test_image_motion_residual.py']
    assert 'zero_weight_graph_identical=True' in runtime['run_pilot.py']
    assert "('optimized',control_flow)" in runtime['run_pilot.py']
    assert 'skip_zero_neural=skip_zero_neural' in runtime['real_checkpoint_gpu_smoke.py']
