import ast
from pathlib import Path
import runpy

ROOT=Path(__file__).resolve().parents[1]


def test_bounded_probe_embeds_training_collector_without_mutating_frozen_jobs():
    paths=[ROOT/f'kaggle/biohub-{name}/biohub-{name}.ipynb' for name in ('owned-detector-pu-transfer-v1','owned-detector-pu-selection-v1')]
    original=[p.read_bytes() for p in paths]
    nb,meta=runpy.run_path(str(ROOT/'scripts/build-detector-calibration-probe.py'))['build']()
    assert [p.read_bytes() for p in paths]==original
    assert meta['enable_gpu'] and not meta['enable_tpu'] and not meta['enable_internet']
    assert meta['kernel_sources']==['indarkarhana/biohub-image-motion-linker-v1/2','indarkarhana/biohub-owned-detector-fit-pair-v1/1']
    codex=nb['metadata']['codex']
    assert codex['probe_movies']==6 and codex['frames_per_movie']==3 and codex['declared_budget_seconds']==3600
    assert not codex['target_audit_opened'] and not codex['authorized_for_submission']
    source=''.join(nb['cells'][1]['source'])
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='runtime_sources')
    runtime=ast.literal_eval(node.value)
    assert runtime['run_selection.py']==(ROOT/'scripts/collect-detector-calibration-probe.py').read_text()
    assert 'calibration_scope(split,probe=True)' in runtime['run_selection.py']
    assert 'autocast' not in runtime['run_selection.py']
    launch=''.join(nb['cells'][-1]['source'])
    assert "'--candidate'" in launch and "'--parent'" in launch
    assert "'--checkpoint'" not in launch and '--embryo-audit' not in launch
