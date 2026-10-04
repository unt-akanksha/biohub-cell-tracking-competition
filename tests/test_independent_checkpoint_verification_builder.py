from pathlib import Path
import runpy


def test_verification_runs_only_cpu_with_exact_training_artifact():
    root = Path(__file__).resolve().parents[1]
    builder = runpy.run_path(str(root/'scripts/build-independent-checkpoint-verification.py'))
    nb,meta = builder['build']()
    assert not meta['enable_gpu'] and not meta['enable_tpu'] and not meta['enable_internet']
    assert meta['kernel_sources'] == ['indarkarhana/biohub-independent-real-pilot-v1/4']
    source = ''.join(nb['cells'][0]['source'])
    assert "['verify'](root,work/'pilot_launch.ipynb')" in source
    assert 'threading.Timer(600,timeout)' in source
