import ast
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
M = runpy.run_path(str(ROOT/'scripts/build-blindspot-cpu-probe.py'))


def test_probe_is_private_offline_cpu_with_no_inputs():
    source,meta = M['build']()
    assert meta['is_private']
    for key in ('enable_gpu','enable_tpu','enable_internet','dataset_sources','competition_sources','kernel_sources','model_sources'):
        assert not meta[key]
    assert 'timeout=600' in source and 'declared_budget_seconds=900' in source
    assert len(meta['title']) <= 50


def test_exact_actual_model_and_check_source_embedded():
    source,_ = M['build']()
    node = next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
                and isinstance(n.targets[0],ast.Name) and n.targets[0].id == 'SOURCES')
    sources = ast.literal_eval(node.value)
    assert sources['blindspot_restoration.py'] == (ROOT/'research/blindspot_restoration.py').read_text()
    assert sources['probe-blindspot-restoration.py'] == (ROOT/'scripts/probe-blindspot-restoration.py').read_text()
