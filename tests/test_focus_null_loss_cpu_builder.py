import ast
from pathlib import Path
import runpy

ROOT=Path(__file__).resolve().parents[1]


def test_no_gpu_or_data_and_bounded_cpu_worker():
    source,meta=runpy.run_path(str(ROOT/'scripts/build-focus-null-loss-cpu.py'))['build']()
    assert not meta['enable_gpu'] and not meta['enable_tpu'] and not meta['enable_internet']
    assert all(not meta[k] for k in ('dataset_sources','kernel_sources','competition_sources','model_sources'))
    tree=ast.parse(source)
    sources=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='SOURCES'))
    assert len(sources)==5 and 'timeout=600' in source and 'declared_budget_seconds=900' in source
    for value in sources.values():ast.parse(value)
    assert 'torch.cuda.is_initialized()' in sources['probe.py']
