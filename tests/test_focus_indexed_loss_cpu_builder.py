from pathlib import Path
import ast
import runpy
import pytest
from research.focus_indexed_parent_loss import require_fitting_sample

ROOT=Path(__file__).resolve().parents[1]


def test_cpu_only_no_inputs_and_embedded_existing_loss():
    wrapper,meta=runpy.run_path(str(ROOT/'scripts/build-focus-indexed-loss-cpu.py'))['build']()
    assert meta['enable_gpu'] is False and meta['enable_tpu'] is False and meta['enable_internet'] is False
    assert all(not meta[k] for k in ('competition_sources','dataset_sources','kernel_sources','model_sources'))
    tree=ast.parse(wrapper)
    assignment=next(n for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='SOURCES')
    bundle=ast.literal_eval(assignment.value)
    assert bundle['sparse_parent_loss.py']==(ROOT/'research/sparse_parent_loss.py').read_text()
    assert bundle['sparse_parent_missing_null.py']==(ROOT/'research/sparse_parent_missing_null.py').read_text()


def test_fitting_only_optimizer_guard_without_importing_torch():
    value={'role':'fitting'};assert require_fitting_sample(value) is value
    for sample in ({'role':'diagnostic'},{'role':'embargo'},{}):
        with pytest.raises(ValueError):require_fitting_sample(sample)
