import importlib.util
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('fork16_training_control', ROOT / 'scripts/train-trajectory-event-fork16-source-v1.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_single_writer_lock_rejects_duplicate_and_releases(tmp_path):
    path = tmp_path / 'training.lock'
    with module.exclusive_lock(path):
        with pytest.raises(OSError):
            with module.exclusive_lock(path):
                pytest.fail('Duplicate writer acquired the live lock')
    with module.exclusive_lock(path):
        pass


def test_atomic_checkpoint_json_replacement(tmp_path):
    path = tmp_path / 'latest.json'
    module.write_json(path, dict(steps=2))
    module.write_json(path, dict(steps=4))
    assert module.read(path) == dict(steps=4)
    assert not path.with_name('latest.json.tmp').exists()


def test_schedule_detects_missing_or_duplicate_cases():
    valid = dict(epoch=1, position=2, history=[{}], loss_sum=1., order=[2, 0, 1])
    module.validate_schedule(dict(steps=5), valid, 3, 3)
    with pytest.raises(AssertionError):
        module.validate_schedule(dict(steps=5), dict(valid, order=[2, 2, 1]), 3, 3)
    with pytest.raises(AssertionError):
        module.validate_schedule(dict(steps=4), valid, 3, 3)
    module.validate_schedule(dict(steps=9), dict(epoch=3, position=0, history=[{}, {}, {}], loss_sum=0., order=[]), 3, 3)
