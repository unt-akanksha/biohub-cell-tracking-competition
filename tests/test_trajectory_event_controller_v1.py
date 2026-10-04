import importlib.util
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('event_controller',ROOT/'scripts/collect-trajectory-event-source-v1.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_controller_recognizes_terminal_failure_without_restarting():
    assert not module.terminal_status('running')
    assert module.terminal_status('complete_prelabel_predictions')
    assert module.terminal_status('failed')
    assert module.terminal_status('timeout')
    with pytest.raises(ValueError):
        module.terminal_status('unknown')


def test_controller_bounds_and_recovery_are_present():
    source=(ROOT/'scripts/collect-trajectory-event-source-v1.py').read_text()
    assert 'args.through_batch <= 7' in source
    assert 'time.sleep(30)' in source
    assert source.index("script('harvest-trajectory-event-source-v1.py'") < source.index("script('retire-trajectory-event-images-v1.py'")
    assert 'no restart or next batch' in source
    assert 'Never duplicate a recorded launch' in source
