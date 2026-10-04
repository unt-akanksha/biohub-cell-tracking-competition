import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('event_download', ROOT / 'scripts/download-trajectory-event-source-v1.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_all_source_batch_downloaders_are_bounded_and_compile():
    base = ROOT / 'scripts/download-trajectory-division-archive-v1.py'
    for index in range(8):
        scope = ROOT / '.biohub/cache/trajectory-event-source-v1-plan' / ('batch-' + str(index)) / 'MOVIES.json'
        source = module.configured_source(base, scope)
        compile(source, str(base), 'exec')
        assert '5_500_000_000' in source
        assert 'Image metadata does not match complete inference contract' in source
        assert 'Image checksum' not in source  # Transfer uses ZIP CRC plus returned SHA256.
        assert "'archive_member_smoke_passed'" in source


def test_selection_scope_cannot_be_used_for_source_collection(tmp_path):
    plan = json.loads((ROOT / '.biohub/cache/trajectory-event-source-v1-plan/batch-0/MOVIES.json').read_text())
    plan['movies'][0]['role'] = 'selection'
    scope = tmp_path / 'MOVIES.json'
    scope.write_text(json.dumps(plan))
    with pytest.raises(AssertionError):
        module.configured_source(ROOT / 'scripts/download-trajectory-division-archive-v1.py', scope)
