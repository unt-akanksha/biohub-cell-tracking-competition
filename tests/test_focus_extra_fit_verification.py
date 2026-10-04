import json
from pathlib import Path
import runpy
import pytest

ROOT=Path(__file__).resolve().parents[1]
VERIFY=runpy.run_path(str(ROOT/'scripts/verify-focus-extra-fit-cache.py'))


def test_incomplete_outputs_cannot_become_label_input(tmp_path):
    with pytest.raises(FileNotFoundError):VERIFY['verify'](tmp_path)


def test_old_cache_cannot_be_relabelled_as_new_fitting(tmp_path):
    original=json.loads((ROOT/'reports/experiments/focus-adaptation-cache-v1-result.json').read_text())['terminal']
    original['run_id']='focus-extra-fit-cache-v1'
    (tmp_path/'focus_extra_fit_cache_terminal.json').write_text(json.dumps(original))
    with pytest.raises(ValueError,match='Complete bounded'):VERIFY['verify'](tmp_path)
