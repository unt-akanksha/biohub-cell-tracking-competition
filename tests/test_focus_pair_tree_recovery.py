import json
from pathlib import Path
import runpy
import subprocess
import sys

import pytest

from research.focus_pair_tree_resume import resume

ROOT = Path(__file__).resolve().parents[1]
DRIVER = runpy.run_path(str(ROOT/'scripts/recover-focus-pair-tree-profile.py'))


def test_resume_role_guard_before_library_or_input_access(tmp_path):
    with pytest.raises(ValueError, match='Only fitting'):
        resume(None, None, None, None, None, None, None, None, tmp_path/'bad', 'diagnostic')
    assert not (tmp_path/'bad').exists()


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows launcher regression')
def test_direct_command_is_actual_pid_with_private_package_path():
    command = DRIVER['direct_command']('import json,os,numpy; print(json.dumps(dict(pid=os.getpid(), numpy=numpy.__file__)), flush=True)')
    assert command[:3] == [sys._base_executable, '-S', '-c']
    child = subprocess.Popen(command, stdout=subprocess.PIPE, text=True, cwd=ROOT)
    try:
        text, _ = child.communicate(timeout=20)
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=10)
    assert child.returncode == 0
    identity = json.loads(text)
    assert identity['pid'] == child.pid
    assert Path(identity['numpy']).is_relative_to(Path(sys.prefix)/'Lib/site-packages')


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows process memory regression')
def test_real_child_allocation_detected_and_owned_process_stopped():
    result = DRIVER['guard_smoke']()
    assert result['peak_working_bytes'] > 32*1024**2
    assert result['detected_over_32_mib'] and result['owned_child_terminated']
    assert result['printed_pid'] in (None, result['monitored_pid'])
