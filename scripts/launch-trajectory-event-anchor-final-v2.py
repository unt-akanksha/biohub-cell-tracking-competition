"""One-shot same-inference final save, with a full 12-hour platform reservation."""
import argparse
import csv
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.trajectory_runtime_v1 import sha
from research.trajectory_event_quota_v1 import saved_kernel_complete
from research.trajectory_event_final_budget_v1 import final_budget
from research.submission_sharding import validate_submission_kernel_metadata


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    reports = ROOT / 'reports/experiments'
    receipt = reports / 'trajectory-event-anchor-final-v2-launch.json'
    assert not receipt.exists(), 'Previous launch attempted; inspect remote version first'
    build_path = reports / 'trajectory-event-anchor-final-v2-build.json'
    build = read(build_path)
    assert build['quality_decision'] == 'one_exploratory_entry_after_final_runtime_and_output_verification'
    assert build['strict_quality_failures_preserved'] and not build['strict_quality_gates_pass']
    assert build['executable_cells_identical_to_verified_public_test'] and not build['failed_gate_included']
    assert build['requested_platform_timeout_seconds'] == 43200
    proof_path = reports / 'trajectory-event-anchor-production-v1-verification-r2.json'
    assert sha(proof_path) == build['public_verification_sha256']
    proof = read(proof_path)
    assert proof['status'] == 'public_production_verified_not_submittable_version'
    stage = ROOT / '.biohub/staging/biohub-event-anchor-candidate-v2'
    metadata = read(stage / 'kernel-metadata.json')
    validate_submission_kernel_metadata(metadata)
    kernel = metadata['id']
    assert kernel == build['kernel'] == 'indarkarhana/biohub-event-anchor-candidate'
    for name, key in ((metadata['code_file'], 'notebook_sha256'), ('derived-contract.json', 'contract_sha256'), ('overlays.json', 'overlay_sha256')):
        assert sha(stage / name) == build[key]
    env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
    cli = str(Path(sys.executable).parent / 'Scripts/kaggle.exe')
    assert Path(cli).exists(), 'Use authenticated base Python'
    def run(command, timeout=60):
        return subprocess.run(command, capture_output=True, text=True, encoding='utf-8', env=env, check=True, timeout=timeout).stdout
    state = json.loads(run([sys.executable, str(ROOT / 'scripts/get-kaggle-kernel-state.py'), '--kernel-slug', kernel]))
    assert state['current_version_number'] == 1 and state['is_private'] and not state['enable_internet']
    assert saved_kernel_complete(run([cli, 'kernels', 'status', kernel]), kernel)
    remote_check = """import json,subprocess
from pathlib import Path
raw=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=15).stdout
foreign=[]
for value in raw.splitlines():
    if not value.strip():continue
    pid=int(value.strip());proc=Path('/proc')/str(pid)
    try:cmd=(proc/'cmdline').read_bytes().decode(errors='replace');cwd=str((proc/'cwd').resolve())
    except FileNotFoundError:continue
    assert 'biohub' not in cmd.lower() and 'biohub' not in cwd.lower(),'Biohub GPU run still live'
    foreign.append(pid)
print(json.dumps(dict(live_biohub_gpu_processes=[],foreign_gpu_processes_untouched=foreign)))
"""
    remote = subprocess.run(['ssh', '-i', 'C:/Users/IndarKumar/.ssh/rsna_ec2', '-o', 'BatchMode=yes',
        '-o', 'ConnectTimeout=15', '-o', 'StrictHostKeyChecking=yes', '-o', 'HostKeyAlias=13.220.240.128',
        'ubuntu@3.226.249.134', '/home/ubuntu/venv/bin/python -'], input=remote_check,
        capture_output=True, text=True, check=True, timeout=45)
    sequential = json.loads(remote.stdout)
    history = list(csv.DictReader(io.StringIO(run([cli, 'competitions', 'submissions', '-c', 'biohub-cell-tracking-during-development', '--csv']))))
    quota = json.loads(run([cli, 'quota', '--format', 'json']))
    gpu = [q for q in quota if q['resource'] == 'GPU']
    assert len(gpu) == 1 and gpu[0]['remaining'].endswith('h')
    budget = final_budget(float(gpu[0]['remaining'][:-1]), history, {})
    result = dict(status='preflight_passed', utc=datetime.now(timezone.utc).isoformat(), kernel=kernel,
        expected_kernel_version=2, build_sha256=sha(build_path), notebook_sha256=build['notebook_sha256'],
        contract_sha256=build['contract_sha256'], public_verification_sha256=sha(proof_path),
        platform_timeout_seconds=43200, internal_watchdog_seconds=36000, quota=quota, budget=budget,
        submission_statuses=[dict(ref=r['ref'], status=r['status'], publicScore=r['publicScore']) for r in history],
        sequential=sequential, strict_quality_failures_preserved=True, kernel_pushed=False,
        submission_performed=False, source_sha256=sha(Path(__file__)))
    if not args.execute:
        print(json.dumps(result), flush=True)
        return
    result['status'] = 'launch_requested'
    receipt.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    try:
        pushed = subprocess.run([cli, 'kernels', 'push', '-p', '.', '-t', '43200'], cwd=stage,
            capture_output=True, text=True, encoding='utf-8', env=env, timeout=120)
        result.update(status='push_returned', exit_code=pushed.returncode, stdout=pushed.stdout,
                      stderr=pushed.stderr, kernel_pushed=pushed.returncode == 0)
        assert pushed.returncode == 0, 'Inspect launch response before any retry'
    except BaseException as error:
        result.update(status='launch_outcome_requires_inspection', error=repr(error))
        raise
    finally:
        receipt.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
