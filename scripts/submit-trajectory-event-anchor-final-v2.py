"""Verify version2 against exact-replayed public evidence; optionally submit once.

No model promotion or final-submission selection is performed. Strict quality
failures remain in the referenced exploratory release decision.
"""
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
from research.trajectory_runtime_v1 import sha, assemble_csv
from research.trajectory_structured_release_v1 import check_graph_stages, runtime_evidence
from research.trajectory_event_release_v1 import check_event_stage
from research.trajectory_event_quota_v1 import saved_kernel_complete
from research.trajectory_event_final_budget_v1 import final_budget
from research.submission_sharding import validate_submission_kernel_metadata

BASE = Path('C:/Users/IndarKumar/AppData/Local/Programs/Python/Python312/python.exe')
KERNEL = 'indarkarhana/biohub-event-anchor-candidate'
COMPETITION = 'biohub-cell-tracking-during-development'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    reports = ROOT / 'reports/experiments'
    receipt = reports / 'trajectory-event-anchor-final-v2-submission.json'
    assert not receipt.exists(), 'Submission attempt exists; inspect it, never blindly retry'
    build_path = reports / 'trajectory-event-anchor-final-v2-build.json'
    build = read(build_path)
    launch_path = reports / 'trajectory-event-anchor-final-v2-launch.json'
    launch = read(launch_path)
    proof_path = reports / 'trajectory-event-anchor-production-v1-verification-r2.json'
    proof = read(proof_path)
    assert launch['build_sha256'] == sha(build_path)
    assert launch['kernel_pushed'] and launch['exit_code'] == 0 and launch['platform_timeout_seconds'] == 43200
    assert launch['expected_kernel_version'] == build['expected_kernel_version'] == 2
    assert 'Kernel version 2 successfully pushed' in launch['stdout']
    assert proof['status'] == 'public_production_verified_not_submittable_version'
    assert sha(proof_path) == build['public_verification_sha256'] == launch['public_verification_sha256']
    assert build['quality_decision'] == 'one_exploratory_entry_after_final_runtime_and_output_verification'
    assert build['user_runtime_and_quality_risks_disclosed'] and build['strict_quality_failures_preserved']
    assert not build['strict_quality_gates_pass'] and not build['failed_gate_included']
    assert not build['final_selection_authorized'] and build['executable_cells_identical_to_verified_public_test']
    stage = ROOT / '.biohub/staging/biohub-event-anchor-candidate-v2'
    remote = ROOT / '.biohub/cache/trajectory-event-anchor-final-v2-source'
    expected, observed = read(stage / 'kernel-metadata.json'), read(remote / 'kernel-metadata.json')
    validate_submission_kernel_metadata(expected)
    for key in ('id', 'language', 'kernel_type', 'is_private', 'enable_gpu', 'enable_tpu', 'enable_internet',
                'dataset_sources', 'competition_sources', 'model_sources', 'kernel_sources', 'machine_shape', 'docker_image'):
        assert expected[key] == observed[key], 'Remote metadata differs: ' + key
    assert expected['id'] == KERNEL
    assert sha(stage / expected['code_file']) == build['notebook_sha256'] == launch['notebook_sha256']
    cells = lambda p: [''.join(c['source']) for c in read(p)['cells'] if c['cell_type'] == 'code']
    assert cells(stage / expected['code_file']) == cells(remote / observed['code_file'])
    prior_stage = ROOT / '.biohub/staging/biohub-event-anchor-candidate-v1'
    assert sha(prior_stage / expected['code_file']) == build['public_test_notebook_sha256']
    assert cells(stage / expected['code_file']) == cells(prior_stage / expected['code_file'])
    outputs = ROOT / '.biohub/cache/trajectory-event-anchor-final-v2-output'
    old_outputs = ROOT / '.biohub/cache/trajectory-event-anchor-production-v1-output'
    harvest_path = reports / 'trajectory-event-anchor-final-v2-harvest.json'
    harvest = read(harvest_path)
    old_harvest_path = reports / 'trajectory-event-anchor-production-v1-harvest.json'
    old_harvest = read(old_harvest_path)
    assert sha(old_harvest_path) == proof['harvest_sha256']
    for root, manifest in ((outputs, harvest), (old_outputs, old_harvest)):
        assert manifest['kernel'] == KERNEL and manifest['status'] == 'required_production_evidence_downloaded'
        for name, item in manifest['files'].items():
            assert sha(root / name) == item['sha256'], 'Evidence changed: ' + name
    records, paths, times = [], {}, []
    for folder, mode, nframes in (('trajectory-smoke', 'smoke', 8), ('trajectory-complete', 'production', 100)):
        terminal = read(outputs / folder / 'result.json')
        assert terminal['status'] == 'complete' and terminal['mode'] == mode
        assert terminal['contract_sha256'] == build['contract_sha256'] == launch['contract_sha256']
        assert terminal['gpu_count'] == 2 and terminal['worker_count'] == 4
        assert not terminal['ground_truth_opened'] and terminal['submission_created'] == (mode == 'production')
        plans = {str(s['shard_index']): s for s in terminal['shards']}
        assert set(plans) == set(terminal['workers']) == {'0', '1', '2', '3'}
        assert {s['cuda_token'] for s in plans.values()} == {'0', '1'}
        for shard, worker in terminal['workers'].items():
            assert worker['status'] == ('functionality_passed' if mode == 'smoke' else 'complete_prelabel_predictions')
            assert worker['inputs_unchanged'] and not worker['ground_truth_opened'] and 'T4' in worker['device']
            assert worker['solver_backend'] == 'SCIP' and set(worker['movies']) == set(plans[shard]['movie_ids'])
            for stem, arms in worker['movies'].items():
                assert set(arms) == {'original'} and arms['original']['frames'] == nframes
                row = arms['original']
                relative = Path(folder) / ('shard-' + shard) / (stem + '-original')
                path = outputs / relative
                names = ('pre-postprocess.json', 'prediction.json', 'motion-repaired-prediction.json',
                         'structured-repaired-prediction.json', 'repaired-prediction.json')
                graphs = [read(path / name) for name in names]
                for name, graph in zip(names, graphs):
                    assert graph == read(old_outputs / relative / name), 'Exact verified graph differs: ' + stem + '/' + name
                initial, baseline, moved, assigned, final = graphs
                assert sha(path / 'prediction.json') == row['prediction_sha256']
                assert sha(path / 'repaired-prediction.json') == row['repaired_sha256']
                details = read(path / 'repair-details.json')
                check_graph_stages(baseline, moved, assigned, nframes, row['added_edges'], details['structured_assignment'])
                check_event_stage(initial, assigned, final, details['event_assignment'], frames=nframes)
                assert not details['event_assignment']['budget_exhausted'] and not details['event_assignment']['solver_fallbacks']
                records.append(dict(mode=mode, stem=stem, frames=nframes, all_stages_equal_independently_replayed_version1=True,
                                    final_sha256=row['repaired_sha256'], event_seconds=details['event_assignment']['seconds']))
                if mode == 'production':
                    assert stem not in paths
                    paths[stem] = path / 'repaired-prediction.json'
                    times.append(row['seconds'])
                print(json.dumps(dict(mode=mode, stem=stem, exact_verified_graph=True)), flush=True)
    assert set(paths) == {'44b6_0113de3b', '44b6_0b24845f', '6bba_05b6850b', '6bba_05db0fb1'}
    terminal = read(outputs / 'trajectory-complete/result.json')
    csv_info = terminal['csv']
    cache = ROOT / '.biohub/cache/trajectory-event-anchor-final-v2-verification'
    cache.mkdir(exist_ok=True)
    rebuilt_path = cache / 'submission-replay.csv'
    if not rebuilt_path.exists():
        rebuilt = assemble_csv(paths, {s: 100 for s in paths}, rebuilt_path)
        assert rebuilt['rows'] == csv_info['rows'] and rebuilt['movie_rows'] == csv_info['movie_rows']
    assert sha(rebuilt_path) == sha(outputs / 'submission.csv') == sha(outputs / 'trajectory-complete/submission.csv') == csv_info['sha256'] == proof['csv_sha256']
    accepted_path = reports / 'trajectory-event-anchor-kaggle-v1-result.json'
    assert sha(accepted_path) == proof['acceptance_sha256']
    accepted = read(accepted_path)
    risk = runtime_evidence(accepted['runtime']['movie_seconds'], times,
                           proof['runtime_risk']['validation_cohort_projection_hours'], terminal['elapsed_seconds'])
    env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
    cli = str(BASE.parent / 'Scripts/kaggle.exe')
    def run(command, timeout=60):
        return subprocess.run(command, capture_output=True, text=True, encoding='utf-8', env=env, check=True, timeout=timeout).stdout
    state = json.loads(run([str(BASE), str(ROOT / 'scripts/get-kaggle-kernel-state.py'), '--kernel-slug', KERNEL]))
    assert state['current_version_number'] == 2 and state['is_private'] and state['enable_gpu']
    assert not state['enable_internet'] and not state['enable_tpu']
    assert state['competition_sources'] == expected['competition_sources'] and state['dataset_sources'] == expected['dataset_sources']
    assert saved_kernel_complete(run([cli, 'kernels', 'status', KERNEL]), KERNEL)
    result = dict(status='verified_not_submitted', utc=datetime.now(timezone.utc).isoformat(), kernel=KERNEL, kernel_version=2,
        build_sha256=sha(build_path), launch_sha256=sha(launch_path), public_verification_sha256=sha(proof_path),
        harvest_sha256=sha(harvest_path), notebook_sha256=build['notebook_sha256'],
        contract_sha256=build['contract_sha256'], csv_sha256=csv_info['sha256'], rows=csv_info['rows'],
        records=records, runtime_risk=risk, public_wall_seconds=terminal['elapsed_seconds'],
        remote_source_sha256=sha(remote / observed['code_file']), strict_quality_gates_pass=False,
        strict_quality_failures_preserved=True, exploratory_entry_only=True, final_selection_performed=False,
        submission_performed=False, public_leaderboard_score=None, source_sha256=sha(Path(__file__)))
    if not args.execute:
        verification_path = reports / 'trajectory-event-anchor-final-v2-verification.json'
        assert not verification_path.exists(), 'Verification already exists; use its evidence or inspect before retry'
        verification_path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        print(json.dumps({k:v for k,v in result.items() if k != 'records'}), flush=True)
        return
    history = list(csv.DictReader(io.StringIO(run([cli, 'competitions', 'submissions', '-c', COMPETITION, '--csv']))))
    dates = [datetime.fromisoformat(row['date']).replace(tzinfo=timezone.utc) for row in history]
    assert dates == sorted(dates, reverse=True)
    today = datetime.now(timezone.utc).date()
    daily = sum(d.date() == today for d in dates)
    assert daily < 5 and any(d.date() < today for d in dates), 'Daily inventory incomplete or limit reached'
    assert all(row['status'] in ('SubmissionStatus.COMPLETE', 'SubmissionStatus.ERROR') for row in history)
    quota = json.loads(run([cli, 'quota', '--format', 'json']))
    gpu = [q for q in quota if q['resource'] == 'GPU']
    assert len(gpu) == 1 and gpu[0]['remaining'].endswith('h')
    budget = final_budget(float(gpu[0]['remaining'][:-1]), history, {})
    result.update(status='submission_requested', quota=quota, budget=budget, daily_submissions_before=daily)
    receipt.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    try:
        response = run([cli, 'competitions', 'submit', COMPETITION, '-k', KERNEL, '-v', '2', '-f', 'submission.csv',
            '-m', 'Clean licensed image ensemble plus our fixed event assignment; complete-movie pooled gains with disclosed movie regressions; no selector or metric hack; offline two-T4'], timeout=120)
        result.update(status='submitted', submission_performed=True, kaggle_response=response)
    except BaseException as error:
        result.update(status='submission_outcome_requires_inspection', error=type(error).__name__)
        raise
    finally:
        receipt.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        print(json.dumps({k:v for k,v in result.items() if k != 'records'}), flush=True)


if __name__ == '__main__':
    main()
