"""Audit exact frozen structured candidate; optional one-shot Kaggle submission."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.trajectory_runtime_v1 import sha, assemble_csv
from research.submission_sharding import validate_submission_kernel_metadata
from research.trajectory_structured_release_v1 import (
    check_graph_stages, load_portable, replay_movie, runtime_evidence)

KERNEL = 'indarkarhana/biohub-structured-trajectory-candidate'
COMPETITION = 'biohub-cell-tracking-during-development'
CONTRACT = '4e13ec7dea134c2da7b13133d8ed5ba92d5bb392954f35e26c29288c446a4695'
BASE_PYTHON = Path('C:/Users/IndarKumar/AppData/Local/Programs/Python/Python312/python.exe')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_source(staging, remote, build):
    expected = read(staging / 'kernel-metadata.json')
    observed = read(remote / 'kernel-metadata.json')
    validate_submission_kernel_metadata(expected)
    require(expected['id'] == KERNEL, 'Wrong local kernel')
    require(sha(staging / expected['code_file']) == build['notebook_sha256'], 'Frozen notebook changed')
    for key in ('id', 'title', 'language', 'kernel_type', 'is_private', 'enable_gpu',
                'enable_tpu', 'enable_internet', 'dataset_sources', 'kernel_sources',
                'competition_sources', 'model_sources', 'docker_image', 'machine_shape'):
        require(expected[key] == observed[key], 'Remote setting differs: ' + key)
    def cells(path):
        return [''.join(c['source']) for c in read(path)['cells'] if c['cell_type'] == 'code']
    require(cells(staging / expected['code_file']) == cells(remote / observed['code_file']),
            'Remote code differs from frozen candidate')
    return expected, dict(notebook_sha256=sha(remote / observed['code_file']),
                          metadata_sha256=sha(remote / 'kernel-metadata.json'), code_cells_exact=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--outputs', type=Path, required=True)
    p.add_argument('--terminal-sha256', required=True)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--runtime-risk-disclosed', action='store_true')
    args = p.parse_args()
    receipt = ROOT / 'reports/experiments/trajectory-structured-v1-submission.json'
    require(not receipt.exists(), 'Submission already attempted; inspect outcome, do not retry blindly')
    build = read(ROOT / 'reports/experiments/trajectory-structured-production-v1-build.json')
    launch = read(ROOT / 'reports/experiments/trajectory-structured-production-v1-launch.json')
    quality_path = ROOT / 'reports/experiments/trajectory-structured-kaggle-v1-result.json'
    quality = read(quality_path)
    require(sha(quality_path) == build['acceptance_sha256'] == launch['acceptance_sha256'], 'Acceptance changed')
    require(build['contract_sha256'] == launch['contract_sha256'] == quality['contract_sha256'] == CONTRACT,
            'Runtime contract changed')
    require(launch['exit_code'] == 0 and launch['notebook_sha256'] == build['notebook_sha256'], 'Launch differs')
    require(quality['status'] == 'acceptance_passed' and quality['comparison']['diagnostic_gate_passed']
            and quality['production_notebook_eligible'], 'Quality acceptance required')
    metadata, remote_source = verify_source(
        ROOT / '.biohub/staging/biohub-structured-trajectory-candidate-v1',
        ROOT / '.biohub/cache/trajectory-structured-production-v1-source', build)
    folder = args.outputs / 'trajectory-complete'
    terminal_path = folder / 'result.json'
    require(sha(terminal_path) == args.terminal_sha256, 'Downloaded terminal changed')
    terminal = read(terminal_path)
    smoke = read(args.outputs / 'trajectory-smoke/result.json')
    for result, mode in ((terminal, 'production'), (smoke, 'smoke')):
        require(result['status'] == 'complete' and result['mode'] == mode
                and result['gpu_count'] == 2 and result['worker_count'] == 4
                and result['contract_sha256'] == CONTRACT and not result['ground_truth_opened']
                and set(result['workers']) == {'0', '1', '2', '3'}, 'Complete same-contract two-T4 run required')
        require(result['submission_created'] == (mode == 'production'), 'Wrong CSV output mode')
        for worker in result['workers'].values():
            require(worker['status'] == ('functionality_passed' if mode == 'smoke' else 'complete_prelabel_predictions')
                    and worker['contract_sha256'] == CONTRACT and worker['inputs_unchanged']
                    and not worker['ground_truth_opened'] and 'T4' in worker['device']
                    and worker['solver_backend'] == 'SCIP', 'Worker incomplete or wrong configuration')
            if mode == 'smoke':
                require(all(r['original']['frames'] == 8 for r in worker['movies'].values()), 'Incomplete smoke')
    plans = {str(s['shard_index']): s for s in terminal['shards']}
    require(len(plans) == 4 and set(plans) == set(terminal['workers']), 'Shard coverage mismatch')
    require({p['cuda_token'] for p in plans.values()} == {'0', '1'}, 'Wrong GPU slots')
    portable, weights = load_portable(ROOT / '.biohub/cache/trajectory-structured-portable-v1')
    outputs, frames, records, times = {}, {}, [], []
    for shard, worker in terminal['workers'].items():
        require(set(worker['movies']) == set(plans[shard]['movie_ids']), 'Worker movie coverage mismatch')
        for movie, arms in worker['movies'].items():
            require(movie not in outputs and set(arms) == {'original'}, 'Duplicate movie or wrong inference arm')
            row = arms['original']
            require(row['frames'] == 100, 'Incomplete public movie')
            movie_folder = folder / ('shard-' + shard) / (movie + '-original')
            base_path = movie_folder / 'prediction.json'
            final_path = movie_folder / 'repaired-prediction.json'
            require(sha(base_path) == row['prediction_sha256'] and sha(final_path) == row['repaired_sha256'],
                    'Frozen graph hashes differ')
            base, motion, final = read(base_path), read(movie_folder / 'motion-repaired-prediction.json'), read(final_path)
            details = read(movie_folder / 'repair-details.json')['structured_assignment']
            changes = check_graph_stages(base, motion, final, row['frames'], row['added_edges'], details)
            replay = replay_movie(movie_folder, portable, weights, final)
            require(replay['changed_edges'] == changes, 'Replay edit count differs')
            outputs[movie], frames[movie] = final_path, row['frames']
            times.append(row['seconds'])
            records.append(dict(movie=movie,sha256=sha(final_path),added_edges=row['added_edges'],
                                swapped_edges=changes,exact_model_replay=True))
    # Local public-output audit only; hidden inference enumerates actual inputs.
    require(set(outputs) == {'44b6_0113de3b', '44b6_0b24845f', '6bba_05b6850b', '6bba_05db0fb1'},
            'All four independently enumerated public-test movies required')
    require(set(outputs) == set(terminal['csv']['movie_rows']), 'CSV coverage differs')
    csv_hash = terminal['csv']['sha256']
    require(sha(folder / 'submission.csv') == sha(args.outputs / 'submission.csv') == csv_hash,
            'Root and assembled submission differ')
    replay_path = args.outputs / 'independent-submission-replay.csv'
    if not replay_path.exists():
        rebuilt = assemble_csv(outputs, frames, replay_path)
        require(rebuilt['rows'] == terminal['csv']['rows'] and rebuilt['movie_rows'] == terminal['csv']['movie_rows'],
                'CSV row counts differ')
    require(sha(replay_path) == csv_hash, 'Independent CSV reconstruction differs')
    risk = runtime_evidence(quality['runtime']['movie_seconds'], times, build['runtime_projection_hours'],
                            terminal['elapsed_seconds'])
    risk['user_informed_before_submission'] = args.runtime_risk_disclosed
    cli = str(BASE_PYTHON.parent / 'Scripts/kaggle.exe')
    env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
    def run(command, timeout=60):
        return subprocess.run(command, capture_output=True, text=True, encoding='utf-8',
                              check=True, timeout=timeout, env=env)
    state = json.loads(run([str(BASE_PYTHON), str(ROOT / 'scripts/get-kaggle-kernel-state.py'),
                            '--kernel-slug', KERNEL]).stdout)
    require(state['present'] and state['current_version_number'] == 1 and state['is_private']
            and state['enable_gpu'] and not state['enable_tpu'] and not state['enable_internet']
            and state['competition_sources'] == metadata['competition_sources']
            and state['dataset_sources'] == metadata['dataset_sources']
            and not state['kernel_sources'] and not state['model_sources'], 'Remote immutable version/settings differ')
    require('KernelWorkerStatus.COMPLETE' in run([cli, 'kernels', 'status', KERNEL]).stdout, 'Kernel not complete')
    result = dict(status='verified_not_submitted', utc=datetime.now(timezone.utc).isoformat(),
                  kernel=KERNEL, kernel_version=1, contract_sha256=CONTRACT,
                  notebook_sha256=build['notebook_sha256'], terminal_sha256=sha(terminal_path),
                  acceptance_sha256=sha(quality_path), csv_sha256=csv_hash, rows=terminal['csv']['rows'],
                  records=records, frames=frames, runtime_risk=risk, remote_state=state,
                  remote_source=remote_source, production_wall_seconds=terminal['elapsed_seconds'],
                  official_validation_score=quality['summaries']['structured']['score'],
                  public_leaderboard_score=None, source_prefilter_movie_regression_preserved=True,
                  public_backbone_training_overlap=True, public_leaderboard_used_for_selection=False,
                  submission_performed=False, verifier_sha256=sha(Path(__file__)),
                  invariant_module_sha256=sha(ROOT / 'research/trajectory_structured_release_v1.py'))
    if not args.execute:
        (ROOT / 'reports/experiments/trajectory-structured-production-v1-verification.json').write_text(
            json.dumps(result, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(result, indent=2))
        return
    require(args.runtime_risk_disclosed, 'Disclose observed runtime uncertainty before submission')
    quota = json.loads(run([cli, 'quota', '--format', 'json']).stdout)
    gpu = [r for r in quota if r['resource'] == 'GPU']
    require(len(gpu) == 1 and float(gpu[0]['remaining'].removesuffix('h')) - 20 >= 8,
            'Conservative two-GPU ten-hour reservation violates eight-hour reserve')
    raw = run([cli, 'competitions', 'submissions', COMPETITION, '--page-size', '50', '--format', 'json']).stdout
    rows = json.loads(raw[raw.index('['):])
    dates = [datetime.fromisoformat(r['date'].replace('Z', '+00:00')).replace(tzinfo=timezone.utc) for r in rows]
    require(dates == sorted(dates, reverse=True), 'Submission history order ambiguous')
    today = datetime.now(timezone.utc).date()
    daily = sum(d.date() == today for d in dates)
    require(daily < 5, 'Daily submission limit reached')
    require(all(str(r['status']).upper().endswith('COMPLETE') or str(r['status']).upper().endswith('ERROR')
                for r in rows), 'Another competition submission is pending; inspect before launch')
    result.update(status='submission_requested', quota=quota, worst_case_reserved_gpu_hours=20,
                  reserve_hours=8, daily_submissions_before=daily)
    receipt.write_text(json.dumps(result, indent=2), encoding='utf-8')
    try:
        response = run([cli, 'competitions', 'submit', COMPETITION, '-k', KERNEL, '-v', '1', '-f', 'submission.csv',
                        '-m', 'Clean licensed image ensemble plus our source-trained AR2 repair and structured assignment; 18 complete-movie checks; offline two-T4'], timeout=120)
        result.update(status='submitted', submission_performed=True, kaggle_response=response.stdout)
    except BaseException as error:
        result.update(status='submission_outcome_requires_inspection', error=type(error).__name__)
        raise
    finally:
        receipt.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
