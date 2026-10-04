"""Real end-to-end source-fit smoke: 2+2 resumed updates must equal direct4.

Leaves the two real source fits safely paused at step4 for later continuation.
Control-smoke weights are never candidates or selected models.
"""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.trajectory_runtime_v1 import sha


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def saved_checkpoint(folder):
    pointer = read(folder / 'latest-checkpoint.json')
    path = folder / pointer['path']
    assert path.parent == folder and sha(path) == pointer['sha256']
    return path, read(path)


def main():
    reports = ROOT / 'reports/experiments'
    receipt = reports / 'trajectory-event-fork16-resume-v1-audit.json'
    assert not receipt.exists(), 'Inspect existing smoke; never duplicate paused fits'
    corpus_path = reports / 'trajectory-event-fork16-cases-controller-v1.json'
    corpus = read(corpus_path)
    assert corpus['status'] == 'complete_source_corpus_prepared'
    assert corpus['total_cases'] == 5780 and not corpus['failed']
    training = ROOT / 'scripts/train-trajectory-event-fork16-source-v1.py'
    for embryo in ('44b6', '6bba'):
        for suffix in ('', '-control-smoke'):
            name = 'trajectory-event-fork16-fit-' + embryo + '-v1' + suffix
            assert not (ROOT / '.biohub/cache' / name).exists() and not (reports / (name + '.json')).exists()
    started = time.monotonic()
    result = dict(status='running', source_only=True, gpu_used=False,
        selection_or_validation_opened=False, model_selected=False, production_candidate_changed=False,
        corpus_sha256=sha(corpus_path), training_script_sha256=sha(training),
        source_sha256=sha(Path(__file__)), invocations=[], embryos={})
    def persist():
        result['seconds'] = time.monotonic() - started
        receipt.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    persist()
    try:
        for embryo in ('44b6', '6bba'):
            for arguments in (['--steps-limit', '2'], ['--resume', '--steps-limit', '2'],
                              ['--control-smoke', '--steps-limit', '4']):
                assert sha(training) == result['training_script_sha256']
                tick = time.monotonic()
                completed = subprocess.run([sys.executable, str(training), '--embryo', embryo, *arguments],
                    cwd=ROOT, capture_output=True, text=True, encoding='utf-8', timeout=600)
                row = dict(embryo=embryo, arguments=arguments, exit_code=completed.returncode,
                    stdout=completed.stdout, stderr=completed.stderr, seconds=time.monotonic()-tick)
                result['invocations'].append(row)
                persist()
                assert completed.returncode == 0, 'Inspect recorded child outcome before another invocation'
                print(json.dumps({k:v for k,v in row.items() if k not in ('stdout', 'stderr')}), flush=True)
            name = 'trajectory-event-fork16-fit-' + embryo + '-v1'
            folder = ROOT / '.biohub/cache' / name
            control = ROOT / '.biohub/cache' / (name + '-control-smoke')
            path, resumed = saved_checkpoint(folder)
            control_path, direct = saved_checkpoint(control)
            assert resumed == direct and sha(path) == sha(control_path), 'Real resumed checkpoint differs from uninterrupted control'
            assert resumed['optimizer']['steps'] == 4
            report = read(reports / (name + '.json'))
            assert report['status'] == 'paused_at_step_limit' and report['steps'] == 4
            result['embryos'][embryo] = dict(exact_resumed_vs_uninterrupted=True, checkpoint_sha256=sha(path),
                contract_sha256=sha(folder / 'CONTRACT.json'), paused_fit_receipt_sha256=sha(reports / (name + '.json')),
                cases=read(folder / 'CONTRACT.json')['cases'], source_movies=len(read(folder / 'CONTRACT.json')['source_stems']))
            persist()
        result['status'] = 'real_both_embryos_exact_resume_passed'
        persist()
        print(json.dumps({k:v for k,v in result.items() if k != 'invocations'}), flush=True)
    except BaseException as error:
        result.update(status='resume_smoke_failed_requires_inspection', error=repr(error))
        persist()
        raise


if __name__ == '__main__':
    main()
