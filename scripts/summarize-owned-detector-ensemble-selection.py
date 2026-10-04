"""Compare complete fixed ensemble source validation with frozen parent D4."""
import hashlib
import argparse
import json
import math
from pathlib import Path
import runpy

ROOT=Path(__file__).resolve().parents[1]
RUN='owned-detector-ensemble-selection-v1'
WORK='owned_detector_ensemble_selection'


def summarize(score_path=None,execution_path=None):
    base=ROOT/'reports/experiments/detector-spatial-tta-selection-v1-score.json'
    comparison=runpy.run_path(str(ROOT/'scripts/summarize-owned-detector-selection.py'))
    if hashlib.sha256(base.read_bytes()).hexdigest()!=comparison['BASE_SHA']:
        raise ValueError('Frozen baseline changed')
    cache=ROOT/'.biohub/cache/kernel-outputs'
    paths=dict(score=cache/(RUN+'-scoring')/(WORK+'_score')/'selection_score.json',
        manifest=cache/RUN/WORK/'outputs/selection_manifest.json',
        terminal=cache/RUN/WORK/'launcher_terminal.json',baseline=base,
        notebook=ROOT/f'kaggle/biohub-{RUN}/biohub-{RUN}.ipynb',
        probe=ROOT/'reports/experiments/owned-detector-ensemble-probe-v1-result.json')
    cpu_state=None
    if score_path is not None:
        if execution_path is None: raise ValueError('Local CPU execution receipt required')
        execution=json.loads(execution_path.read_text())
        if (execution['status']!='completed' or execution['platform']!='local_cpu'
            or execution['notebook_sha256']!=hashlib.sha256(paths['notebook'].read_bytes()).hexdigest()
            or execution['score_sha256']!=hashlib.sha256(score_path.read_bytes()).hexdigest()):
            raise ValueError('Completed hash-bound local CPU execution required')
        paths.update(score=score_path,local_cpu_execution=execution_path)
    else:
        # This frozen CPU notebook emits a terminal file only on timeout.
        # Establish successful completion from authoritative remote state.
        helper=runpy.run_path(str(ROOT/'scripts/run-owned-detector-evaluation-queue.py'))
        ref='indarkarhana/biohub-owned-detector-ensemble-scoring-v1'
        remote=helper['INSPECT'](ref)
        status=helper['status_value'](helper['command'](['kaggle','kernels','status',ref]),ref)
        if not remote['present'] or remote['current_version_number']!=1 or status!='COMPLETE':
            raise ValueError('Completed exact CPU scorer version required')
        cpu_state=dict(kernel_ref=ref+'/1',status=status,version=1)
    score,manifest,terminal,nb=(json.loads(paths[k].read_text()) for k in ('score','manifest','terminal','notebook'))
    split_path=ROOT/'research/independent_real_baseline_v1_split.json'
    split=json.loads(split_path.read_text())
    expected=runpy.run_path(str(ROOT/'research/owned_detector_ensemble_contract.py'))['receipt'](paths['probe'].read_bytes(),split_path.read_bytes())
    if nb['metadata']['codex']['owned_detector_ensemble']!=expected:
        raise ValueError('Different ensemble recipe')
    runpy.run_path(str(ROOT/'scripts/score-independent-selection.py'))['verify_manifest'](manifest,terminal,nb['metadata']['codex'],split)
    if (score['status']!='scored_complete_selection' or score['source_manifest_run_id']!=RUN
        or score['checkpoint_sha256']!=expected['parent_sha256']
        or [r['stem'] for r in score['per_movie']]!=split['folds'][0]['selection']
        or any(r['processed_frames']!=100 or r['image_shape'][0]!=100 for r in manifest['records'])
        or any(score[k] is not False for k in ('target_audit_opened','authorized_for_submission'))
        or any(not math.isfinite(score['summary'][k]) for k in comparison['FIELDS'])):
        raise ValueError('Complete finite source-only ensemble score required')
    reference=json.loads(base.read_text())['result']
    delta=comparison['difference'](score,reference)
    report=dict(status='verified_owned_detector_ensemble_source_comparison',result=score,
        cpu_execution='local_pinned_official_scorer' if score_path is not None else 'kaggle_cpu_scorer',
        cpu_kernel_ref=None if score_path is not None else 'indarkarhana/biohub-owned-detector-ensemble-scoring-v1/1',
        verified_cpu_state=cpu_state,
        versus_frozen_d4=delta,counts={name:{k:sum(r[k] for r in s['per_movie']) for k in comparison['COUNTS']}
            for name,s in (('baseline',reference),('ensemble',score))},
        authorized_for_submission=False,new_target_movies_opened=0,
        source_sha256={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()},
        caveat='Repeated source-selection development evidence, not an independent private-test estimate. No new target access or submission authorized by this report.')
    target=ROOT/f'reports/experiments/{RUN}-result.json'
    if target.exists(): raise ValueError('Refuse to overwrite completed comparison')
    target.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='result'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--score-path',type=Path)
    parser.add_argument('--execution-path',type=Path)
    args=parser.parse_args()
    summarize(args.score_path,args.execution_path)
