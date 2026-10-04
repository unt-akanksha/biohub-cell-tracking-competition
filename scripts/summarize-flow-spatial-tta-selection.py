"""Complete-movie fixed-node motion comparison against retained parent D4."""
import hashlib
import json
import math
from pathlib import Path
import runpy

ROOT=Path(__file__).resolve().parents[1]
RUN='flow-spatial-tta-selection-v1'


def main():
    paths=dict(baseline=ROOT/'reports/experiments/detector-spatial-tta-selection-v1-score.json',
        probe=ROOT/'reports/experiments/backward-flow-spatial-tta-probe-v2-result.json',
        split=ROOT/'research/independent_real_baseline_v1_split.json',
        notebook=ROOT/f'kaggle/biohub-{RUN}/biohub-{RUN}.ipynb',
        manifest=ROOT/f'.biohub/cache/kernel-outputs/{RUN}/flow_spatial_tta_selection/outputs/selection_manifest.json',
        terminal=ROOT/f'.biohub/cache/kernel-outputs/{RUN}/flow_spatial_tta_selection/launcher_terminal.json',
        score=ROOT/'.biohub/cache/kernel-outputs/flow-spatial-tta-scoring-v1/flow_spatial_tta_score/selection_score.json')
    comparator=runpy.run_path(str(ROOT/'scripts/summarize-owned-detector-selection.py'))
    if hashlib.sha256(paths['baseline'].read_bytes()).hexdigest()!=comparator['BASE_SHA']:
        raise ValueError('Retained baseline changed')
    contract=runpy.run_path(str(ROOT/'research/flow_spatial_tta_contract.py'))['receipt'](
        paths['probe'].read_bytes(),paths['split'].read_bytes())
    split,nb,manifest,terminal,score,base=(json.loads(paths[k].read_text()) for k in
        ('split','notebook','manifest','terminal','score','baseline'))
    if nb['metadata']['codex']['flow_spatial_tta']!=contract: raise ValueError('Different motion recipe')
    runpy.run_path(str(ROOT/'scripts/score-independent-selection.py'))['verify_manifest'](manifest,terminal,nb['metadata']['codex'],split)
    helper=runpy.run_path(str(ROOT/'scripts/run-owned-detector-evaluation-queue.py'))
    ref='indarkarhana/biohub-flow-spatial-tta-scoring-v1'
    remote=helper['INSPECT'](ref)
    if (not remote['present'] or remote['current_version_number']!=1
        or helper['status_value'](helper['command'](['kaggle','kernels','status',ref]),ref)!='COMPLETE'):
        raise ValueError('Exact complete CPU scorer required')
    if (score['status']!='scored_complete_selection' or score['source_manifest_run_id']!=RUN
        or score['checkpoint_sha256']!=contract['checkpoint_sha256']
        or [r['stem'] for r in score['per_movie']]!=split['folds'][0]['selection']
        or any(r['processed_frames']!=100 or r['image_shape'][0]!=100 for r in manifest['records'])
        or any(score[k] is not False for k in ('target_audit_opened','authorized_for_submission'))
        or any(not math.isfinite(score['summary'][k]) for k in comparator['FIELDS'])):
        raise ValueError('Complete finite source-only motion score required')
    baseline=base['result']; delta=comparator['difference'](score,baseline)
    result=dict(status='verified_flow_spatial_tta_source_comparison',result=score,
        versus_frozen_d4=delta,cpu_kernel_ref=ref+'/1',cpu_status='COMPLETE',
        counts={name:{k:sum(r[k] for r in s['per_movie']) for k in comparator['COUNTS']}
            for name,s in (('baseline',baseline),('flow_d4',score))},
        source_sha256={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()},
        authorized_for_submission=False,new_target_movies_opened=0,
        caveat='Repeated source development evidence, not an unbiased private-test estimate. No target opening or submission automatically authorized.')
    target=ROOT/'reports/experiments/flow-spatial-tta-selection-v1-result.json'
    if target.exists(): raise ValueError('Refuse to overwrite completed motion comparison')
    target.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='result'},indent=2))


if __name__=='__main__': main()
