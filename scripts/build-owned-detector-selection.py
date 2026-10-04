"""Eight complete source movies for one VERIFIED arm of the detector pair."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import runpy
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research'))
from owned_detector_logit_targets import contract
INITIAL='76f7da6e32c901e3e3f9d2ab870b8a22235b0b91086cfcf909aa41674896a144'


def build(arm):
    if arm not in ('sparse','pu'): raise ValueError('Specify one frozen paired arm')
    path=ROOT/'reports/experiments/owned-detector-fit-pair-v1-result.json'
    report=json.loads(path.read_text())
    if (report['status']!='verified_detector_fit_pair_not_selection' or not report['paired_inputs_identical']
        or not report['paired_targets_identical'] or report['authorized_for_submission'] is not False
        or report['arms'][arm]['steps']!=1000 or not report['arms'][arm]['frozen_components_unchanged']):
        raise ValueError('Verified complete detector-pair training required')
    checkpoint=report['arms'][arm]['checkpoint_sha256']
    run_id=f'owned-detector-{arm}-selection-v1'; target=ROOT/'kaggle'/('biohub-'+run_id)
    nb,meta=runpy.run_path(str(ROOT/'scripts/build-detector-spatial-tta-selection.py'))['build']()
    source=''.join(nb['cells'][1]['source'])
    assignment=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='runtime_sources')
    runtime=ast.literal_eval(assignment.value)
    runtime['run_selection.py']=runtime['run_selection.py'].replace('detector-spatial-tta-selection-v1',run_id)
    for name in ('owned_detector_pu','owned_detector_logit_targets'):
        runtime[name+'.py']=(ROOT/f'research/{name}.py').read_text()
    runtime['spotiflow_biohub/pu_targets.py']=(ROOT/'research/spotiflow_biohub/pu_targets.py').read_text()
    runtime['raw_probe_result.json']=(ROOT/'.biohub/cache/kernel-outputs/owned-detector-logit-probe-v1/owned_detector_logit_probe/outputs/result.json').read_bytes().decode()
    source=source.replace(ast.get_source_segment(source,assignment),'runtime_sources = '+repr(runtime))
    nb['cells'][1]['source']=source.splitlines(keepends=True)
    for index in (0,len(nb['cells'])-1):
        source=''.join(nb['cells'][index]['source']).replace('detector-spatial-tta-selection-v1',run_id)
        source=source.replace('detector_spatial_tta_selection',f'owned_detector_{arm}_selection')
        if index==len(nb['cells'])-1:
            source=source.replace('biohub-image-motion-linker-v1','biohub-owned-detector-fit-pair-v1')
            source=source.replace('image_motion_linker/outputs/last.pt',f'owned_detector_fit_pair/outputs/{arm}/last.pt')
            if source.count(INITIAL)!=1: raise ValueError('Initialization hash marker changed')
            source=source.replace(INITIAL,checkpoint)
        nb['cells'][index]['source']=source.splitlines(keepends=True)
    receipt=dict(version=1,objective=arm,steps=1000,initialization_sha256=INITIAL,target_contract=contract(),
        frozen_modules=['transformer','flow','batchnorm_statistics'],
        probe_result_sha256='889a94d0a7bb60d40bde3a42a1343072a33cc68a0680dd0d083c41814efaf2ba')
    nb['metadata']['codex'].update(run_id=run_id,checkpoint_sha256=checkpoint,checkpoint_version=1,
        owned_detector_fit=receipt,fit_pair_report_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    meta.update(id='indarkarhana/'+target.name,title=f'Biohub Owned Detector {arm.upper()} Selection v1',code_file=target.name+'.ipynb',
        kernel_sources=['indarkarhana/biohub-owned-detector-fit-pair-v1/1'])
    for cell in nb['cells']: ast.parse(''.join(cell['source']))
    return nb,meta,target


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--arm',choices=('sparse','pu'),required=True)
    nb,meta,target=build(parser.parse_args().arm); target.mkdir(parents=True,exist_ok=True)
    (target/meta['code_file']).write_text(json.dumps(nb),encoding='utf-8')
    (target/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8'); print(target)
