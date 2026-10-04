"""CPU source-only controlled screen; target scores require a separate frozen pass."""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from research.comoving_division_head import fit,predict
from research.image_context_quality import metrics


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    start=time.monotonic();root=ROOT/'.biohub/cache/comoving-division-features-v1'
    output=ROOT/'.biohub/cache/comoving-division-head-v1-output'
    if output.exists():raise ValueError('Preserve fitted results')
    manifest=json.loads((root/'MANIFEST.json').read_text())
    if manifest['status']!='complete' or manifest['old_inference_masks_unchanged'] is not True:raise ValueError('Wrong data contract')
    for name,digest in manifest['source_pins'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Prepared source changed')
    pins={name:sha(ROOT/name) for name in ('research/comoving_division_head.py','scripts/run-comoving-division-head-v1.py',
        'reports/experiments/comoving-division-head-v1-design.md')}
    output.mkdir();source_models={};controls={}
    for source in ('44b6','6bba'):
        packets={}
        for role in ('optimization','selection'):
            name=f'{source}-{role}.npz';path=root/name
            if sha(path)!=manifest['files'][name]['sha256']:raise ValueError('Prepared data changed')
            with np.load(path,allow_pickle=False) as a:packets[role]=dict(a)
        folder=output/f'source-{source}';folder.mkdir()
        for with_motion in (False,True):
            name='motion' if with_motion else 'control';tick=time.monotonic()
            head=fit(packets['optimization'],with_motion);scores=predict(packets['selection'],head)
            head_path=folder/f'{name}-head.npz';np.savez_compressed(head_path,**head)
            with np.load(head_path,allow_pickle=False) as a:replay=predict(packets['selection'],dict(a))
            np.testing.assert_array_equal(scores,replay)
            gate=packets['selection']['eligible'].astype(bool)
            row=metrics(packets['selection']['targets'][gate],scores[gate])
            np.savez_compressed(folder/f'{name}-source-selection.npz',scores=scores,targets=packets['selection']['targets'],eligible=gate)
            outcome=dict(source_embryo=source,held_out_embryo='6bba' if source=='44b6' else '44b6',
                head_sha256=sha(head_path),metrics=row,head_parameters=len(head['coefficients'])+1,
                penalty=.01,elapsed_seconds=time.monotonic()-tick)
            (source_models if with_motion else controls)[source]=outcome
            print(json.dumps(dict(event='source_head',variant=name,source=source,**row)),flush=True)
    if any(sha(ROOT/name)!=digest for name,digest in pins.items()):raise ValueError('Frozen fitting code changed')
    passed=all(r['metrics']['source_gate_passed'] for r in source_models.values())
    if passed:(output/'frozen-source-policy.json').write_text(json.dumps(source_models,indent=2)+'\n')
    result=dict(status='passed_source_selection' if passed else 'rejected_source_selection',run_id='comoving-division-head-v1',
        source_models=source_models,controls=controls,source_pins=pins,source_manifest_sha256=sha(root/'MANIFEST.json'),
        old_inference_masks_unchanged=True,held_out_embryo_scores_opened=False,authorized_for_submission=False,
        gpu_hours=0,elapsed_seconds=time.monotonic()-start)
    path=output/'result.json';path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],elapsed_seconds=result['elapsed_seconds'],result_sha256=sha(path))),flush=True)


if __name__=='__main__':main()
