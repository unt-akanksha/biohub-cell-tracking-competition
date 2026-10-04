"""Unfreeze the real detector/features; preserve the parent-only residual model."""
import argparse
import ast
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT/'kaggle/biohub-independent-joint-broad-v1'
CHECKPOINT_SHA = '3b54d1669d93a5a7e2b5d01b51b10b0b62c08e2ccf941abe0dbabb52121f3cea'


def build(steps=1000):
    if steps not in (100,1000):
        raise ValueError('Only small gate or bounded extension allowed')
    nb,meta = runpy.run_path(str(ROOT/'scripts/build-independent-motion-residual.py'))['build']()
    for cell in nb['cells']:
        source = ''.join(cell['source']).replace('independent-motion-residual-v1','independent-joint-broad-v1')
        source = source.replace('/kaggle/working/independent_motion_residual','/kaggle/working/independent_joint_broad')
        cell['source'] = source.splitlines(keepends=True)
    launch = ''.join(nb['cells'][-1]['source'])
    launch = launch.replace('biohub-independent-real-pilot-v1','biohub-independent-motion-broad-pair-v1')
    launch = launch.replace('independent_real_pilot/outputs/last.pt','independent_motion_broad_pair/outputs/control/last.pt')
    old_sha = runpy.run_path(str(ROOT/'scripts/build-independent-association-pilot.py'))['CHECKPOINT_SHA']
    if launch.count(old_sha) != 1 or launch.count("'--steps','100'") != 1:
        raise ValueError('Unexpected initialization launcher')
    launch = launch.replace(old_sha,CHECKPOINT_SHA).replace("'--steps','100'",
        f"'--steps','{steps}','--training-scope','full','--joint'")
    nb['cells'][-1]['source'] = launch.splitlines(keepends=True)
    nb['metadata']['codex'].update(run_id='independent-joint-broad-v1',max_steps=steps,
        detector_frozen=False,joint_training=True,checkpoint_sha256=CHECKPOINT_SHA,checkpoint_version=1,
        training_scope='Frozen120 source training movies; no selection or target audit',
        step100_gate_required=True)
    meta.update(id='indarkarhana/'+TARGET.name,title='Biohub Independent Joint Broad v1',
        code_file=TARGET.name+'.ipynb',kernel_sources=['indarkarhana/biohub-independent-motion-broad-pair-v1/1'])
    for cell in nb['cells']:
        ast.parse(''.join(cell['source']))
    return nb,meta


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps',type=int,choices=(100,1000),default=1000)
    args = parser.parse_args()
    nb,meta = build(args.steps)
    TARGET.mkdir(parents=True,exist_ok=True)
    (TARGET/meta['code_file']).write_text(json.dumps(nb))
    (TARGET/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    print(TARGET)
