"""Frozen 120-source-movie paired fit; each arm must pass its 100-update gate."""
import ast
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT/'kaggle/biohub-independent-motion-broad-pair-v1'


def build():
    nb,meta = runpy.run_path(str(ROOT/'scripts/build-independent-motion-residual.py'))['build'](True)
    for cell in nb['cells']:
        source = ''.join(cell['source']).replace('independent-motion-row-v1','independent-motion-broad-pair-v1')
        source = source.replace('/kaggle/working/independent_motion_row','/kaggle/working/independent_motion_broad_pair')
        cell['source'] = source.splitlines(keepends=True)
    launch = ''.join(nb['cells'][-1]['source'])
    if launch.count("'--steps','100'") != 1:
        raise ValueError('Unrecognized small-run launcher')
    launch = launch.replace("'--steps','100'","'--steps','1000','--training-scope','full'")
    nb['cells'][-1]['source'] = launch.splitlines(keepends=True)
    nb['metadata']['codex'].update(run_id='independent-motion-broad-pair-v1',max_steps=1000,
        training_scope='Frozen fold-zero 120 training movies; selection and target audit excluded',
        sequential_arms=['control','row'],step100_gate_required=True)
    meta.update(id='indarkarhana/'+TARGET.name,title='Biohub Independent Motion Broad Pair v1',
                code_file=TARGET.name+'.ipynb')
    for cell in nb['cells']:
        ast.parse(''.join(cell['source']))
    return nb,meta


if __name__ == '__main__':
    nb,meta = build()
    TARGET.mkdir(parents=True,exist_ok=True)
    (TARGET/meta['code_file']).write_text(json.dumps(nb))
    (TARGET/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    print(TARGET)
