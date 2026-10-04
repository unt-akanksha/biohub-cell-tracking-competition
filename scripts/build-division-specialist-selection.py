"""Full-movie specialist inference with the b642 native-node reference."""
import argparse
import ast
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT/'kaggle/biohub-division-specialist-selection-v1'


def build(sha,version):
    nb,meta = runpy.run_path(str(ROOT/'scripts/build-independent-known-null-selection.py'))['build'](sha,version)
    for cell in nb['cells']:
        source = ''.join(cell['source']).replace('independent-known-null-selection-v1','division-specialist-selection-v1')
        source = source.replace('/kaggle/working/independent_known_null_selection','/kaggle/working/division_specialist_selection')
        cell['source'] = source.splitlines(keepends=True)
    launch = ''.join(nb['cells'][-1]['source'])
    launch = launch.replace('biohub-independent-known-null-v1','biohub-independent-division-specialist-v1')
    launch = launch.replace('independent_known_null/outputs/last.pt','independent_division_specialist/outputs/last.pt')
    launch = launch.replace('biohub-independent-joint-selection-v1','biohub-independent-known-null-selection-v1')
    launch = launch.replace("p/'independent_joint_selection'","p/'independent_known_null_selection'")
    nb['cells'][-1]['source'] = launch.splitlines(keepends=True)
    nb['metadata']['codex'].update(run_id='division-specialist-selection-v1',division_specialist=True,
        node_reference_kernel='indarkarhana/biohub-independent-known-null-selection-v1/1')
    meta.update(id='indarkarhana/'+TARGET.name,title='Biohub Division Specialist Selection v1',
        code_file=TARGET.name+'.ipynb',kernel_sources=[f'indarkarhana/biohub-independent-division-specialist-v1/{version}',
        'indarkarhana/biohub-independent-known-null-selection-v1/1'])
    for cell in nb['cells']:
        ast.parse(''.join(cell['source']))
    return nb,meta


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sha256',required=True)
    parser.add_argument('--version',type=int,required=True)
    args = parser.parse_args()
    nb,meta = build(args.sha256,args.version)
    TARGET.mkdir(parents=True,exist_ok=True)
    (TARGET/meta['code_file']).write_text(json.dumps(nb),encoding='utf-8')
    (TARGET/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(TARGET)
