"""Stage source-selection inference for an explicitly hash-bound residual model."""
import argparse
import ast
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT/'kaggle/biohub-independent-motion-residual-selection-v1'


def build(sha,version):
    base = runpy.run_path(str(ROOT/'scripts/build-independent-selection-inference.py'))
    nb,meta = base['build'](sha,version)
    source = ''.join(nb['cells'][1]['source'])
    assignment = next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id == 'runtime_sources')
    runtime = ast.literal_eval(assignment.value)
    for name in ('motion_residual.py','independent_motion_prior.py'):
        runtime[name] = (ROOT/'research'/name).read_text(encoding='utf-8')
    runtime['run_selection.py'] = runtime['run_selection.py'].replace(
        "run_id='independent-selection-inference-v1'","run_id='independent-motion-residual-selection-v1'")
    source = source.replace(ast.get_source_segment(source,assignment),'runtime_sources = '+repr(runtime))
    nb['cells'][1]['source'] = source.splitlines(keepends=True)
    for index in (0,len(nb['cells'])-1):
        source = ''.join(nb['cells'][index]['source'])
        source = source.replace("run_id='independent-selection-inference-v1'","run_id='independent-motion-residual-selection-v1'")
        source = source.replace('biohub-independent-real-pilot-v1','biohub-independent-motion-residual-v1')
        source = source.replace('independent_real_pilot/outputs/last.pt','independent_motion_residual/outputs/last.pt')
        source = source.replace("Path('/kaggle/working/independent_selection')",
                                "Path('/kaggle/working/independent_motion_residual_selection')")
        nb['cells'][index]['source'] = source.splitlines(keepends=True)
    nb['metadata']['codex']['run_id'] = 'independent-motion-residual-selection-v1'
    meta.update(id='indarkarhana/'+TARGET.name,title='Biohub Independent Motion Residual Selection v1',
        code_file=TARGET.name+'.ipynb',kernel_sources=[f'indarkarhana/biohub-independent-motion-residual-v1/{version}'])
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
    (TARGET/meta['code_file']).write_text(json.dumps(nb))
    (TARGET/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    print(TARGET)
