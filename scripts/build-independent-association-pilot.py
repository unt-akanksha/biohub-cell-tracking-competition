"""Prepare the bounded, hash-bound association-only GPU smoke."""
import ast
import argparse
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT/'kaggle/biohub-independent-association-pilot-v1'
CHECKPOINT_SHA = '51664e6817519745330c3687c81aee50bf321b0fd59ded97e7fd79a91f21e34c'


def build(steps=100):
    if steps not in (100,1000):
        raise ValueError('Only smoke or bounded optimization profiles allowed')
    base = runpy.run_path(str(ROOT/'scripts/build-independent-selection-inference.py'))
    nb,meta = base['build'](CHECKPOINT_SHA,4)
    source = ''.join(nb['cells'][1]['source'])
    assignment = next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id == 'runtime_sources')
    runtime = ast.literal_eval(assignment.value)
    runtime['run_association.py'] = (ROOT/'scripts/run-independent-association-pilot.py').read_text(encoding='utf-8')
    runtime['sparse_parent_loss.py'] = (ROOT/'research/sparse_parent_loss.py').read_text(encoding='utf-8')
    runtime['seeded_frame_dataset.py'] = (ROOT/'research/seeded_frame_dataset.py').read_text(encoding='utf-8')
    source = source.replace(ast.get_source_segment(source,assignment),'runtime_sources = '+repr(runtime))
    nb['cells'][1]['source'] = source.splitlines(keepends=True)
    for index in (0,len(nb['cells'])-1):
        source = ''.join(nb['cells'][index]['source'])
        source = source.replace("run_id='independent-selection-inference-v1'","run_id='independent-association-pilot-v1'")
        source = source.replace("Path('/kaggle/working/independent_selection')","Path('/kaggle/working/independent_association')")
        source = source.replace("runtime/'run_selection.py'","runtime/'run_association.py'")
        source = source.replace("'--checkpoint',str(checkpoint)",f"'--steps','{steps}','--checkpoint',str(checkpoint)")
        nb['cells'][index]['source'] = source.splitlines(keepends=True)
    for cell in nb['cells']:
        ast.parse(''.join(cell['source']))
    nb['metadata']['codex'].update(run_id='independent-association-pilot-v1',max_steps=steps,
        detector_frozen=True,training_scope='Same four source movies; no selection or target audit')
    meta.update(id='indarkarhana/'+TARGET.name,title='Biohub Independent Association Pilot v1',
        code_file=TARGET.name+'.ipynb')
    return nb,meta


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps',type=int,choices=(100,1000),default=100)
    args = parser.parse_args()
    nb,meta = build(args.steps)
    TARGET.mkdir(parents=True,exist_ok=True)
    (TARGET/meta['code_file']).write_text(json.dumps(nb))
    (TARGET/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    print(TARGET)
