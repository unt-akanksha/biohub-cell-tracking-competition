"""Prepare hash-bound source-selection inference; never launch it here."""
import argparse
import ast
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
BASE = runpy.run_path(str(ROOT/'scripts/build-independent-real-pilot.py'))
TARGET = ROOT/'kaggle/biohub-independent-selection-inference-v1'


def build(checkpoint_sha256, version):
    if len(checkpoint_sha256) != 64 or any(c not in '0123456789abcdef' for c in checkpoint_sha256) or version < 1:
        raise ValueError('Verified checkpoint SHA and completed kernel version required')
    notebook, metadata = BASE['build']()
    source = ''.join(notebook['cells'][1]['source'])
    assignment = next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
                      and isinstance(n.targets[0],ast.Name) and n.targets[0].id == 'runtime_sources')
    runtime = ast.literal_eval(assignment.value)
    runtime['run_selection.py'] = (ROOT/'scripts/run-independent-selection-inference.py').read_text()
    source = source.replace(ast.get_source_segment(source,assignment),'runtime_sources = '+repr(runtime))
    notebook['cells'][1]['source'] = source.splitlines(keepends=True)
    launch = ''.join(notebook['cells'][-1]['source'])
    start, end = launch.index('command = ['), launch.index('import signal\n')
    command = f'''checkpoint_candidates = [Path('/kaggle/input')/p/'independent_real_pilot/outputs/last.pt'
    for p in ('biohub-independent-real-pilot-v1',
              'notebooks/indarkarhana/biohub-independent-real-pilot-v1',
              'kernels/indarkarhana/biohub-independent-real-pilot-v1')]
checkpoint = next((p for p in checkpoint_candidates if p.is_file()),None)
if checkpoint is None:
    raise RuntimeError('Completed optimization checkpoint input not mounted')
command = [sys.executable,'-u',str(runtime/'run_selection.py'),
    '--repo',str(repo),'--runtime',str(runtime),'--manifest',str(runtime/'split.json'),
    '--data',str(data),'--output',str(work/'outputs'),
    '--checkpoint',str(checkpoint),'--sha256',{checkpoint_sha256!r}]
'''
    notebook['cells'][-1]['source'] = (launch[:start]+command+launch[end:]).splitlines(keepends=True)
    for index in (0,len(notebook['cells'])-1):
        source = ''.join(notebook['cells'][index]['source']).replace(
            "run_id='independent-real-pilot-v1'","run_id='independent-selection-inference-v1'")
        source = source.replace("Path('/kaggle/working/independent_real_pilot')",
                                "Path('/kaggle/working/independent_selection')")
        notebook['cells'][index]['source'] = source.splitlines(keepends=True)
    for cell in notebook['cells']:
        ast.parse(''.join(cell['source']))
    notebook['metadata']['codex'] = dict(run_id='independent-selection-inference-v1',
        checkpoint_sha256=checkpoint_sha256,checkpoint_version=version,declared_budget_seconds=3600,
        target_audit_opened=False,authorized_for_submission=False)
    metadata.update(id='indarkarhana/'+TARGET.name,title='Biohub Independent Selection Inference v1',
        code_file=TARGET.name+'.ipynb',kernel_sources=[f'indarkarhana/biohub-independent-real-pilot-v1/{version}'])
    return notebook, metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sha256',required=True)
    parser.add_argument('--version',type=int,required=True)
    args = parser.parse_args()
    notebook, metadata = build(args.sha256,args.version)
    TARGET.mkdir(parents=True,exist_ok=True)
    (TARGET/metadata['code_file']).write_text(json.dumps(notebook))
    (TARGET/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
    print(TARGET)
