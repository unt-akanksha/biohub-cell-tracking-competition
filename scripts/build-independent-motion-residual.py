"""Small gated GPU experiment: frozen detector, motion prior, learned residual."""
import ast
import argparse
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT/'kaggle/biohub-independent-motion-residual-v1'


def build(row_negatives=False):
    run_id = 'independent-motion-row-v1' if row_negatives else 'independent-motion-residual-v1'
    target_name = 'biohub-'+run_id
    work_name = 'independent_motion_row' if row_negatives else 'independent_motion_residual'
    base = runpy.run_path(str(ROOT/'scripts/build-independent-association-pilot.py'))
    nb,meta = base['build'](100)
    source = ''.join(nb['cells'][1]['source'])
    assignment = next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id == 'runtime_sources')
    runtime = ast.literal_eval(assignment.value)
    for target,path in [('motion_residual.py','research/motion_residual.py'),
                        ('independent_motion_prior.py','research/independent_motion_prior.py'),
                        ('tests/test_motion_residual.py','tests/test_motion_residual.py')]:
        runtime[target] = (ROOT/path).read_text(encoding='utf-8')
    if row_negatives:
        runtime['sparse_parent_row_loss.py'] = (ROOT/'research/sparse_parent_row_loss.py').read_text(encoding='utf-8')
        runtime['tests/test_sparse_parent_row_loss.py'] = (ROOT/'tests/test_sparse_parent_row_loss.py').read_text(encoding='utf-8')
        runtime['tests/test_seeded_frame_dataset.py'] = (ROOT/'tests/test_seeded_frame_dataset.py').read_text(encoding='utf-8')
        runtime['run_row_pair.py'] = (ROOT/'scripts/run-independent-motion-row-pair.py').read_text(encoding='utf-8')
    runtime['run_association.py'] = runtime['run_association.py'].replace(
        "run_id='independent-association-pilot-v1'",f"run_id={run_id!r}")
    source = source.replace(ast.get_source_segment(source,assignment),'runtime_sources = '+repr(runtime))
    nb['cells'][1]['source'] = source.splitlines(keepends=True)
    for index in (0,len(nb['cells'])-1):
        source = ''.join(nb['cells'][index]['source']).replace(
            "run_id='independent-association-pilot-v1'",f"run_id={run_id!r}")
        source = source.replace("Path('/kaggle/working/independent_association')",
                                f"Path('/kaggle/working/{work_name}')")
        if index:
            gate = '''gate = subprocess.run([sys.executable,'-m','pytest','tests/test_motion_residual.py','-q'],
    cwd=runtime,timeout=180,text=True,capture_output=True)
print(gate.stdout,flush=True)
print(gate.stderr,flush=True)
if gate.returncode != 0:
    raise RuntimeError('Motion residual numerical gate failed')
command.append('--motion-residual')
'''
            if row_negatives:
                gate = gate.replace("'tests/test_motion_residual.py','-q'",
                    "'tests/test_motion_residual.py','tests/test_sparse_parent_row_loss.py','tests/test_seeded_frame_dataset.py','-q'")
                gate += "command.append('--row-negatives')\n"
                source = source.replace("runtime/'run_association.py'","runtime/'run_row_pair.py'")
            source = source.replace('import signal\n',gate+'import signal\n')
        nb['cells'][index]['source'] = source.splitlines(keepends=True)
    for cell in nb['cells']:
        ast.parse(''.join(cell['source']))
    nb['metadata']['codex'].update(run_id=run_id,motion_residual=True,row_negatives=row_negatives)
    meta.update(id='indarkarhana/'+target_name,title='Biohub Independent Motion Row v1' if row_negatives else 'Biohub Independent Motion Residual v1',
        code_file=target_name+'.ipynb')
    return nb,meta


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--row-negatives',action='store_true')
    args = parser.parse_args()
    nb,meta = build(args.row_negatives)
    target = ROOT/'kaggle'/meta['id'].split('/')[1]
    target.mkdir(parents=True,exist_ok=True)
    (target/meta['code_file']).write_text(json.dumps(nb))
    (target/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    print(target)
