"""Frozen-detector real-data gate for annotation-backed missing-parent nulls."""
import argparse
import ast
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT/'kaggle/biohub-independent-known-null-v1'
CHECKPOINT_SHA = 'c5023345d31d91929a8d05219310a9edf1aeecf576d7593cbc5e65208c76b470'


def build(steps=100):
    if steps not in (100, 1000):
        raise ValueError('Only small gate or bounded extension permitted')
    base = runpy.run_path(str(ROOT/'scripts/build-independent-joint-broad.py'))
    nb, meta = base['build'](100)
    source = ''.join(nb['cells'][1]['source'])
    assignment = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                      and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'runtime_sources')
    runtime = ast.literal_eval(assignment.value)
    runtime['run_association.py'] = runtime['run_association.py'].replace(
        "run_id='independent-joint-broad-v1'", "run_id='independent-known-null-v1'")
    for name, path in [('annotated_missing_parent.py', 'research/annotated_missing_parent.py'),
                       ('sparse_parent_missing_null.py', 'research/sparse_parent_missing_null.py'),
                       ('tests/test_sparse_parent_missing_null.py', 'tests/test_sparse_parent_missing_null.py')]:
        runtime[name] = (ROOT/path).read_text(encoding='utf-8')
    runtime['tests/test_sparse_parent_missing_null.py'] = runtime['tests/test_sparse_parent_missing_null.py'].replace(
        "str(Path(__file__).resolve().parents[1]/'research')", "str(Path(__file__).resolve().parents[1])")
    source = source.replace(ast.get_source_segment(source, assignment), 'runtime_sources = '+repr(runtime))
    nb['cells'][1]['source'] = source.splitlines(keepends=True)
    for index in (0, len(nb['cells'])-1):
        source = ''.join(nb['cells'][index]['source']).replace('independent-joint-broad-v1', 'independent-known-null-v1')
        source = source.replace('independent_joint_broad', 'independent_known_null')
        nb['cells'][index]['source'] = source.splitlines(keepends=True)
    launch = ''.join(nb['cells'][-1]['source'])
    launch = launch.replace('biohub-independent-motion-broad-pair-v1', 'biohub-independent-joint-broad-v1')
    launch = launch.replace('independent_motion_broad_pair/outputs/control/last.pt', 'independent_joint_broad/outputs/last.pt')
    if launch.count(base['CHECKPOINT_SHA']) != 1:
        raise ValueError('Initialization hash marker changed')
    launch = launch.replace(base['CHECKPOINT_SHA'], CHECKPOINT_SHA)
    launch = launch.replace("'--steps','100'", f"'--steps','{steps}'")
    launch = launch.replace("'--joint'", "'--known-null'")
    marker = 'process = subprocess.Popen(command, start_new_session=True)'
    if launch.count(marker) != 1:
        raise ValueError('Worker launch marker changed')
    launch = launch.replace(marker, "subprocess.run([sys.executable,'-m','pytest',str(runtime/'tests/test_sparse_parent_missing_null.py'),'-q'], cwd=runtime,timeout=180,check=True)\n"+marker)
    nb['cells'][-1]['source'] = launch.splitlines(keepends=True)
    nb['metadata']['codex'].update(run_id='independent-known-null-v1', max_steps=steps,
        checkpoint_sha256=CHECKPOINT_SHA, detector_frozen=True, joint_training=False,
        known_null=True, step100_gate_required=True)
    meta.update(id='indarkarhana/'+TARGET.name, title='Biohub Independent Known Null v1',
                code_file=TARGET.name+'.ipynb', kernel_sources=['indarkarhana/biohub-independent-joint-broad-v1/1'])
    for cell in nb['cells']:
        ast.parse(''.join(cell['source']))
    return nb, meta


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, choices=(100, 1000), default=100)
    nb, meta = build(parser.parse_args().steps)
    TARGET.mkdir(parents=True, exist_ok=True)
    (TARGET/meta['code_file']).write_text(json.dumps(nb), encoding='utf-8')
    (TARGET/'kernel-metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(TARGET)
