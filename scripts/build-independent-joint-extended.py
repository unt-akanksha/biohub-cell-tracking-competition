"""Bounded longer real-domain fit with numerical and real step-100 gates."""
import argparse
import ast
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'kaggle/biohub-independent-joint-extended-v1'
CHECKPOINT_SHA = 'c5023345d31d91929a8d05219310a9edf1aeecf576d7593cbc5e65208c76b470'


def build(steps=6000):
    if steps not in (100, 6000):
        raise ValueError('Only functionality probe or frozen longer profile permitted')
    base = runpy.run_path(str(ROOT / 'scripts/build-independent-joint-broad.py'))
    nb, meta = base['build'](100)
    source = ''.join(nb['cells'][1]['source'])
    assignment = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                      and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'runtime_sources')
    runtime = ast.literal_eval(assignment.value)
    assert runtime['run_association.py'].count("run_id='independent-joint-broad-v1'") == 1
    runtime['run_association.py'] = runtime['run_association.py'].replace(
        "run_id='independent-joint-broad-v1'", "run_id='independent-joint-extended-v1'")
    for name, path in [('checkpoint_bn_guard.py', 'research/checkpoint_bn_guard.py'),
                       ('tests/test_checkpoint_bn_guard.py', 'tests/test_checkpoint_bn_guard.py')]:
        runtime[name] = (ROOT / path).read_text(encoding='utf-8')
    # Test imports support both repository layout and the standalone runtime root.
    runtime['tests/test_checkpoint_bn_guard.py'] = runtime['tests/test_checkpoint_bn_guard.py'].replace(
        "sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'research'))",
        "sys.path.insert(0, str(Path(__file__).resolve().parents[1]))")
    source = source.replace(ast.get_source_segment(source, assignment), 'runtime_sources = ' + repr(runtime))
    nb['cells'][1]['source'] = source.splitlines(keepends=True)
    for index in (0, len(nb['cells']) - 1):
        source = ''.join(nb['cells'][index]['source'])
        source = source.replace('independent-joint-broad-v1', 'independent-joint-extended-v1')
        source = source.replace('independent_joint_broad', 'independent_joint_extended')
        source = source.replace('declared_budget_seconds=3600', 'declared_budget_seconds=7200')
        source = source.replace('threading.Timer(3540,', 'threading.Timer(7140,')
        source = source.replace('3480-(time.monotonic()-started)', '7080-(time.monotonic()-started)')
        nb['cells'][index]['source'] = source.splitlines(keepends=True)
    launch = ''.join(nb['cells'][-1]['source'])
    launch = launch.replace('biohub-independent-motion-broad-pair-v1', 'biohub-independent-joint-broad-v1')
    launch = launch.replace('independent_motion_broad_pair/outputs/control/last.pt', 'independent_joint_broad/outputs/last.pt')
    assert launch.count(base['CHECKPOINT_SHA']) == 1
    launch = launch.replace(base['CHECKPOINT_SHA'], CHECKPOINT_SHA)
    launch = launch.replace("'--steps','100'", f"'--steps','{steps}'")
    launch = launch.replace("'--joint'", "'--joint','--checkpoint-bn-once'")
    marker = 'process = subprocess.Popen(command, start_new_session=True)'
    assert launch.count(marker) == 1
    launch = launch.replace(marker, "subprocess.run([sys.executable,'-m','pytest',str(runtime/'tests/test_checkpoint_bn_guard.py'),'-q'], cwd=runtime,timeout=180,check=True)\n" + marker)
    nb['cells'][-1]['source'] = launch.splitlines(keepends=True)
    nb['metadata']['codex'].update(run_id='independent-joint-extended-v1', max_steps=steps,
        declared_budget_seconds=7200, checkpoint_sha256=CHECKPOINT_SHA, checkpoint_bn_once=True,
        checkpoint_version=1, step100_gate_required=True)
    meta.update(id='indarkarhana/' + TARGET.name, title='Biohub Independent Joint Extended v1',
                code_file=TARGET.name + '.ipynb',
                kernel_sources=['indarkarhana/biohub-independent-joint-broad-v1/1'])
    for cell in nb['cells']:
        ast.parse(''.join(cell['source']))
    return nb, meta


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, choices=(100, 6000), default=6000)
    nb, meta = build(parser.parse_args().steps)
    TARGET.mkdir(parents=True, exist_ok=True)
    (TARGET / meta['code_file']).write_text(json.dumps(nb), encoding='utf-8')
    (TARGET / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(TARGET)
