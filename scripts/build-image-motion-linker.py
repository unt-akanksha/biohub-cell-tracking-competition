"""Bounded linker fit using frozen b642 detection and independently trained flow."""
import argparse
import ast
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT/'kaggle/biohub-image-motion-linker-v1'
INITIAL = 'b64254aece75ef709e517621757a5313ac665e0803050ce4f13fa2a76602d6ef'


def build(steps=100):
    if steps not in (100,1000):
        raise ValueError('Only small gate or bounded fit supported')
    report = json.loads((ROOT/'reports/experiments/backward-flow-selection-v1-score.json').read_text())
    if report['deltas']['static'] <= 0 or report['result']['flow_checkpoint_sha256'] != '3006ee0f904640b16dd988d6404a1b3ac4f933ca68ca912d69fe4598f96d4788':
        raise ValueError('Verified full-movie image motion gain required')
    base = runpy.run_path(str(ROOT/'scripts/build-independent-known-null.py'))
    nb,meta = base['build'](steps)
    source = ''.join(nb['cells'][1]['source'])
    assignment = next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id == 'runtime_sources')
    runtime = ast.literal_eval(assignment.value)
    if steps == 1000:
        probe_report = json.loads((ROOT/'reports/experiments/image-motion-linker-v1-training.json').read_text())
        if (probe_report['status'] != 'verified_image_motion_linker_training_not_selection'
            or probe_report['flow_unchanged'] is not True or probe_report['detector_unchanged'] is not True
            or probe_report['known_null_columns'] <= 0):
            raise ValueError('Successful real image-motion probe required')
        runtime['image_motion_probe.json'] = (ROOT/'.biohub/cache/kernel-outputs/image-motion-linker-probe-v1/image_motion_linker/outputs/result.json').read_text()
    runtime['run_association.py'] = runtime['run_association.py'].replace(
        "run_id='independent-known-null-v1'","run_id='image-motion-linker-v1'")
    for name in ('image_motion_residual','calibrated_motion_scores','backward_flow_ops','backward_flow_model'):
        runtime[name+'.py'] = (ROOT/f'research/{name}.py').read_text(encoding='utf-8')
    runtime['tests/test_image_motion_residual.py'] = (ROOT/'tests/test_image_motion_residual.py').read_text(encoding='utf-8').replace(
        "str(Path(__file__).resolve().parents[1]/'research')","str(Path(__file__).resolve().parents[1])")
    source = source.replace(ast.get_source_segment(source,assignment),'runtime_sources = '+repr(runtime))
    nb['cells'][1]['source'] = source.splitlines(keepends=True)
    for index in (0,len(nb['cells'])-1):
        source = ''.join(nb['cells'][index]['source']).replace('independent-known-null-v1','image-motion-linker-v1')
        source = source.replace('independent_known_null','image_motion_linker')
        nb['cells'][index]['source'] = source.splitlines(keepends=True)
    launch = ''.join(nb['cells'][-1]['source'])
    launch = launch.replace('biohub-independent-joint-broad-v1','biohub-independent-known-null-v1')
    launch = launch.replace('independent_joint_broad/outputs/last.pt','independent_known_null/outputs/last.pt')
    if launch.count(base['CHECKPOINT_SHA']) != 1:
        raise ValueError('Initialization hash marker changed')
    launch = launch.replace(base['CHECKPOINT_SHA'],INITIAL)
    locator = '''flow_candidates = [Path('/kaggle/input')/p/'backward_flow_fit/outputs/last.pt' for p in (
    'biohub-backward-flow-fit-v1','notebooks/indarkarhana/biohub-backward-flow-fit-v1',
    'kernels/indarkarhana/biohub-backward-flow-fit-v1')]
flow_checkpoint = next((p for p in flow_candidates if p.is_file()),None)
if flow_checkpoint is None:
    raise RuntimeError('Completed image-flow checkpoint not mounted')
'''
    launch = launch.replace('command = [',locator+'command = [')
    launch = launch.replace("'--known-null'","'--image-motion','--flow-checkpoint',str(flow_checkpoint)")
    marker = 'process = subprocess.Popen(command, start_new_session=True)'
    launch = launch.replace(marker,"subprocess.run([sys.executable,'-m','pytest',str(runtime/'tests/test_image_motion_residual.py'),'-q'],cwd=runtime,timeout=180,check=True)\n"+marker)
    nb['cells'][-1]['source'] = launch.splitlines(keepends=True)
    nb['metadata']['codex'].update(run_id='image-motion-linker-v1',image_motion=True,
        checkpoint_sha256=INITIAL,checkpoint_version=2,flow_checkpoint_sha256=report['result']['flow_checkpoint_sha256'])
    meta.update(id='indarkarhana/'+TARGET.name,title='Biohub Image Motion Linker v1',code_file=TARGET.name+'.ipynb',
        kernel_sources=['indarkarhana/biohub-independent-known-null-v1/2','indarkarhana/biohub-backward-flow-fit-v1/1'])
    for cell in nb['cells']:
        ast.parse(''.join(cell['source']))
    return nb,meta


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps',type=int,choices=(100,1000),default=100)
    nb,meta = build(parser.parse_args().steps)
    TARGET.mkdir(parents=True,exist_ok=True)
    (TARGET/meta['code_file']).write_text(json.dumps(nb),encoding='utf-8')
    (TARGET/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(TARGET)
