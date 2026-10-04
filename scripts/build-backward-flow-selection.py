"""Frozen cached-node motion inference; runs only after verified full training."""
import ast
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT/'kaggle/biohub-backward-flow-selection-v1'


def build(report=None):
    if report is None:
        report = json.loads((ROOT/'reports/experiments/backward-flow-fit-v1-result.json').read_text())
    sha = report['checkpoint_sha256']
    if (report['status'] != 'verified_full_motion_fit_not_tracking_validation'
        or report['small_fit_gate_passed'] is not True or report['probe_inputs_replayed'] is not True
        or report['steps'] != 1000 or len(sha)!=64 or set(sha)-set('0123456789abcdef')):
        raise ValueError('Verified completed motion-learning fit required')
    nb,meta = runpy.run_path(str(ROOT/'scripts/build-independent-real-pilot.py'))['build']()
    source = ''.join(nb['cells'][1]['source'])
    assignment = next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id == 'runtime_sources')
    runtime = ast.literal_eval(assignment.value)
    for name,path in [('run_pilot.py','scripts/run-backward-flow-selection.py'),
        ('run_selection.py','scripts/run-independent-selection-inference.py'),
        ('backward_flow_ops.py','research/backward_flow_ops.py'),('backward_flow_model.py','research/backward_flow_model.py'),
        ('feature_tta_reference.py','research/feature_tta_reference.py')]:
        runtime[name] = (ROOT/path).read_text(encoding='utf-8')
    source = source.replace(ast.get_source_segment(source,assignment),'runtime_sources = '+repr(runtime))
    nb['cells'][1]['source'] = source.splitlines(keepends=True)
    for index in (0,len(nb['cells'])-1):
        source = ''.join(nb['cells'][index]['source']).replace('independent-real-pilot-v1','backward-flow-selection-v1')
        source = source.replace('independent_real_pilot','backward_flow_selection')
        if index == len(nb['cells'])-1:
            marker = 'command = ['
            locator = '''def mounted(slug,folder):
    candidates = [Path('/kaggle/input')/p/folder for p in (slug,'notebooks/indarkarhana/'+slug,'kernels/indarkarhana/'+slug)]
    root = next((p for p in candidates if p.is_dir()),None)
    if root is None:
        raise RuntimeError('Required completed input is missing: '+slug)
    return root
fit = mounted('biohub-backward-flow-fit-v1','backward_flow_fit')
reference = mounted('biohub-independent-joint-selection-v1','independent_joint_selection')
'''
            source = source.replace(marker,locator+marker)
            source = source.replace("'--steps', '100', ",'')
            source = source.replace("'--data', str(data)",f"'--checkpoint',str(fit/'outputs/last.pt'),'--sha256',{sha!r},'--reference',str(reference),'--data',str(data)")
        nb['cells'][index]['source'] = source.splitlines(keepends=True)
    nb['metadata']['codex'] = dict(run_id='backward-flow-selection-v1',checkpoint_sha256=sha,
        declared_budget_seconds=3600,target_audit_opened=False,authorized_for_submission=False)
    meta.update(id='indarkarhana/'+TARGET.name,title='Biohub Backward Flow Selection v1',code_file=TARGET.name+'.ipynb',
        kernel_sources=['indarkarhana/biohub-backward-flow-fit-v1/1','indarkarhana/biohub-independent-joint-selection-v1/1'])
    for cell in nb['cells']:
        ast.parse(''.join(cell['source']))
    return nb,meta


if __name__ == '__main__':
    nb,meta = build()
    TARGET.mkdir(parents=True,exist_ok=True)
    (TARGET/meta['code_file']).write_text(json.dumps(nb),encoding='utf-8')
    (TARGET/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(TARGET)
