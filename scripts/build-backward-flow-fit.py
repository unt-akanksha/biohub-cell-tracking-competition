"""1000-step extension only after the verified real probe; separate artifact."""
import ast
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT/'kaggle/biohub-backward-flow-fit-v1'


def build():
    report = json.loads((ROOT/'reports/experiments/backward-flow-probe-v1-result.json').read_text())
    if report['small_fit_gate_passed'] is not True or report['checkpoint_sha256'] != 'c1203ffb03ac98b4aa5b332ec26b4469ba5b9e784e6e095f03a41c533bba5884':
        raise ValueError('Verified successful motion probe required')
    nb,meta = runpy.run_path(str(ROOT/'scripts/build-backward-flow-probe.py'))['build']()
    source = ''.join(nb['cells'][1]['source'])
    assignment = next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id == 'runtime_sources')
    runtime = ast.literal_eval(assignment.value)
    runtime['probe_result.json'] = (ROOT/'.biohub/cache/kernel-outputs/backward-flow-probe-v1/backward_flow_probe/outputs/result.json').read_text()
    source = source.replace(ast.get_source_segment(source,assignment),'runtime_sources = '+repr(runtime))
    nb['cells'][1]['source'] = source.splitlines(keepends=True)
    for index in (0,len(nb['cells'])-1):
        source = ''.join(nb['cells'][index]['source']).replace('backward-flow-probe-v1','backward-flow-fit-v1')
        source = source.replace('backward_flow_probe','backward_flow_fit')
        if index == len(nb['cells'])-1:
            source = source.replace("'--data', str(data)","'--steps','1000','--data', str(data)")
        nb['cells'][index]['source'] = source.splitlines(keepends=True)
    nb['metadata']['codex'].update(run_id='backward-flow-fit-v1',max_steps=1000,
        probe_checkpoint_sha256=report['checkpoint_sha256'],step100_gate_required=True)
    meta.update(id='indarkarhana/'+TARGET.name,title='Biohub Backward Flow Fit v1',code_file=TARGET.name+'.ipynb')
    for cell in nb['cells']:
        ast.parse(''.join(cell['source']))
    return nb,meta


if __name__ == '__main__':
    nb,meta = build()
    TARGET.mkdir(parents=True,exist_ok=True)
    (TARGET/meta['code_file']).write_text(json.dumps(nb),encoding='utf-8')
    (TARGET/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(TARGET)
