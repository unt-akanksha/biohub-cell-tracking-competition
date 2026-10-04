"""Fixed100step image-supervision probe; does not launch or authorize extension."""
import ast
import json
from pathlib import Path
import runpy
ROOT=Path(__file__).resolve().parents[1]
TARGET=ROOT/'kaggle/biohub-backward-flow-image-probe-v1'


def build():
    gate=json.loads((ROOT/'.biohub/cache/kernel-outputs/backward-flow-image-loss-smoke-v1/backward_flow_image_loss_smoke/test_result.json').read_text())
    if gate['exit_code']!=0 or not gate['cpu_only'] or '6 passed' not in gate['stdout']:
        raise ValueError('Completed CPU numerical image-loss gate required')
    nb,meta=runpy.run_path(str(ROOT/'scripts/build-backward-flow-probe.py'))['build']()
    source=''.join(nb['cells'][1]['source'])
    assignment=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='runtime_sources')
    runtime=ast.literal_eval(assignment.value)
    runtime['backward_flow_image_loss.py']=(ROOT/'research/backward_flow_image_loss.py').read_text()
    runtime['tests/test_backward_flow_image_loss.py']=(ROOT/'tests/test_backward_flow_image_loss.py').read_text().replace(
        "str(Path(__file__).resolve().parents[1]/'research')","str(Path(__file__).resolve().parents[1])")
    source=source.replace(ast.get_source_segment(source,assignment),'runtime_sources = '+repr(runtime))
    nb['cells'][1]['source']=source.splitlines(keepends=True)
    for index in (0,len(nb['cells'])-1):
        source=''.join(nb['cells'][index]['source']).replace('backward-flow-probe-v1','backward-flow-image-probe-v1')
        source=source.replace('backward_flow_probe','backward_flow_image_probe')
        if index==len(nb['cells'])-1:
            marker='process = subprocess.Popen(command, start_new_session=True)'
            if source.count(marker)!=1: raise ValueError('Launch contract changed')
            source=source.replace(marker,"command.extend(['--loss-profile','image'])\n"+marker)
        nb['cells'][index]['source']=source.splitlines(keepends=True)
    nb['metadata']['codex'].update(run_id='backward-flow-image-probe-v1',
        loss_profile=dict(image_weight=.25,smoothness_weight=.01),
        comparison='Fixed original sparse100step probe; exact input replay and diagnostic scope required')
    meta.update(id='indarkarhana/'+TARGET.name,title='Biohub Backward Flow Image Probe v1',code_file=TARGET.name+'.ipynb')
    for cell in nb['cells']: ast.parse(''.join(cell['source']))
    return nb,meta


if __name__=='__main__':
    nb,meta=build(); TARGET.mkdir(parents=True,exist_ok=True)
    (TARGET/meta['code_file']).write_text(json.dumps(nb),encoding='utf-8')
    (TARGET/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(TARGET)
