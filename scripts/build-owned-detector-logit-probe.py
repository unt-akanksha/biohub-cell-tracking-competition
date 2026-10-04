"""Real raw-logit teacher gate; reuses validated loss and original input replay."""
import ast
import json
from pathlib import Path
import runpy
ROOT=Path(__file__).resolve().parents[1]
TARGET=ROOT/'kaggle/biohub-owned-detector-logit-probe-v1'


def build():
    nb,meta=runpy.run_path(str(ROOT/'scripts/build-owned-detector-pu-fp32-probe.py'))['build']()
    source=''.join(nb['cells'][1]['source'])
    assignment=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='runtime_sources')
    runtime=ast.literal_eval(assignment.value)
    runtime['run_pilot.py']=runtime['run_pilot.py'].replace('owned-detector-pu-fp32-probe-v1','owned-detector-logit-probe-v1')
    runtime['owned_detector_logit_targets.py']=(ROOT/'research/owned_detector_logit_targets.py').read_text()
    runtime['tests/test_owned_detector_logit_targets.py']=(ROOT/'tests/test_owned_detector_logit_targets.py').read_text().replace(
        "str(Path(__file__).resolve().parents[1]/'research')","str(Path(__file__).resolve().parents[1])")
    source=source.replace(ast.get_source_segment(source,assignment),'runtime_sources = '+repr(runtime))
    nb['cells'][1]['source']=source.splitlines(keepends=True)
    for index in (0,len(nb['cells'])-1):
        source=''.join(nb['cells'][index]['source']).replace('owned-detector-pu-fp32-probe-v1','owned-detector-logit-probe-v1')
        source=source.replace('owned_detector_pu_fp32_probe','owned_detector_logit_probe')
        if index==len(nb['cells'])-1:
            marker='process = subprocess.Popen(command, start_new_session=True)'
            source=source.replace(marker,"command.append('--logit-targets')\n"+marker)
        nb['cells'][index]['source']=source.splitlines(keepends=True)
    nb['metadata']['codex'].update(run_id='owned-detector-logit-probe-v1',teacher_peak_order='raw_logits')
    meta.update(id='indarkarhana/'+TARGET.name,title='Biohub Owned Detector Logit Probe v1',code_file=TARGET.name+'.ipynb')
    for cell in nb['cells']: ast.parse(''.join(cell['source']))
    return nb,meta


if __name__=='__main__':
    nb,meta=build(); TARGET.mkdir(parents=True,exist_ok=True)
    (TARGET/meta['code_file']).write_text(json.dumps(nb),encoding='utf-8')
    (TARGET/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(TARGET)
