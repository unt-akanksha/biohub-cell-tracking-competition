"""First four complete embryo-held-out movies for the frozen D4 candidate."""
import ast
import json
from pathlib import Path
import runpy
ROOT=Path(__file__).resolve().parents[1]
TARGET=ROOT/'kaggle/biohub-detector-spatial-tta-audit-v1'


def build():
    gate=runpy.run_path(str(ROOT/'research/embryo_audit_contract.py'))
    report=(ROOT/'reports/experiments/detector-spatial-tta-selection-v1-score.json').read_bytes()
    contract=gate['verify_gate'](report,(ROOT/'research/independent_real_baseline_v1_split.json').read_bytes())
    nb,meta=runpy.run_path(str(ROOT/'scripts/build-detector-spatial-tta-selection.py'))['build']()
    source=''.join(nb['cells'][1]['source'])
    assignment=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='runtime_sources')
    runtime=ast.literal_eval(assignment.value)
    runtime['embryo_audit_contract.py']=(ROOT/'research/embryo_audit_contract.py').read_text()
    runtime['frozen_selection_report.json']=report.decode('utf-8')
    source=source.replace(ast.get_source_segment(source,assignment),'runtime_sources = '+repr(runtime))
    nb['cells'][1]['source']=source.splitlines(keepends=True)
    for index in (0,len(nb['cells'])-1):
        source=''.join(nb['cells'][index]['source']).replace('detector-spatial-tta-selection-v1','detector-spatial-tta-audit-v1')
        source=source.replace('detector_spatial_tta_selection','detector_spatial_tta_audit')
        if index==len(nb['cells'])-1:
            source=source.replace("command.extend(['--standalone-image-flow','--detector-spatial-tta'])",
                "command.extend(['--standalone-image-flow','--detector-spatial-tta','--embryo-audit'])")
        nb['cells'][index]['source']=source.splitlines(keepends=True)
    nb['metadata']['codex'].update(run_id='detector-spatial-tta-audit-v1',embryo_audit=contract,target_audit_opened=True)
    meta.update(id='indarkarhana/biohub-detector-spatial-tta-embryo-audit-v1',title='Biohub Detector Spatial TTA Embryo Audit v1',code_file=TARGET.name+'.ipynb')
    for cell in nb['cells']: ast.parse(''.join(cell['source']))
    return nb,meta


if __name__=='__main__':
    nb,meta=build(); TARGET.mkdir(parents=True,exist_ok=True)
    (TARGET/meta['code_file']).write_text(json.dumps(nb),encoding='utf-8')
    (TARGET/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(TARGET)
