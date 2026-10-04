"""Hash-bound full movie evaluation of both embedded, frozen image components."""
import ast
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT/'kaggle/biohub-image-motion-linker-selection-v1'


def build(report=None):
    if report is None:
        report = json.loads((ROOT/'reports/experiments/image-motion-linker-v2-training.json').read_text())
    if (report['steps'] != 1000 or report['status'] != 'verified_image_motion_linker_training_not_selection'
        or report['flow_unchanged'] is not True or report['detector_unchanged'] is not True
        or report['probe_inputs_replayed'] is not True):
        raise ValueError('Completed verified full image-motion linker fit required')
    nb,meta = runpy.run_path(str(ROOT/'scripts/build-independent-known-null-selection.py'))['build'](report['checkpoint_sha256'],2)
    source = ''.join(nb['cells'][1]['source'])
    assignment = next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id == 'runtime_sources')
    runtime = ast.literal_eval(assignment.value)
    runtime['run_selection.py'] = runtime['run_selection.py'].replace(
        "run_id='independent-known-null-selection-v1'", "run_id='image-motion-linker-selection-v1'")
    for name in ('image_motion_residual','backward_flow_ops','backward_flow_model'):
        runtime[name+'.py'] = (ROOT/f'research/{name}.py').read_text(encoding='utf-8')
    source = source.replace(ast.get_source_segment(source,assignment),'runtime_sources = '+repr(runtime))
    nb['cells'][1]['source'] = source.splitlines(keepends=True)
    for index in (0,len(nb['cells'])-1):
        source = ''.join(nb['cells'][index]['source']).replace('independent-known-null-selection-v1','image-motion-linker-selection-v1')
        source = source.replace('independent_known_null_selection','image_motion_linker_selection')
        if index == len(nb['cells'])-1:
            source = source.replace('biohub-independent-known-null-v1','biohub-image-motion-linker-v1')
            source = source.replace('independent_known_null/outputs/last.pt','image_motion_linker/outputs/last.pt')
            source = source.replace('biohub-independent-joint-selection-v1','biohub-independent-known-null-selection-v1')
            source = source.replace("p/'independent_joint_selection'","p/'independent_known_null_selection'")
        nb['cells'][index]['source'] = source.splitlines(keepends=True)
    nb['metadata']['codex'].update(run_id='image-motion-linker-selection-v1',image_motion=True,
        frozen_flow_sha256=report['frozen_flow_sha256'],node_reference_kernel='indarkarhana/biohub-independent-known-null-selection-v1/1')
    meta.update(id='indarkarhana/'+TARGET.name,title='Biohub Image Motion Linker Selection v1',code_file=TARGET.name+'.ipynb',
        kernel_sources=['indarkarhana/biohub-image-motion-linker-v1/2','indarkarhana/biohub-independent-known-null-selection-v1/1'])
    for cell in nb['cells']:
        ast.parse(''.join(cell['source']))
    return nb,meta


if __name__ == '__main__':
    nb,meta = build()
    TARGET.mkdir(parents=True,exist_ok=True)
    (TARGET/meta['code_file']).write_text(json.dumps(nb),encoding='utf-8')
    (TARGET/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(TARGET)
