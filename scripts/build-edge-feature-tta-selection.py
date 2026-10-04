"""Full-movie feature-only ablation, requiring exact native node equality."""
import ast
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT/'kaggle/biohub-edge-feature-tta-selection-v1'
CHECKPOINT_SHA = 'c5023345d31d91929a8d05219310a9edf1aeecf576d7593cbc5e65208c76b470'


def build():
    nb,meta = runpy.run_path(str(ROOT/'scripts/build-independent-joint-selection.py'))['build'](CHECKPOINT_SHA,1)
    source = ''.join(nb['cells'][1]['source'])
    assignment = next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id == 'runtime_sources')
    runtime = ast.literal_eval(assignment.value)
    for name in ('edge_feature_tta.py','feature_tta_reference.py'):
        runtime[name] = (ROOT/'research'/name).read_text(encoding='utf-8')
    source = source.replace(ast.get_source_segment(source,assignment),'runtime_sources = '+repr(runtime))
    nb['cells'][1]['source'] = source.splitlines(keepends=True)
    for cell in nb['cells']:
        source = ''.join(cell['source']).replace('independent-joint-selection-v1','edge-feature-tta-selection-v1')
        source = source.replace('/kaggle/working/independent_joint_selection','/kaggle/working/edge_feature_tta_selection')
        cell['source'] = source.splitlines(keepends=True)
    launch = ''.join(nb['cells'][-1]['source'])
    marker = 'process = subprocess.Popen(command, start_new_session=True)'
    reference = '''reference_candidates = [Path('/kaggle/input')/p/'independent_joint_selection' for p in (
    'biohub-independent-joint-selection-v1',
    'notebooks/indarkarhana/biohub-independent-joint-selection-v1',
    'kernels/indarkarhana/biohub-independent-joint-selection-v1')]
reference = next((p for p in reference_candidates if (p/'outputs/selection_manifest.json').is_file()),None)
if reference is None:
    raise RuntimeError('Exact native selection input missing')
command.extend(['--edge-feature-tta','--node-reference',str(reference)])
'''
    assert launch.count(marker) == 1
    nb['cells'][-1]['source'] = launch.replace(marker,reference+marker).splitlines(keepends=True)
    nb['metadata']['codex'].update(run_id='edge-feature-tta-selection-v1',edge_feature_tta=True,
        node_reference_kernel='indarkarhana/biohub-independent-joint-selection-v1/1')
    meta.update(id='indarkarhana/'+TARGET.name,title='Biohub Edge Feature TTA Selection v1',
        code_file=TARGET.name+'.ipynb',kernel_sources=['indarkarhana/biohub-independent-joint-broad-v1/1',
        'indarkarhana/biohub-independent-joint-selection-v1/1'])
    for cell in nb['cells']:
        ast.parse(''.join(cell['source']))
    return nb,meta


if __name__ == '__main__':
    nb,meta = build()
    TARGET.mkdir(parents=True,exist_ok=True)
    (TARGET/meta['code_file']).write_text(json.dumps(nb),encoding='utf-8')
    (TARGET/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(TARGET)
