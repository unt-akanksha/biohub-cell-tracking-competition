"""Offline real-data feature-averaging probe; no promotion from three frames."""
import ast
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT/'kaggle/biohub-edge-feature-tta-probe-v1'
CHECKPOINT_SHA = 'c5023345d31d91929a8d05219310a9edf1aeecf576d7593cbc5e65208c76b470'


def build():
    nb,meta = runpy.run_path(str(ROOT/'scripts/build-independent-joint-selection.py'))['build'](CHECKPOINT_SHA,1)
    source = ''.join(nb['cells'][1]['source'])
    assignment = next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id == 'runtime_sources')
    runtime = ast.literal_eval(assignment.value)
    for name,path in [('run_feature_probe.py','scripts/run-edge-feature-tta-probe.py'),
                      ('edge_feature_tta.py','research/edge_feature_tta.py'),
                      ('tests/test_edge_feature_tta.py','tests/test_edge_feature_tta.py')]:
        runtime[name] = (ROOT/path).read_text(encoding='utf-8')
    runtime['tests/test_edge_feature_tta.py'] = runtime['tests/test_edge_feature_tta.py'].replace(
        "sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'research'))",
        "sys.path.insert(0, str(Path(__file__).resolve().parents[1]))")
    source = source.replace(ast.get_source_segment(source,assignment),'runtime_sources = '+repr(runtime))
    nb['cells'][1]['source'] = source.splitlines(keepends=True)
    for index in (0,len(nb['cells'])-1):
        source = ''.join(nb['cells'][index]['source']).replace('independent-joint-selection-v1','edge-feature-tta-probe-v1')
        source = source.replace('/kaggle/working/independent_joint_selection','/kaggle/working/edge_feature_tta_probe')
        source = source.replace("runtime/'run_selection.py'", "runtime/'run_feature_probe.py'")
        if index == len(nb['cells'])-1:
            marker = 'process = subprocess.Popen(command, start_new_session=True)'
            assert source.count(marker) == 1
            source = source.replace(marker,"subprocess.run([sys.executable,'-m','pytest',str(runtime/'tests/test_edge_feature_tta.py'),'-q'],cwd=runtime,timeout=180,check=True)\n"+marker)
        nb['cells'][index]['source'] = source.splitlines(keepends=True)
    nb['metadata']['codex'].update(run_id='edge-feature-tta-probe-v1',evaluation_scope='Three training frames only',feature_views=8)
    meta.update(id='indarkarhana/'+TARGET.name,title='Biohub Edge Feature TTA Probe v1',code_file=TARGET.name+'.ipynb')
    for cell in nb['cells']:
        ast.parse(''.join(cell['source']))
    return nb,meta


if __name__ == '__main__':
    nb,meta = build()
    TARGET.mkdir(parents=True,exist_ok=True)
    (TARGET/meta['code_file']).write_text(json.dumps(nb),encoding='utf-8')
    (TARGET/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(TARGET)
