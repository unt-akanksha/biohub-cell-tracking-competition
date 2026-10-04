"""CPU paired scorer with immutable flow and native selection notebooks."""
import ast
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT/'kaggle/biohub-backward-flow-scoring-v1'


def build():
    nb,meta = runpy.run_path(str(ROOT/'scripts/build-independent-joint-scoring.py'))['build'](1)
    source = ''.join(nb['cells'][-1]['source'])
    assignment = next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id == 'scoring_sources')
    bundle = ast.literal_eval(assignment.value)
    for path in ('scripts/score-backward-flow-selection.py','scripts/score-independent-motion-prior.py',
                 'research/backward_flow_linking.py','research/independent_motion_prior.py','tests/test_backward_flow_linking.py'):
        bundle[path] = (ROOT/path).read_text(encoding='utf-8')
    bundle['flow_launch.ipynb'] = (ROOT/'kaggle/biohub-backward-flow-selection-v1/biohub-backward-flow-selection-v1.ipynb').read_text(encoding='utf-8')
    tail = source[source.index('\nimport runpy'):]
    marker = 'result = runpy.run_path'
    locator = '''flow_candidates = [Path('/kaggle/input')/p/'backward_flow_selection' for p in (
    'biohub-backward-flow-selection-v1','notebooks/indarkarhana/biohub-backward-flow-selection-v1',
    'kernels/indarkarhana/biohub-backward-flow-selection-v1')]
flow_root = next((p for p in flow_candidates if (p/'outputs/flow_manifest.json').is_file()),None)
if flow_root is None:
    raise RuntimeError('Complete flow input not mounted')
subprocess.run([sys.executable,'-m','pytest','tests/test_backward_flow_linking.py','-q'],cwd=scoring,timeout=180,check=True)
'''
    tail = tail.replace(marker,locator+marker)
    tail = tail.replace("scoring/'scripts/score-independent-selection.py'","scoring/'scripts/score-backward-flow-selection.py'")
    tail = tail.replace("root,scoring/'selection_launch.ipynb',truth)","root,scoring/'selection_launch.ipynb',truth,flow_root,scoring/'flow_launch.ipynb')")
    nb['cells'][-1]['source'] = ('scoring_sources = '+repr(bundle)+tail).splitlines(keepends=True)
    bootstrap = ''.join(nb['cells'][0]['source']).replace('independent_joint_score','backward_flow_score')
    bootstrap = bootstrap.replace('independent-joint-scoring-v1','backward-flow-scoring-v1')
    nb['cells'][0]['source'] = bootstrap.splitlines(keepends=True)
    nb['metadata']['codex']['run_id'] = 'backward-flow-scoring-v1'
    meta.update(id='indarkarhana/'+TARGET.name,title='Biohub Backward Flow Scoring v1',code_file=TARGET.name+'.ipynb',
        kernel_sources=['indarkarhana/biohub-independent-joint-selection-v1/1','indarkarhana/biohub-backward-flow-selection-v1/1'])
    for cell in nb['cells']:
        ast.parse(''.join(cell['source']))
    return nb,meta


if __name__ == '__main__':
    nb,meta = build()
    TARGET.mkdir(parents=True,exist_ok=True)
    (TARGET/meta['code_file']).write_text(json.dumps(nb),encoding='utf-8')
    (TARGET/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(TARGET)
