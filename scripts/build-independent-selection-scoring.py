"""Offline CPU scoring directly from versioned complete GPU graph outputs."""
import ast
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT/'kaggle/biohub-independent-selection-scoring-v1'


def build():
    base = runpy.run_path(str(ROOT/'scripts/build-independent-real-pilot.py'))
    original,meta = base['build']()
    sources = {p:(ROOT/p).read_text() for p in (
        'scripts/score-independent-selection.py','scripts/verify-independent-real-pilot.py',
        'scripts/run-independent-selection-inference.py','research/empty_graph_schema.py',
        'research/focus3d_bridge_rescue.py')}
    sources['selection_launch.ipynb'] = (ROOT/'kaggle/biohub-independent-selection-inference-v1/biohub-independent-selection-inference-v1.ipynb').read_text()
    scorer_path = '.biohub/vendor/kaggle-cell-tracking-competition/src/tracking_cellmot/'
    for name in ('metrics.py','division_metrics.py'):
        sources[scorer_path+name] = (ROOT/scorer_path/name).read_text(encoding='utf-8')
    code = 'scoring_sources = '+repr(sources)+'\n'
    code += '''import runpy
scoring = work/'scoring'
for name,content in scoring_sources.items():
    path = scoring/name
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(content)
candidates = [Path('/kaggle/input')/p/'independent_selection' for p in (
    'biohub-independent-selection-inference-v1',
    'notebooks/indarkarhana/biohub-independent-selection-inference-v1',
    'kernels/indarkarhana/biohub-independent-selection-inference-v1')]
root = next((p for p in candidates if (p/'outputs/selection_manifest.json').is_file()),None)
if root is None:
    raise RuntimeError('Completed selection graph input not mounted')
truth_candidates = [Path('/kaggle/input')/p/'train' for p in (
    'competitions/biohub-cell-tracking-during-development',
    'biohub-cell-tracking-during-development')]
truth = next((p for p in truth_candidates if p.is_dir()),None)
if truth is None:
    raise RuntimeError('Competition truth input not mounted')
result = runpy.run_path(str(scoring/'scripts/score-independent-selection.py'))['score'](
    root,scoring/'selection_launch.ipynb',truth)
(work/'selection_score.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2),flush=True)
watchdog.cancel()
'''
    bootstrap = ''.join(original['cells'][0]['source']).replace(
        '/kaggle/working/independent_real_pilot','/kaggle/working/independent_selection_score')
    bootstrap = bootstrap.replace("run_id='independent-real-pilot-v1'","run_id='independent-selection-scoring-v1'")
    cells = []
    for source in (bootstrap,''.join(original['cells'][2]['source']),code):
        ast.parse(source)
        cells.append(dict(cell_type='code',execution_count=None,metadata={},outputs=[],source=source.splitlines(keepends=True)))
    original['cells'] = cells
    original['metadata']['codex'] = dict(run_id='independent-selection-scoring-v1',
        cpu_only=True,target_audit_opened=False,authorized_for_submission=False)
    meta.update(id='indarkarhana/'+TARGET.name,title='Biohub Independent Selection Scoring v1',
        code_file=TARGET.name+'.ipynb',enable_gpu=False,enable_tpu=False,
        kernel_sources=['indarkarhana/biohub-independent-selection-inference-v1/1'])
    meta.pop('machine_shape',None)
    return original,meta


if __name__ == '__main__':
    nb,meta = build()
    TARGET.mkdir(parents=True,exist_ok=True)
    (TARGET/meta['code_file']).write_text(json.dumps(nb))
    (TARGET/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    print(TARGET)
