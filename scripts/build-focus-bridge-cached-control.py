"""Prepare evaluation recovery from a hash-bound completed control CSV.

Only use after inspecting terminal output of the parent job. Building this
notebook does not launch it and does not relax the frozen scientific policy.
"""
import argparse
import ast
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
PAIRED = runpy.run_path(str(ROOT/'scripts/build-focus-bridge-official-paired-v2.py'))


def build(control_sha256):
    if len(control_sha256)!=64 or any(c not in '0123456789abcdef' for c in control_sha256):
        raise ValueError('A verified lowercase SHA-256 is required')
    notebook = PAIRED['build']()
    cells = notebook['cells']
    inference = [i for i,c in enumerate(cells) if 'Prediction completed in' in ''.join(c['source'])]
    if len(inference)!=1:
        raise ValueError('Base inference cell changed')
    cells.pop(inference[0])
    post = [i for i,c in enumerate(cells) if 'def filter_output_graph(' in ''.join(c['source'])]
    if len(post)!=1:
        raise ValueError('Base postprocessing cell changed')
    text = ''.join(cells[post[0]]['source'])
    anchor = 'DEEPCENTER_VETO_DETECTOR = load_deepcenter_veto_detector()'
    if text.count(anchor)!=1:
        raise ValueError('DeepCenter initialization changed')
    # Keep public function definitions for DeepCenter but never rerun the
    # public graph postprocessing or modify the cached control.
    text = text[:text.index(anchor)] + '''
# The setup already located and hash-checked the exact checkpoint. Do not
# recursively walk the competition Zarr chunks to rediscover it.
def _dc_checkpoint_candidates():
    return [Path(_deepcenter_materialized_path)]
''' + anchor + '\n'
    text += f'''
import hashlib as _resume_hash
_resume_candidates = _preflight_controls
if len(_resume_candidates) != 1:
    raise RuntimeError('Expected one hash-matched saved control CSV')
SUBMISSION_PATH = _resume_candidates[0]
test_stems = {PAIRED['STEMS']!r}
_resume_control = pd.read_csv(SUBMISSION_PATH)
if set(_resume_control.dataset) != set(test_stems):
    raise RuntimeError('Cached control movie coverage changed')
print('Resuming the fixed bridge evaluation from verified control:', SUBMISSION_PATH)
'''
    cells[post[0]]['source'] = text.splitlines(keepends=True)
    preflight = f'''
from pathlib import Path
import hashlib
_control_paths = [Path('/kaggle/input') / prefix / 'control_validation.csv' for prefix in (
    'biohub-focus-bridge-recovered-control-v1',
    'datasets/indarkarhana/biohub-focus-bridge-recovered-control-v1')]
_preflight_controls = [p for p in _control_paths if p.is_file()
    and hashlib.sha256(p.read_bytes()).hexdigest() == {control_sha256!r}]
if len(_preflight_controls) != 1:
    raise RuntimeError('Missing hash-bound control input; stopping before dependency/model setup')
'''
    cells.insert(2, PAIRED['BASE']['code_cell'](preflight))
    for cell in cells:
        if cell['cell_type']=='code':
            ast.parse(''.join(cell['source']))
    notebook['metadata']['codex']['cached_control_sha256'] = control_sha256
    notebook['metadata']['codex']['recovery_revision'] = {
        'name': 'smoothed-proposal-boundary-compatibility-v1',
        'reason': 'Recovered GEFF metadata contains negative spatial coordinates',
        'proposal_coordinates_modified': False,
        'out_of_image_proposals_eligible_for_addition': False,
        'thresholds_changed': False,
        'parent_run_modified': False,
    }
    return notebook


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--control-sha256',required=True)
    parser.add_argument('--parent-version',type=int,required=True)
    args=parser.parse_args()
    if args.parent_version<1:
        raise ValueError('Parent version must be positive')
    notebook=build(args.control_sha256)
    target=ROOT/'kaggle/biohub-focus-bridge-cached-control-v1'
    target.mkdir(parents=True,exist_ok=True)
    name=target.name+'.ipynb'
    (target/name).write_text(json.dumps(notebook))
    metadata=json.loads((PAIRED['TARGET']/'kernel-metadata.json').read_text())
    metadata.update({'id':'indarkarhana/'+target.name,'title':'Biohub FOCUS Bridge Cached Control v1','code_file':name})
    # Kaggle rejects outputs of failed runs as kernel inputs. The verified
    # control is recovered into a private dataset instead.
    metadata['dataset_sources'].append('indarkarhana/biohub-focus-bridge-recovered-control-v1')
    (target/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
    print(target)
