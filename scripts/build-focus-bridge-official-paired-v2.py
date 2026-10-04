"""Generate a four-movie paired evaluation using the packaged official scorer."""
import ast
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
BASE = runpy.run_path(str(ROOT / 'scripts/build-948tta2-lsm-consensus-candidate.py'))
TARGET = ROOT / 'kaggle/biohub-focus-bridge-official-paired-v2'
STEMS = ['44b6_81c256f0','44b6_24264f12','6bba_f1fde7e0','6bba_23af9eeb']


def build():
    assert BASE['sha256_file'](BASE['SOURCE_NOTEBOOK']) == BASE['SOURCE_NOTEBOOK_SHA256']
    source = json.loads(BASE['SOURCE_NOTEBOOK'].read_text())
    # Base production path is redirected to a four-image-only staging directory.
    cells = [BASE['markdown_cell']('Attributed 948TTA2 control plus project FOCUS bridge. Official paired evaluation only.')]
    watchdog = BASE['WATCHDOG'].replace('948tta2-lsm-consensus-v1', 'focus-bridge-official-paired-v2')
    watchdog = watchdog.replace('43200','10800').replace('42000','10200').replace('42,000','10,200')
    cells.append(BASE['code_cell'](watchdog))
    for index in range(4, 10):
        text = ''.join(source['cells'][index]['source'])
        if index == 6:
            text = text.replace('TEST_DIR = COMP_DIR / "test"',
                'TEST_DIR = Path("/kaggle/working/bridge_images")\n'
                'TEST_DIR.mkdir(exist_ok=True)\n'
                f'for _stem in {STEMS!r}:\n'
                '    _image = COMP_DIR / "train" / f"{_stem}.zarr"\n'
                '    if not _image.exists(): raise FileNotFoundError(_image)\n'
                '    (TEST_DIR / _image.name).symlink_to(_image, target_is_directory=True)')
            text = text.replace('SUBMISSION_PATH = WORKING_DIR / "submission.csv"',
                                'SUBMISSION_PATH = WORKING_DIR / "control_validation.csv"')
        cells.append(BASE['code_cell'](text))
    cells.append(BASE['code_cell']((ROOT/'research/focus3d_bridge_rescue.py').read_text()))
    cells.append(BASE['code_cell']((ROOT/'research/evaluate_focus_bridge.py').read_text()))
    for cell in cells:
        if cell['cell_type'] == 'code': ast.parse(''.join(cell['source']))
    return {'cells': cells, 'metadata': {'kernelspec': {'display_name':'Python 3','language':'python','name':'python3'},
            'codex': {'run_id':'focus-bridge-official-paired-v2','frozen_stems':STEMS,
                      'authorized_for_submission':False}}, 'nbformat':4,'nbformat_minor':4}


if __name__ == '__main__':
    notebook = build()
    TARGET.mkdir(parents=True, exist_ok=True)
    name = TARGET.name + '.ipynb'
    (TARGET/name).write_text(json.dumps(notebook))
    metadata = json.loads(BASE['SOURCE_METADATA'].read_text())
    metadata.pop('id_no',None)
    metadata.update({'id':'indarkarhana/'+TARGET.name,'title':'Biohub FOCUS Bridge Official Paired v2',
                     'code_file':name,'is_private':True,
                     'kernel_sources':['indarkarhana/biohub-focus3d-bridge-proposals-v1/1']})
    (TARGET/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
    print(TARGET)
