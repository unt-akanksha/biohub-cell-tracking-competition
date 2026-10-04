"""Frozen complete-movie evaluation of the completed 6000-update real fit."""
import argparse
import ast
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'kaggle/biohub-independent-extended-selection-v1'


def build(sha, version):
    nb, meta = runpy.run_path(str(ROOT / 'scripts/build-independent-joint-selection.py'))['build'](sha, version)
    for cell in nb['cells']:
        source = ''.join(cell['source'])
        source = source.replace('independent-joint-selection-v1', 'independent-extended-selection-v1')
        source = source.replace('biohub-independent-joint-broad-v1', 'biohub-independent-joint-extended-v1')
        source = source.replace('independent_joint_broad/outputs/last.pt', 'independent_joint_extended/outputs/last.pt')
        source = source.replace('/kaggle/working/independent_joint_selection', '/kaggle/working/independent_extended_selection')
        ast.parse(source)
        cell['source'] = source.splitlines(keepends=True)
    nb['metadata']['codex'].update(run_id='independent-extended-selection-v1')
    meta.update(id='indarkarhana/' + TARGET.name, title='Biohub Independent Extended Selection v1',
                code_file=TARGET.name + '.ipynb',
                kernel_sources=[f'indarkarhana/biohub-independent-joint-extended-v1/{version}'])
    return nb, meta


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--version', type=int, required=True)
    args = parser.parse_args()
    nb, meta = build(args.sha256, args.version)
    TARGET.mkdir(parents=True, exist_ok=True)
    (TARGET / meta['code_file']).write_text(json.dumps(nb), encoding='utf-8')
    (TARGET / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(TARGET)
