"""Full selection with exact native-node checks for the frozen detector."""
import argparse
import ast
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT/'kaggle/biohub-independent-known-null-selection-v1'


def build(sha, version):
    if len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha) or version < 2:
        raise ValueError('Hash-bound completed fit, not probe version, required')
    base = runpy.run_path(str(ROOT/'scripts/build-edge-feature-tta-selection.py'))
    nb, meta = base['build']()
    for cell in nb['cells']:
        source = ''.join(cell['source']).replace('edge-feature-tta-selection-v1', 'independent-known-null-selection-v1')
        source = source.replace('/kaggle/working/edge_feature_tta_selection', '/kaggle/working/independent_known_null_selection')
        cell['source'] = source.splitlines(keepends=True)
    launch = ''.join(nb['cells'][-1]['source'])
    launch = launch.replace('biohub-independent-joint-broad-v1', 'biohub-independent-known-null-v1')
    launch = launch.replace('independent_joint_broad/outputs/last.pt', 'independent_known_null/outputs/last.pt')
    if launch.count(base['CHECKPOINT_SHA']) != 1:
        raise ValueError('Initialization marker drift')
    launch = launch.replace(base['CHECKPOINT_SHA'], sha).replace("'--edge-feature-tta',", '')
    nb['cells'][-1]['source'] = launch.splitlines(keepends=True)
    nb['metadata']['codex'].update(run_id='independent-known-null-selection-v1', known_null=True,
        edge_feature_tta=False, checkpoint_sha256=sha, checkpoint_version=version)
    meta.update(id='indarkarhana/'+TARGET.name, title='Biohub Independent Known Null Selection v1',
        code_file=TARGET.name+'.ipynb', kernel_sources=[f'indarkarhana/biohub-independent-known-null-v1/{version}',
        'indarkarhana/biohub-independent-joint-selection-v1/1'])
    for cell in nb['cells']:
        ast.parse(''.join(cell['source']))
    return nb, meta


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--version', type=int, required=True)
    args = parser.parse_args()
    nb, meta = build(args.sha256, args.version)
    TARGET.mkdir(parents=True, exist_ok=True)
    (TARGET/meta['code_file']).write_text(json.dumps(nb), encoding='utf-8')
    (TARGET/'kernel-metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(TARGET)
