"""Stage one exact completed broad-pair arm for complete-movie evaluation."""
import argparse
import ast
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]


def build(sha,version,arm):
    if arm not in ('control','row'):
        raise ValueError('Unknown paired arm')
    nb,meta = runpy.run_path(str(ROOT/'scripts/build-independent-motion-residual-selection.py'))['build'](sha,version)
    run_id = f'independent-motion-broad-{arm}-selection-v1'
    for cell in nb['cells']:
        source = ''.join(cell['source'])
        source = source.replace('independent-motion-residual-selection-v1',run_id)
        source = source.replace('biohub-independent-motion-residual-v1','biohub-independent-motion-broad-pair-v1')
        source = source.replace('independent_motion_residual/outputs/last.pt',
                                f'independent_motion_broad_pair/outputs/{arm}/last.pt')
        source = source.replace('/kaggle/working/independent_motion_residual_selection',
                                f'/kaggle/working/independent_motion_broad_{arm}_selection')
        cell['source'] = source.splitlines(keepends=True)
        ast.parse(source)
    nb['metadata']['codex'].update(run_id=run_id,paired_training_arm=arm)
    name = 'biohub-'+run_id
    title = f'Biohub Independent Motion Broad {arm.title()} Selection v1'
    if arm == 'control':
        name = 'biohub-motion-broad-control-selection-v1'
        title = 'Biohub Motion Broad Control Selection v1'
    meta.update(id='indarkarhana/'+name,title=title,
        code_file=name+'.ipynb',kernel_sources=[f'indarkarhana/biohub-independent-motion-broad-pair-v1/{version}'])
    return nb,meta


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sha256',required=True)
    parser.add_argument('--version',type=int,required=True)
    parser.add_argument('--arm',choices=('control','row'),required=True)
    args = parser.parse_args()
    nb,meta = build(args.sha256,args.version,args.arm)
    target = ROOT/'kaggle'/meta['id'].split('/')[1]
    target.mkdir(parents=True,exist_ok=True)
    (target/meta['code_file']).write_text(json.dumps(nb))
    (target/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    print(target)
