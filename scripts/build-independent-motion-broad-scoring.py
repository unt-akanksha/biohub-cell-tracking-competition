"""CPU scoring for a versioned complete broad-fit selection output."""
import argparse
import ast
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]


def build(arm,version):
    if arm not in ('control','row') or version < 1:
        raise ValueError('Known arm and completed selection version required')
    selection_name = f'biohub-independent-motion-broad-{arm}-selection-v1'
    if arm == 'control':
        selection_name = 'biohub-motion-broad-control-selection-v1'
    run_id = f'independent-motion-broad-{arm}-scoring-v1'
    nb,meta = runpy.run_path(str(ROOT/'scripts/build-independent-selection-scoring.py'))['build']()
    source = ''.join(nb['cells'][-1]['source'])
    assignment = next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id == 'scoring_sources')
    bundle = ast.literal_eval(assignment.value)
    bundle['selection_launch.ipynb'] = (ROOT/'kaggle'/selection_name/(selection_name+'.ipynb')).read_text()
    tail = source[source.index('\nimport runpy'):]
    tail = tail.replace('biohub-independent-selection-inference-v1',selection_name)
    tail = tail.replace("p/'independent_selection'",f"p/'independent_motion_broad_{arm}_selection'")
    nb['cells'][-1]['source'] = ('scoring_sources = '+repr(bundle)+tail).splitlines(keepends=True)
    bootstrap = ''.join(nb['cells'][0]['source']).replace('independent_selection_score',f'independent_motion_broad_{arm}_score')
    bootstrap = bootstrap.replace('independent-selection-scoring-v1',run_id)
    nb['cells'][0]['source'] = bootstrap.splitlines(keepends=True)
    nb['metadata']['codex'].update(run_id=run_id,paired_training_arm=arm)
    name = 'biohub-'+run_id
    meta.update(id='indarkarhana/'+name,title=f'Biohub Independent Motion Broad {arm.title()} Scoring v1',
        code_file=name+'.ipynb',kernel_sources=[f'indarkarhana/{selection_name}/{version}'])
    for cell in nb['cells']:
        ast.parse(''.join(cell['source']))
    return nb,meta


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm',choices=('control','row'),required=True)
    parser.add_argument('--version',type=int,required=True)
    args = parser.parse_args()
    nb,meta = build(args.arm,args.version)
    target = ROOT/'kaggle'/meta['id'].split('/')[1]
    target.mkdir(parents=True,exist_ok=True)
    (target/meta['code_file']).write_text(json.dumps(nb))
    (target/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    print(target)
