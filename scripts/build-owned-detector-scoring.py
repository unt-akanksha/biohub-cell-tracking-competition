"""CPU scorer bound to the exact inference notebook and completed paired arm."""
import argparse
import ast
import json
from pathlib import Path
import runpy
ROOT=Path(__file__).resolve().parents[1]


def build(arm):
    if arm not in ('sparse','pu'): raise ValueError('Specify paired arm')
    run_id=f'owned-detector-{arm}-scoring-v1'; selection=f'biohub-owned-detector-{arm}-selection-v1'
    target=ROOT/'kaggle'/('biohub-'+run_id)
    nb,meta=runpy.run_path(str(ROOT/'scripts/build-detector-spatial-tta-scoring.py'))['build']()
    source=''.join(nb['cells'][-1]['source'])
    assignment=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='scoring_sources')
    bundle=ast.literal_eval(assignment.value)
    bundle['selection_launch.ipynb']=(ROOT/'kaggle'/selection/(selection+'.ipynb')).read_text()
    tail=source[source.index('\nimport runpy'):].replace('biohub-detector-spatial-tta-selection-v1',selection)
    tail=tail.replace("p/'detector_spatial_tta_selection'",f"p/'owned_detector_{arm}_selection'")
    nb['cells'][-1]['source']=('scoring_sources = '+repr(bundle)+tail).splitlines(keepends=True)
    source=''.join(nb['cells'][0]['source']).replace('detector_spatial_tta_score',f'owned_detector_{arm}_score')
    source=source.replace('detector-spatial-tta-scoring-v1',run_id)
    nb['cells'][0]['source']=source.splitlines(keepends=True); nb['metadata']['codex']['run_id']=run_id
    meta.update(id='indarkarhana/'+target.name,title=f'Biohub Owned Detector {arm.upper()} Scoring v1',code_file=target.name+'.ipynb',
        kernel_sources=[f'indarkarhana/{selection}/1'])
    for cell in nb['cells']: ast.parse(''.join(cell['source']))
    return nb,meta,target


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--arm',choices=('sparse','pu'),required=True)
    nb,meta,target=build(parser.parse_args().arm); target.mkdir(parents=True,exist_ok=True)
    (target/meta['code_file']).write_text(json.dumps(nb),encoding='utf-8')
    (target/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8'); print(target)
