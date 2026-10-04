"""Full eight-source fixed ensemble validation after verified real smoke."""
import argparse
import ast
import json
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
RUN='owned-detector-ensemble-selection-v1'
WORK='owned_detector_ensemble_selection'


def assignment(source,name):
    return next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id==name)


def build(scoring=False):
    report=(ROOT/'reports/experiments/owned-detector-ensemble-probe-v1-result.json').read_bytes()
    contract=runpy.run_path(str(ROOT/'research/owned_detector_ensemble_contract.py'))['receipt'](
        report,(ROOT/'research/independent_real_baseline_v1_split.json').read_bytes())
    name='biohub-owned-detector-ensemble-scoring-v1' if scoring else 'biohub-'+RUN
    target=ROOT/'kaggle'/name
    if not scoring:
        nb,meta=runpy.run_path(str(ROOT/'scripts/build-detector-spatial-tta-selection.py'))['build']()
        source=''.join(nb['cells'][1]['source']); node=assignment(source,'runtime_sources')
        runtime=ast.literal_eval(node.value)
        # Use current checked runner; all previous launch notebooks remain untouched.
        runtime['run_selection.py']=(ROOT/'scripts/run-independent-selection-inference.py').read_text().replace(
            "run_id='independent-selection-inference-v1'",f"run_id='{RUN}'")
        for module in ('owned_detector_ensemble','owned_detector_ensemble_contract','owned_detector_logit_targets','owned_detector_pu'):
            runtime[module+'.py']=(ROOT/f'research/{module}.py').read_text()
        runtime['spotiflow_biohub/__init__.py']=''
        runtime['spotiflow_biohub/pu_targets.py']=(ROOT/'research/spotiflow_biohub/pu_targets.py').read_text()
        runtime['ensemble_probe_report.json']=report.decode()
        runtime['raw_probe_result.json']=(ROOT/'.biohub/cache/kernel-outputs/owned-detector-logit-probe-v1/owned_detector_logit_probe/outputs/result.json').read_bytes().decode()
        nb['cells'][1]['source']=source.replace(ast.get_source_segment(source,node),'runtime_sources = '+repr(runtime)).splitlines(keepends=True)
        for index in (0,len(nb['cells'])-1):
            source=''.join(nb['cells'][index]['source']).replace('detector-spatial-tta-selection-v1',RUN)
            source=source.replace('detector_spatial_tta_selection',WORK)
            if index==len(nb['cells'])-1:
                marker="command.extend(['--standalone-image-flow','--detector-spatial-tta'])"
                if source.count(marker)!=1: raise ValueError('Frozen launch marker changed')
                locator="""
secondary = next((Path('/kaggle/input')/p/'owned_detector_fit_pair/outputs/pu/last.pt'
    for p in ('biohub-owned-detector-fit-pair-v1','notebooks/indarkarhana/biohub-owned-detector-fit-pair-v1',
              'kernels/indarkarhana/biohub-owned-detector-fit-pair-v1')
    if (Path('/kaggle/input')/p/'owned_detector_fit_pair/outputs/pu/last.pt').is_file()),None)
if secondary is None: raise RuntimeError('Frozen owned PU input missing')
command.extend(['--detector-ensemble-secondary',str(secondary)])
"""
                source=source.replace(marker,marker+locator)
            nb['cells'][index]['source']=source.splitlines(keepends=True)
        nb['metadata']['codex'].update(run_id=RUN,owned_detector_ensemble=contract)
        meta['kernel_sources'].append('indarkarhana/biohub-owned-detector-fit-pair-v1/1')
        title='Biohub Owned Detector Ensemble Selection v1'
    else:
        nb,meta=runpy.run_path(str(ROOT/'scripts/build-detector-spatial-tta-scoring.py'))['build']()
        source=''.join(nb['cells'][-1]['source']); node=assignment(source,'scoring_sources')
        bundle=ast.literal_eval(node.value)
        bundle['selection_launch.ipynb']=(ROOT/'kaggle'/('biohub-'+RUN)/('biohub-'+RUN+'.ipynb')).read_text()
        bundle['research/owned_detector_ensemble_contract.py']=(ROOT/'research/owned_detector_ensemble_contract.py').read_text()
        tail=source[source.index('\nimport runpy'):].replace('biohub-detector-spatial-tta-selection-v1','biohub-'+RUN)
        tail=tail.replace("p/'detector_spatial_tta_selection'",f"p/'{WORK}'")
        nb['cells'][-1]['source']=('scoring_sources = '+repr(bundle)+tail).splitlines(keepends=True)
        source=''.join(nb['cells'][0]['source']).replace('detector_spatial_tta_score',WORK+'_score')
        source=source.replace('detector-spatial-tta-scoring-v1',RUN+'-scoring')
        nb['cells'][0]['source']=source.splitlines(keepends=True)
        nb['metadata']['codex']['run_id']=RUN+'-scoring'
        meta['kernel_sources']=['indarkarhana/biohub-'+RUN+'/1']
        title='Biohub Owned Detector Ensemble Scoring v1'
    meta.update(id='indarkarhana/'+name,title=title,code_file=name+'.ipynb')
    if len(title)>50 or len(name)>50: raise ValueError('Kaggle title/slug exceeds50 characters')
    for cell in nb['cells']: ast.parse(''.join(cell['source']))
    return nb,meta,target


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--scoring',action='store_true')
    nb,meta,target=build(parser.parse_args().scoring)
    if target.exists(): raise ValueError('Refuse to overwrite ensemble selection staging')
    target.mkdir(parents=True)
    (target/meta['code_file']).write_text(json.dumps(nb),encoding='utf-8')
    (target/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(target)
