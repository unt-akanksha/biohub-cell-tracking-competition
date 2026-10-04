"""Source-only motion averaging with exact retained detector coordinates."""
import argparse
import ast
import json
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
RUN='flow-spatial-tta-selection-v1'
WORK='flow_spatial_tta_selection'


def assignment(source,name):
    return next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id==name)


def build(scoring=False):
    report=(ROOT/'reports/experiments/backward-flow-spatial-tta-probe-v2-result.json').read_bytes()
    contract=runpy.run_path(str(ROOT/'research/flow_spatial_tta_contract.py'))['receipt'](
        report,(ROOT/'research/independent_real_baseline_v1_split.json').read_bytes())
    name='biohub-flow-spatial-tta-scoring-v1' if scoring else 'biohub-'+RUN
    target=ROOT/'kaggle'/name
    if not scoring:
        nb,meta=runpy.run_path(str(ROOT/'scripts/build-detector-spatial-tta-selection.py'))['build']()
        source=''.join(nb['cells'][1]['source']); node=assignment(source,'runtime_sources')
        runtime=ast.literal_eval(node.value)
        runtime['run_selection.py']=(ROOT/'scripts/run-independent-selection-inference.py').read_text().replace(
            "run_id='independent-selection-inference-v1'",f"run_id='{RUN}'")
        for module in ('backward_flow_spatial_tta','flow_spatial_tta_contract','calibrated_motion_scores','feature_tta_reference'):
            runtime[module+'.py']=(ROOT/f'research/{module}.py').read_text()
        runtime['flow_spatial_tta_probe.json']=report.decode()
        nb['cells'][1]['source']=source.replace(ast.get_source_segment(source,node),'runtime_sources = '+repr(runtime)).splitlines(keepends=True)
        for index in (0,len(nb['cells'])-1):
            source=''.join(nb['cells'][index]['source']).replace('detector-spatial-tta-selection-v1',RUN)
            source=source.replace('detector_spatial_tta_selection',WORK)
            if index==len(nb['cells'])-1:
                marker="command.extend(['--standalone-image-flow','--detector-spatial-tta'])"
                if source.count(marker)!=1: raise ValueError('Parent detector launch marker changed')
                locator="""
reference = next((Path('/kaggle/input')/p/'detector_spatial_tta_selection'
    for p in ('biohub-detector-spatial-tta-selection-v1','notebooks/indarkarhana/biohub-detector-spatial-tta-selection-v1',
              'kernels/indarkarhana/biohub-detector-spatial-tta-selection-v1')
    if (Path('/kaggle/input')/p/'detector_spatial_tta_selection/outputs/selection_manifest.json').is_file()),None)
if reference is None: raise RuntimeError('Frozen complete parent D4 reference missing')
command.extend(['--flow-spatial-tta-reference',str(reference)])
"""
                source=source.replace(marker,marker+locator)
            nb['cells'][index]['source']=source.splitlines(keepends=True)
        nb['metadata']['codex'].update(run_id=RUN,flow_spatial_tta=contract,
            node_reference_kernel='indarkarhana/biohub-detector-spatial-tta-selection-v1/1')
        meta['kernel_sources'].append('indarkarhana/biohub-detector-spatial-tta-selection-v1/1')
        title='Biohub Flow Spatial TTA Selection v1'
    else:
        nb,meta=runpy.run_path(str(ROOT/'scripts/build-detector-spatial-tta-scoring.py'))['build']()
        source=''.join(nb['cells'][-1]['source']); node=assignment(source,'scoring_sources')
        bundle=ast.literal_eval(node.value)
        bundle['selection_launch.ipynb']=(ROOT/'kaggle'/('biohub-'+RUN)/('biohub-'+RUN+'.ipynb')).read_text()
        bundle['research/flow_spatial_tta_contract.py']=(ROOT/'research/flow_spatial_tta_contract.py').read_text()
        tail=source[source.index('\nimport runpy'):].replace('biohub-detector-spatial-tta-selection-v1','biohub-'+RUN)
        tail=tail.replace("p/'detector_spatial_tta_selection'",f"p/'{WORK}'")
        nb['cells'][-1]['source']=('scoring_sources = '+repr(bundle)+tail).splitlines(keepends=True)
        source=''.join(nb['cells'][0]['source']).replace('detector_spatial_tta_score','flow_spatial_tta_score')
        source=source.replace('detector-spatial-tta-scoring-v1','flow-spatial-tta-scoring-v1')
        nb['cells'][0]['source']=source.splitlines(keepends=True)
        nb['metadata']['codex']['run_id']='flow-spatial-tta-scoring-v1'
        meta['kernel_sources']=['indarkarhana/biohub-'+RUN+'/1']
        title='Biohub Flow Spatial TTA Scoring v1'
    if len(name)>50 or len(title)>50: raise ValueError('Short Kaggle title/slug required')
    meta.update(id='indarkarhana/'+name,title=title,code_file=name+'.ipynb')
    for cell in nb['cells']: ast.parse(''.join(cell['source']))
    return nb,meta,target


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--scoring',action='store_true')
    nb,meta,target=build(parser.parse_args().scoring)
    if target.exists(): raise ValueError('Refuse to overwrite flow selection staging')
    target.mkdir(parents=True)
    (target/meta['code_file']).write_text(json.dumps(nb),encoding='utf-8')
    (target/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(target)
