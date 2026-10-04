"""Build only after full training-only calibration passes; never launch here."""
import argparse
import ast
import json
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
RUN='owned-detector-calibrated-selection-v1'
WORK='owned_detector_calibrated_selection'


def assignment(source,name):
    return next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id==name)


def build(scoring=False):
    gate=runpy.run_path(str(ROOT/'research/detector_confidence_calibration.py'))
    report=(ROOT/'reports/experiments/detector-calibration-full-v1-result.json').read_bytes()
    split=json.loads((ROOT/'research/independent_real_baseline_v1_split.json').read_text())
    receipt=gate['receipt'](report,split,gate['SPARSE_SHA'])
    name='biohub-'+RUN+('-scoring' if scoring else '')
    target=ROOT/'kaggle'/name
    if not scoring:
        nb,meta,_=runpy.run_path(str(ROOT/'scripts/build-owned-detector-selection.py'))['build']('sparse')
        source=''.join(nb['cells'][1]['source']); node=assignment(source,'runtime_sources')
        runtime=ast.literal_eval(node.value)
        for module in ('detector_confidence_calibration','detector_calibration_records'):
            runtime[module+'.py']=(ROOT/f'research/{module}.py').read_text()
        runtime['detector_calibration_result.json']=report.decode()
        nb['cells'][1]['source']=source.replace(ast.get_source_segment(source,node),'runtime_sources = '+repr(runtime)).splitlines(keepends=True)
        for index in (0,len(nb['cells'])-1):
            source=''.join(nb['cells'][index]['source']).replace('owned-detector-sparse-selection-v1',RUN)
            source=source.replace('owned_detector_sparse_selection',WORK)
            if index==len(nb['cells'])-1:
                marker="command.extend(['--standalone-image-flow','--detector-spatial-tta'])"
                if source.count(marker)!=1: raise ValueError('Frozen detector inference marker changed')
                source=source.replace(marker,marker+"\ncommand.extend(['--detector-calibration-json',str(runtime/'detector_calibration_result.json')])")
            nb['cells'][index]['source']=source.splitlines(keepends=True)
        nb['metadata']['codex'].update(run_id=RUN,detector_confidence_calibration=receipt)
        title='Biohub Owned Detector Calibrated Selection v1'
    else:
        nb,meta,_=runpy.run_path(str(ROOT/'scripts/build-owned-detector-scoring.py'))['build']('sparse')
        source=''.join(nb['cells'][-1]['source']); node=assignment(source,'scoring_sources')
        bundle=ast.literal_eval(node.value)
        bundle['selection_launch.ipynb']=(ROOT/'kaggle'/('biohub-'+RUN)/('biohub-'+RUN+'.ipynb')).read_text()
        for module in ('detector_confidence_calibration','detector_calibration_records'):
            bundle['research/'+module+'.py']=(ROOT/f'research/{module}.py').read_text()
        tail=source[source.index('\nimport runpy'):].replace('biohub-owned-detector-sparse-selection-v1','biohub-'+RUN)
        tail=tail.replace("p/'owned_detector_sparse_selection'",f"p/'{WORK}'")
        nb['cells'][-1]['source']=('scoring_sources = '+repr(bundle)+tail).splitlines(keepends=True)
        source=''.join(nb['cells'][0]['source']).replace('owned_detector_sparse_score',WORK+'_score')
        source=source.replace('owned-detector-sparse-scoring-v1',RUN+'-scoring')
        nb['cells'][0]['source']=source.splitlines(keepends=True)
        nb['metadata']['codex']['run_id']=RUN+'-scoring'
        meta['kernel_sources']=['indarkarhana/biohub-'+RUN+'/1']
        title='Biohub Owned Detector Calibrated Selection v1 Scoring'
    meta.update(id='indarkarhana/'+name,title=title,code_file=name+'.ipynb')
    for cell in nb['cells']: ast.parse(''.join(cell['source']))
    return nb,meta,target


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--scoring',action='store_true')
    nb,meta,target=build(parser.parse_args().scoring)
    if target.exists(): raise ValueError('Refuse to overwrite calibrated selection staging')
    target.mkdir(parents=True)
    (target/meta['code_file']).write_text(json.dumps(nb),encoding='utf-8')
    (target/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(target)
