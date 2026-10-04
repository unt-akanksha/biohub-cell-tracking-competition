"""Separate exposed-target diagnostic; never modifies an existing notebook."""
import argparse
import ast
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
RUN = 'owned-detector-pu-transfer-v1'
WORK = 'owned_detector_pu_transfer'


def assignment(source, name):
    return next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id==name)


def build(scoring=False):
    gate = runpy.run_path(str(ROOT/'research/owned_detector_transfer_contract.py'))
    report = (ROOT/'reports/experiments/owned-detector-selection-v1-comparison.json').read_bytes()
    receipt = gate['verify'](report,(ROOT/'research/independent_real_baseline_v1_split.json').read_bytes())
    name = 'biohub-'+RUN+('-scoring' if scoring else '')
    target = ROOT/'kaggle'/name
    if not scoring:
        nb,meta,_ = runpy.run_path(str(ROOT/'scripts/build-owned-detector-selection.py'))['build']('pu')
        source = ''.join(nb['cells'][1]['source']); node = assignment(source,'runtime_sources')
        runtime = ast.literal_eval(node.value)
        runtime['owned_detector_transfer_contract.py'] = (ROOT/'research/owned_detector_transfer_contract.py').read_text()
        runtime['frozen_owned_comparison.json'] = report.decode()
        nb['cells'][1]['source'] = source.replace(ast.get_source_segment(source,node),'runtime_sources = '+repr(runtime)).splitlines(keepends=True)
        for index in (0,len(nb['cells'])-1):
            source = ''.join(nb['cells'][index]['source']).replace('owned-detector-pu-selection-v1',RUN)
            source = source.replace('owned_detector_pu_selection',WORK)
            if index==len(nb['cells'])-1:
                marker = "command.extend(['--standalone-image-flow','--detector-spatial-tta'])"
                if source.count(marker)!=1: raise ValueError('Frozen inference command marker changed')
                source = source.replace(marker,marker+"\ncommand.append('--owned-transfer-diagnostic')")
            nb['cells'][index]['source'] = source.splitlines(keepends=True)
        nb['metadata']['codex'].update(run_id=RUN,owned_transfer_diagnostic=receipt,target_audit_opened=True)
        title = 'Biohub Owned Detector PU Transfer v1'
    else:
        nb,meta,_ = runpy.run_path(str(ROOT/'scripts/build-owned-detector-scoring.py'))['build']('pu')
        source = ''.join(nb['cells'][-1]['source']); node = assignment(source,'scoring_sources')
        bundle = ast.literal_eval(node.value)
        bundle['selection_launch.ipynb'] = (ROOT/'kaggle'/('biohub-'+RUN)/('biohub-'+RUN+'.ipynb')).read_text()
        bundle['research/owned_detector_transfer_contract.py'] = (ROOT/'research/owned_detector_transfer_contract.py').read_text()
        tail = source[source.index('\nimport runpy'):].replace('biohub-owned-detector-pu-selection-v1','biohub-'+RUN)
        tail = tail.replace("p/'owned_detector_pu_selection'",f"p/'{WORK}'")
        nb['cells'][-1]['source'] = ('scoring_sources = '+repr(bundle)+tail).splitlines(keepends=True)
        source = ''.join(nb['cells'][0]['source']).replace('owned_detector_pu_score',WORK+'_score')
        source = source.replace('owned-detector-pu-scoring-v1',RUN+'-scoring')
        nb['cells'][0]['source'] = source.splitlines(keepends=True)
        nb['metadata']['codex']['run_id'] = RUN+'-scoring'
        meta['kernel_sources'] = ['indarkarhana/biohub-'+RUN+'/1']
        title = 'Biohub Owned Detector PU Transfer v1 Scoring'
    meta.update(id='indarkarhana/'+name,title=title,code_file=name+'.ipynb')
    for cell in nb['cells']: ast.parse(''.join(cell['source']))
    return nb,meta,target


if __name__=='__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--scoring',action='store_true')
    nb,meta,target = build(parser.parse_args().scoring)
    if target.exists(): raise ValueError('Refuse to overwrite immutable diagnostic staging')
    target.mkdir(parents=True)
    (target/meta['code_file']).write_text(json.dumps(nb),encoding='utf-8')
    (target/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(target)
