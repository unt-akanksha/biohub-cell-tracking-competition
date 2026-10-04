"""Stage frozen source-gated flow transfer and its offline CPU scorer."""
import argparse
import ast
import json
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
RUN='flow-tta-transfer-v1'
WORK='flow_tta_transfer'


def assignment(source,name):
    return next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id==name)


def build(scoring=False):
    source_report=(ROOT/'reports/experiments/flow-spatial-tta-selection-v1-result.json').read_bytes()
    probe=(ROOT/'reports/experiments/backward-flow-spatial-tta-probe-v2-result.json').read_bytes()
    split=(ROOT/'research/independent_real_baseline_v1_split.json').read_bytes()
    policy=runpy.run_path(str(ROOT/'research/flow_transfer_contract.py'))['verify'](source_report,probe,split)
    name='biohub-flow-tta-transfer-scoring-v1' if scoring else 'biohub-'+RUN
    target=ROOT/'kaggle'/name
    parent=runpy.run_path(str(ROOT/'scripts/build-flow-spatial-tta-selection.py'))
    nb,meta,_=parent['build'](scoring)
    if not scoring:
        source=''.join(nb['cells'][1]['source']); node=assignment(source,'runtime_sources')
        runtime=ast.literal_eval(node.value)
        runtime['run_selection.py']=runtime['run_selection.py'].replace("run_id='flow-spatial-tta-selection-v1'",f"run_id='{RUN}'")
        runtime['flow_transfer_contract.py']=(ROOT/'research/flow_transfer_contract.py').read_text()
        runtime['frozen_flow_source_comparison.json']=source_report.decode()
        nb['cells'][1]['source']=source.replace(ast.get_source_segment(source,node),'runtime_sources = '+repr(runtime)).splitlines(keepends=True)
        for index in (0,len(nb['cells'])-1):
            source=''.join(nb['cells'][index]['source']).replace('flow-spatial-tta-selection-v1',RUN)
            source=source.replace('flow_spatial_tta_selection',WORK)
            if index==len(nb['cells'])-1:
                source=source.replace('biohub-detector-spatial-tta-selection-v1','biohub-detector-spatial-tta-embryo-audit-v1')
                source=source.replace('detector_spatial_tta_selection','detector_spatial_tta_audit')
                anchor="command.extend(['--flow-spatial-tta-reference',str(reference)])"
                if source.count(anchor)!=1: raise ValueError('Transfer launch anchor changed')
                source=source.replace(anchor,anchor+"\ncommand.append('--flow-transfer-diagnostic')")
            nb['cells'][index]['source']=source.splitlines(keepends=True)
        nb['metadata']['codex'].update(run_id=RUN,flow_spatial_tta=policy,target_audit_opened=True,
            node_reference_kernel='indarkarhana/biohub-detector-spatial-tta-embryo-audit-v1/1')
        meta['kernel_sources']=['indarkarhana/biohub-image-motion-linker-v1/2',
            'indarkarhana/biohub-detector-spatial-tta-embryo-audit-v1/1']
        title='Biohub Flow TTA Transfer v1'
    else:
        source=''.join(nb['cells'][-1]['source']); node=assignment(source,'scoring_sources')
        bundle=ast.literal_eval(node.value)
        bundle['selection_launch.ipynb']=(ROOT/'kaggle'/('biohub-'+RUN)/('biohub-'+RUN+'.ipynb')).read_text()
        bundle['research/flow_transfer_contract.py']=(ROOT/'research/flow_transfer_contract.py').read_text()
        tail=source[source.index('\nimport runpy'):].replace('biohub-flow-spatial-tta-selection-v1','biohub-'+RUN)
        tail=tail.replace("p/'flow_spatial_tta_selection'",f"p/'{WORK}'")
        nb['cells'][-1]['source']=('scoring_sources = '+repr(bundle)+tail).splitlines(keepends=True)
        source=''.join(nb['cells'][0]['source']).replace('flow_spatial_tta_score',WORK+'_score')
        source=source.replace('flow-spatial-tta-scoring-v1','flow-tta-transfer-scoring-v1')
        nb['cells'][0]['source']=source.splitlines(keepends=True)
        nb['metadata']['codex']['run_id']='flow-tta-transfer-scoring-v1'
        meta['kernel_sources']=['indarkarhana/biohub-'+RUN+'/1']
        title='Biohub Flow TTA Transfer Scoring v1'
    meta.update(id='indarkarhana/'+name,title=title,code_file=name+'.ipynb')
    if max(len(name),len(title))>50: raise ValueError('Short title/slug required')
    for cell in nb['cells']: ast.parse(''.join(cell['source']))
    return nb,meta,target


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--scoring',action='store_true')
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args(); nb,meta,target=build(args.scoring)
    if args.check:
        print(json.dumps(dict(target=str(target),gpu=meta['enable_gpu'],run_id=nb['metadata']['codex']['run_id'])))
        raise SystemExit(0)
    if target.exists(): raise ValueError('Refuse to overwrite frozen transfer staging')
    target.mkdir(parents=True)
    (target/meta['code_file']).write_text(json.dumps(nb),encoding='utf-8')
    (target/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(target)
