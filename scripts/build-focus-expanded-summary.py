"""Prepare frozen-head summaries for twelve fitting movies, no diagnostics."""
import ast
import hashlib
import json
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
RUN='focus-expanded-summary-v1';SLUG='biohub-'+RUN
PARENT_SHA='d298b10947a73d96a693175dcca49937ba52567fe85678d66e4fc1e1fec6a8e1'
HEAD_SHA='da576fc4c669ef8327380623082d1a634ce07a745340eba4bfeb464f3a0e0272'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(source,old,new):
    if source.count(old)!=1:raise ValueError('Frozen source adaptation expected exactly one match')
    return source.replace(old,new)


def worker():
    original=(ROOT/'scripts/train-focus-adaptation-head.py').read_text(encoding='utf-8')
    if hashlib.sha256(original.encode()).hexdigest()!=HEAD_SHA:raise ValueError('Exact original forward/loader required')
    source=original.split('    physical=evaluate(True);initial=evaluate();')[0]
    start=source.index("    output=args.features_root/'outputs'")
    end=source.index('    args.output.mkdir',start)
    loader=source[start:end]
    loader=replace_once(loader,"    packets={'fitting':[],'diagnostic':[]}\n",'')
    loader=replace_once(loader,"    output=args.features_root/'outputs'","    output=feature_root/'outputs'")
    loader=replace_once(loader,"spec['feature_worker_result_sha256']","group['feature_worker_result_sha256']")
    loader=replace_once(loader,"spec['feature_spec_file_sha256']","group['feature_spec_file_sha256']")
    loader=replace_once(loader,"feature_spec['contract']!=spec['contract']","feature_spec['contract']!=group['contract']")
    loader=replace_once(loader,"spec['feature_records']","group['feature_records']")
    loader=replace_once(loader,"        if record['role']=='replay':continue","        if record['role']!='fitting':raise ValueError('Fitting-only summary inventory required')")
    loader=replace_once(loader,"    if not all(packets.values()):raise ValueError('Nonempty disjoint fitting and diagnostic examples required')\n",'')
    loader=replace_once(loader,"            counts=validate_pair(packet,coords);row=item['windows'][t]",
        "            counts=validate_pair(packet,coords);row=item['windows'][t]\n            if counts['source_frame']!=t:raise ValueError('Feature frame order changed')\n            if counts['source_nodes']==0 and counts['known_absent']:raise ValueError('Empty-source known null requires separate declared handling')")
    grouped="    packets={'fitting':[],'diagnostic':[]}\n    for group,feature_root in zip(spec['feature_groups'],(args.features_root,args.extra_features_root)):\n"
    grouped+=''.join('    '+line if line.strip() else line for line in loader.splitlines(keepends=True))
    grouped+="    if not packets['fitting'] or packets['diagnostic']:raise ValueError('Only nonempty fitting packets permitted')\n"
    source=source[:start]+grouped+source[end:]
    tail='''    from focus_parent_presence import summarize,metrics
    model.requires_grad_(False).eval()
    records=[];replayed=[]
    with torch.no_grad():
        for stem in spec['contract']['fitting_stems']:
            is_replay=stem in spec['replay_summaries']
            if not is_replay and replayed!=list(spec['replay_summaries']):
                raise ValueError('All four previous fitting summaries must replay before new movies')
            chunks=[]
            for sample in packets['fitting']:
                if sample['stem']!=stem:continue
                scores,_=forward(sample);prior,_=forward(sample,True)
                data=summarize(sample['packet'],scores.cpu().numpy(),prior.cpu().numpy())
                data['source_frame']=np.full(len(data['offset']),int(sample['packet']['source_frame']),np.int64)
                chunks.append(data)
            if not chunks:raise ValueError('Each fixed fitting movie must supply supervised summaries')
            combined={k:np.concatenate([d[k] for d in chunks]) for k in chunks[0]}
            if is_replay:
                reference=args.replay_root/'outputs'/(stem+'.npz')
                if sha(reference)!=spec['replay_summaries'][stem]:raise ValueError('Earlier fitting reference changed')
                with np.load(reference,allow_pickle=False) as saved:
                    if set(saved.files)!=set(combined) or any(not np.array_equal(saved[k],v) for k,v in combined.items()):
                        raise ValueError('Previous fitting summary array replay failed')
                replayed.append(stem)
            path=args.output/(stem+'.npz');np.savez_compressed(path,**combined)
            with np.load(path,allow_pickle=False) as saved:
                if any(not np.array_equal(saved[k],v) for k,v in combined.items()):raise ValueError('Summary round trip changed arrays')
            records.append(dict(stem=stem,role='fitting',sha256=sha(path),rows=len(combined['offset']),baseline=metrics(combined)))
            print(json.dumps(dict(stage='fitting_summary_completed',stem=stem,rows=len(combined['offset']),replayed=is_replay)),flush=True)
    after=tensor_hash(model.state_dict())
    if after!=initial_hash:raise ValueError('Frozen original head changed')
    result=dict(status='completed_focus_expanded_fitting_summaries',records=records,replayed_stems=replayed,
        model_before=initial_hash,model_after=after,source_hashes=source_hashes,
        training_spec_sha256=sha(args.runtime/'training_spec.json'),optimizer_run=False,
        diagnostic_evaluated=False,source_selection_opened=False,new_target_movies_opened=0,
        authorized_for_submission=False,elapsed_seconds=time.monotonic()-started)
    (args.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result),flush=True)
'''
    cli='\n\nif __name__'+original.split('\n\nif __name__')[1]
    cli=replace_once(cli,"'checkpoint','features-root'","'checkpoint','features-root','extra-features-root','replay-root'")
    result=source+tail+cli;ast.parse(result);return result


def parent():
    path=ROOT/'kaggle/biohub-focus-presence-summary-v1/biohub-focus-presence-summary-v1.ipynb'
    if sha(path)!=PARENT_SHA:raise ValueError('Exact successful summary notebook required')
    nb=json.loads(path.read_text());source=''.join(nb['cells'][1]['source'])
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='runtime_sources')
    return path,nb,source,node,ast.literal_eval(node.value)


def make_spec():
    _,_,_,_,runtime=parent();spec=json.loads(runtime['training_spec.json'])
    _,old=runpy.run_path(str(ROOT/'scripts/fit-focus-parent-presence.py'))['load']()
    extra=runpy.run_path(str(ROOT/'scripts/verify-focus-extra-fit-features.py'))['verify'](ROOT/'.biohub/cache/kernel-outputs/focus-extra-fit-features-v1')
    receipt=ROOT/'reports/experiments/focus-extra-fit-features-v1-result.json'
    if json.loads(receipt.read_text())!=extra:raise ValueError('Actual verified additional feature receipt required')
    policy=extra['contract'];previous=spec['contract']['fitting_stems']
    if previous!=policy['previous_fitting_stems'] or spec['contract']['diagnostic_stems']!=policy['unchanged_diagnostic_stems']:
        raise ValueError('Original fitting/diagnostic partition changed')
    keys=('feature_records','feature_worker_result_sha256','feature_spec_file_sha256','contract')
    first={k:spec[k] for k in keys};first['feature_records']=[r for r in first['feature_records'] if r['role']=='fitting']
    second=dict(feature_records=[r for r in extra['records'] if r['role']=='fitting'],
        feature_worker_result_sha256=extra['worker_result_sha256'],contract=policy,
        feature_spec_file_sha256=sha(ROOT/'.biohub/cache/kernel-outputs/focus-extra-fit-features-v1/focus_extra_fit_features/outputs/features_spec.json'))
    spec.update(feature_groups=[first,second],contract=dict(policy,fitting_stems=previous+policy['fitting_stems'],diagnostic_stems=[]),
        replay_summaries={r['stem']:r['sha256'] for r in old['worker']['records'] if r['role']=='fitting'},
        unchanged_diagnostic_summaries={r['stem']:r['sha256'] for r in old['worker']['records'] if r['role']=='diagnostic'},
        extra_feature_receipt_sha256=sha(receipt),previous_summary_worker_sha256=old['worker_result_sha256'])
    for k in keys:
        if k!='contract':spec.pop(k)
    return spec


def build(spec=None):
    if spec is None:spec=make_spec()
    path,nb,source,node,runtime=parent();runtime.pop('source_hashes.json')
    runtime['run_pilot.py']=worker();runtime['training_spec.json']=json.dumps(spec,allow_nan=False)
    runtime.pop('presence_reference.json')
    runtime['source_hashes.json']=json.dumps({k:hashlib.sha256(v.encode()).hexdigest() for k,v in runtime.items()})
    nb['cells'][1]['source']=source.replace(ast.get_source_segment(source,node),'runtime_sources = '+repr(runtime)).splitlines(keepends=True)
    for index in (0,len(nb['cells'])-1):
        source=''.join(nb['cells'][index]['source']).replace('focus-presence-summary-v1',RUN).replace('focus_presence_summary','focus_expanded_summary')
        if index==len(nb['cells'])-1:
            locator="""def required_kernel(slug,relative):
    candidates=[Path('/kaggle/input')/prefix/slug/relative for prefix in ('','notebooks/indarkarhana','kernels/indarkarhana')]
    value=next((p for p in candidates if p.is_dir()),None)
    if value is None:raise RuntimeError('Missing verified kernel input: '+slug)
    return value
extra_features_root=required_kernel('biohub-focus-extra-fit-features-v1','focus_extra_fit_features')
replay_root=required_kernel('biohub-focus-presence-summary-v1','focus_presence_summary')
"""
            source=replace_once(source,'command = [',locator+'command = [')
            source=replace_once(source,"'--features-root', str(features_root),","'--features-root', str(features_root), '--extra-features-root',str(extra_features_root),'--replay-root',str(replay_root),")
        nb['cells'][index]['source']=source.splitlines(keepends=True)
    for cell in nb['cells']:ast.parse(''.join(cell['source']))
    nb['metadata']['codex'].update(run_id=RUN,contract=spec['contract'],optimizer_run=False,
        scope='Frozen original head: exact old four fitting-summary replay then eight new fitting summaries; no diagnostic evaluation',parent_notebook_sha256=PARENT_SHA)
    meta=json.loads((path.parent/'kernel-metadata.json').read_text())
    meta.update(id='indarkarhana/'+SLUG,title=SLUG,code_file=SLUG+'.ipynb',kernel_sources=[
        'indarkarhana/biohub-image-motion-linker-v1/2','indarkarhana/biohub-focus-adaptation-features-v1/1',
        'indarkarhana/biohub-focus-extra-fit-features-v1/1','indarkarhana/biohub-focus-presence-summary-v1/1'])
    return nb,meta


if __name__=='__main__':
    target=ROOT/'kaggle'/SLUG
    if target.exists():raise ValueError('Never overwrite staged or launched experiment')
    nb,meta=build();target.mkdir()
    (target/meta['code_file']).write_text(json.dumps(nb),encoding='utf-8')
    (target/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    (target/'staged_identity.json').write_text(json.dumps(dict(run_id=RUN,notebook_sha256=sha(target/meta['code_file']),
        metadata_sha256=sha(target/'kernel-metadata.json'),builder_sha256=sha(Path(__file__)),status='staged_not_launched'),indent=2),encoding='utf-8')
    print(target)
