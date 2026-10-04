"""Reuse verified frozen-head loading; collect compact presence evidence only."""
import ast
import hashlib
import json
from pathlib import Path
import runpy

ROOT=Path(__file__).resolve().parents[1]
SLUG='biohub-focus-presence-summary-v1'


def worker():
    source=(ROOT/'scripts/train-focus-adaptation-head.py').read_text(encoding='utf-8')
    if hashlib.sha256(source.encode()).hexdigest()!='da576fc4c669ef8327380623082d1a634ce07a745340eba4bfeb464f3a0e0272':
        raise ValueError('Exact completed real-data loader and forward implementation required')
    marker='    physical=evaluate(True);initial=evaluate();'
    prefix=source.split(marker)[0]
    tail='''    from focus_parent_presence import summarize,metrics
    model.requires_grad_(False).eval()
    reference=json.loads((args.runtime/'presence_reference.json').read_text())
    physical=evaluate(True);initial=evaluate()
    for name,value in [('physical_diagnostic',physical),('initial_diagnostic',initial)]:
        old=reference[name]
        if any(value[k]!=old[k] for k in ('known_parent','known_absent','correct_parent','correct_absent')) or abs(value['nll']-old['nll'])>2e-6:
            raise ValueError('Previous completed GPU diagnostic replay failed')
    print(json.dumps(dict(stage='previous_diagnostic_replayed',physical=physical,initial=initial)),flush=True)
    records=[];pooled={}
    with torch.no_grad():
        for role,samples in packets.items():
            for stem in spec['contract'][role+'_stems']:
                chunks=[]
                for sample in samples:
                    if sample['stem']!=stem:continue
                    scores,_=forward(sample);prior,_=forward(sample,True)
                    data=summarize(sample['packet'],scores.cpu().numpy(),prior.cpu().numpy())
                    data['source_frame']=np.full(len(data['offset']),int(sample['packet']['source_frame']),np.int64)
                    chunks.append(data)
                combined={k:np.concatenate([d[k] for d in chunks]) for k in chunks[0]}
                path=args.output/(stem+'.npz');np.savez_compressed(path,**combined)
                with np.load(path,allow_pickle=False) as saved:
                    if any(not np.array_equal(saved[k],v) for k,v in combined.items()):raise ValueError('Presence summary round trip changed arrays')
                records.append(dict(stem=stem,role=role,sha256=sha(path),rows=len(combined['offset']),baseline=metrics(combined)))
                pooled.setdefault(role,[]).append(combined)
    diagnostic={k:np.concatenate([d[k] for d in pooled['diagnostic']]) for k in pooled['diagnostic'][0]}
    summary_replay=metrics(diagnostic)
    if any(summary_replay[k]!=initial[k] for k in ('known_parent','known_absent','correct_parent','correct_absent')) or abs(summary_replay['nll']-initial['nll'])>2e-6:
        raise ValueError('Presence factorization differs from full neural parent/null objective')
    after=tensor_hash(model.state_dict())
    if after!=initial_hash:raise ValueError('Frozen head changed during summary extraction')
    result=dict(status='completed_focus_presence_summaries',records=records,physical_diagnostic=physical,initial_diagnostic=initial,
        summary_replay=summary_replay,model_before=initial_hash,model_after=after,source_hashes=source_hashes,
        training_spec_sha256=sha(args.runtime/'training_spec.json'),optimizer_run=False,source_selection_opened=False,
        new_target_movies_opened=0,authorized_for_submission=False,elapsed_seconds=time.monotonic()-started)
    (args.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result),flush=True)
'''
    result=prefix+tail+'\n\nif __name__'+source.split('\n\nif __name__')[1]
    ast.parse(result);return result


def build():
    prior=runpy.run_path(str(ROOT/'scripts/verify-focus-adaptation-head.py'))['verify'](ROOT/'.biohub/cache/kernel-outputs/focus-adaptation-head-v1')
    nb,meta=runpy.run_path(str(ROOT/'scripts/build-focus-adaptation-head.py'))['build']()
    source=''.join(nb['cells'][1]['source'])
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='runtime_sources')
    runtime=ast.literal_eval(node.value);runtime.pop('source_hashes.json')
    runtime['run_pilot.py']=worker()
    runtime['focus_parent_presence.py']=(ROOT/'research/focus_parent_presence.py').read_text(encoding='utf-8')
    runtime['presence_reference.json']=json.dumps({k:prior['worker'][k] for k in ('physical_diagnostic','initial_diagnostic')})
    runtime['source_hashes.json']=json.dumps({k:hashlib.sha256(v.encode()).hexdigest() for k,v in runtime.items()})
    nb['cells'][1]['source']=source.replace(ast.get_source_segment(source,node),'runtime_sources = '+repr(runtime)).splitlines(keepends=True)
    for index in (0,len(nb['cells'])-1):
        source=''.join(nb['cells'][index]['source']).replace('focus-adaptation-head-v1','focus-presence-summary-v1').replace('focus_adaptation_head','focus_presence_summary')
        nb['cells'][index]['source']=source.splitlines(keepends=True)
    for cell in nb['cells']:ast.parse(''.join(cell['source']))
    nb['metadata']['codex'].update(run_id='focus-presence-summary-v1',scope='Frozen original head compact presence summaries; no optimizer',optimizer_run=False)
    meta.update(id='indarkarhana/'+SLUG,title=SLUG,code_file=SLUG+'.ipynb')
    return nb,meta


if __name__=='__main__':
    target=ROOT/'kaggle'/SLUG
    if target.exists():raise ValueError('Never overwrite staged experiment')
    nb,meta=build();target.mkdir()
    (target/meta['code_file']).write_text(json.dumps(nb),encoding='utf-8')
    (target/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8');print(target)
