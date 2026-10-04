"""Repair a single-point functional assertion; same100-step model experiment.

Never overwrite v1. Save resumable checkpoints and diagnostics BEFORE audit.
"""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SLUG='biohub-blindspot-real-probe-v2'
PARENT_SHA='4aab44d86d15688621f2cc983e39604e609de28b31751522fc109754412f3063'


def replace_once(source,before,after):
    if source.count(before)!=1: raise ValueError('Frozen repair site changed')
    return source.replace(before,after)


def build():
    folder=ROOT/'kaggle/biohub-blindspot-real-probe-v1'
    parent=folder/'biohub-blindspot-real-probe-v1.ipynb'
    if hashlib.sha256(parent.read_bytes()).hexdigest()!=PARENT_SHA:
        raise ValueError('Original executed notebook changed')
    nb=json.loads(parent.read_text())
    source=''.join(nb['cells'][1]['source'])
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
              and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='runtime_sources')
    runtime=ast.literal_eval(node.value)
    worker=runtime['run_pilot.py']
    worker=replace_once(worker,"policy = identity(args.manifest.read_bytes())",
                        "policy = identity(args.manifest.read_bytes())\n    policy['run_id']='blindspot-real-probe-v2'")
    checkpoint="""    checkpoint=args.output/'last.pt'
    torch.save(dict(model={k:v.detach().cpu() for k,v in core.state_dict().items()},identity=policy,step=100),checkpoint)
"""
    worker=replace_once(worker,checkpoint,'')
    worker=replace_once(worker,'    model.eval()\n    rows=[]',checkpoint+'    model.eval()\n    rows=[]')
    old="""    if max(r['prediction_std'] for r in rows)<=0: raise ValueError('Collapsed constant restoration')
    sample=training[:1,:,:17,:18,:19].cuda().requires_grad_(True)
    gradient,=torch.autograd.grad(core(sample)[0,0,8,9,9],sample)
    if float(gradient[0,0,8,9,9])!=0 or float(gradient.abs().sum())<=0:
        raise ValueError('Real trained model failed nontrivial blind spot check')
"""
    new="""    (args.output/'diagnostic_snapshot.json').write_text(json.dumps(dict(identity=policy,steps=100,losses=losses,
        per_movie=rows,comparison=proxy_gate(rows,policy['diagnostic_stems']),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        note='Saved before functional audit; not promotion authorization'),indent=2,allow_nan=False))
    from blindspot_functional_audit import audit
    sensitivity=audit(core,training[[0,len(training)//2,len(training)-1]].cuda())
    (args.output/'functional_audit.json').write_text(json.dumps(sensitivity,indent=2,allow_nan=False))
    if not sensitivity['passed']:
        raise ValueError('Functional audit failed: '+str(sensitivity['conditions']))
    sample=training[:1,:,:17,:18,:19].cuda()
"""
    worker=replace_once(worker,old,new)
    worker=replace_once(worker,"trained_center_gradient=0.,trained_other_gradient_l1=float(gradient.abs().sum()),",
        "trained_center_gradient=0.,trained_other_gradient_l1=sensitivity['observed_gradient_l1_max'],functional_audit=sensitivity,")
    line="        if (step+1)%20==0: print(json.dumps(dict(step=step+1,loss=losses[-1])),flush=True)"
    worker=replace_once(worker,line,"""        if (step+1)%20==0:
            torch.save(dict(model={k:v.detach().cpu() for k,v in core.state_dict().items()},
                optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),loader_generator=rng.get_state(),
                torch_rng_cpu=torch.get_rng_state(),torch_rng_cuda=torch.cuda.get_rng_state_all(),
                identity=policy,step=step+1,losses=losses),args.output/f'step_{step+1:04d}.pt')
            print(json.dumps(dict(step=step+1,loss=losses[-1],resumable_checkpoint_saved=True)),flush=True)""")
    runtime['run_pilot.py']=worker
    runtime['blindspot_functional_audit.py']=(ROOT/'research/blindspot_functional_audit.py').read_text()
    nb['cells'][1]['source']=source.replace(ast.get_source_segment(source,node),'runtime_sources = '+repr(runtime)).splitlines(keepends=True)
    for i in (0,len(nb['cells'])-1):
        nb['cells'][i]['source']=''.join(nb['cells'][i]['source']).replace('blindspot-real-probe-v1','blindspot-real-probe-v2').replace('blindspot_real_probe','blindspot_real_probe_v2').splitlines(keepends=True)
    for cell in nb['cells']: ast.parse(''.join(cell['source']))
    nb['metadata']['codex'].update(run_id='blindspot-real-probe-v2',repair_parent_sha256=PARENT_SHA,
        repair='Separate local ReLU inactivity from global sensitivity; save checkpoints/diagnostics before audit')
    meta=json.loads((folder/'kernel-metadata.json').read_text())
    meta.update(id='indarkarhana/'+SLUG,title=SLUG,code_file=SLUG+'.ipynb')
    return nb,meta


if __name__=='__main__':
    target=ROOT/'kaggle'/SLUG
    if target.exists(): raise ValueError('Do not overwrite staged or executed repair')
    nb,meta=build(); target.mkdir()
    (target/meta['code_file']).write_text(json.dumps(nb))
    (target/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    print(target)
