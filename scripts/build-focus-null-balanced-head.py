"""Immutable twelve-movie head experiment with fitting-count-derived null loss."""
import ast
import hashlib
import json
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
RUN='focus-null-balanced-head-v1';SLUG='biohub-'+RUN
PARENT_SHA='e2df68e7e4355840197d42f0c18c9a609c31f1786a32a9b5b74f861dad8a9288'
EXPANDED=runpy.run_path(str(ROOT/'scripts/build-focus-expanded-summary.py'))
replace_once=EXPANDED['replace_once']


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def parent():
    path=ROOT/'kaggle/biohub-focus-adaptation-head-v1/biohub-focus-adaptation-head-v1.ipynb'
    if sha(path)!=PARENT_SHA:raise ValueError('Exact completed head notebook required')
    nb=json.loads(path.read_text());source=''.join(nb['cells'][1]['source'])
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='runtime_sources')
    return path,nb,source,node,ast.literal_eval(node.value)


def worker():
    original=(ROOT/'scripts/train-focus-adaptation-head.py').read_text(encoding='utf-8')
    if hashlib.sha256(original.encode()).hexdigest()!=EXPANDED['HEAD_SHA']:raise ValueError('Original worker changed')
    expanded=EXPANDED['worker']()
    start=expanded.index("    packets={'fitting':[],'diagnostic':[]}")
    loader=expanded[start:expanded.index('    args.output.mkdir',start)]
    loader=replace_once(loader,"            if record['role']!='fitting':raise ValueError('Fitting-only summary inventory required')",
        "            if record['role']=='replay':continue\n            if record['role'] not in ('fitting','diagnostic'):raise ValueError('Unexpected role')\n            if record['role']=='diagnostic' and group is not spec['feature_groups'][0]:raise ValueError('New group cannot supply diagnostics')")
    loader=replace_once(loader,"    if not packets['fitting'] or packets['diagnostic']:raise ValueError('Only nonempty fitting packets permitted')",
        "    if not all(packets.values()):raise ValueError('Both roles required')\n    for role in packets:\n        expected=set(spec['contract'][role+'_stems'])\n        if {s['stem'] for s in packets[role]}!=expected:raise ValueError('Complete fixed role inventory required')\n    if set(spec['contract']['fitting_stems']) & set(spec['contract']['diagnostic_stems']):raise ValueError('Role overlap')\n    counts={role:{key:sum(int(((s['packet']['labels']>=0)&(s['packet']['labels']<len(s['packet']['source_indices']))).sum()) if key=='known_parent' else int((s['packet']['labels']==len(s['packet']['source_indices'])).sum()) for s in rows) for key in ('known_parent','known_absent')} for role,rows in packets.items()}\n    if counts!=spec['supervision_counts']:raise ValueError('Declared fitting/diagnostic counts changed')\n    null_weight=fitting_null_weight(**counts['fitting'])\n    if null_weight!=spec['null_weight']:raise ValueError('Fitting-only loss weight changed')")
    a=original.index("    output=args.features_root/'outputs'");b=original.index('    args.output.mkdir',a)
    source=original[:a]+loader+original[b:]
    source=replace_once(source,'    from focus_adaptation_training import SETTINGS,diagnostic_gate','    from focus_adaptation_training import SETTINGS,diagnostic_gate\n    from focus_null_balanced_loss import fitting_null_weight,null_balanced_parent_loss')
    source=replace_once(source,'    optimizer=torch.optim.AdamW',
        "    for name,actual in [('physical_diagnostic',physical),('initial_diagnostic',initial)]:\n        reference=spec['initial_controls'][name]\n        if any(actual[k]!=reference[k] for k in ('known_parent','known_absent','correct_parent','correct_absent')) or abs(actual['nll']-reference['nll'])>2e-6:raise ValueError('Initial diagnostic replay failed before optimizer')\n    optimizer=torch.optim.AdamW")
    source=replace_once(source,'        scores,labels=forward(sample);loss=indexed_parent_loss(scores,labels)',
        '        scores,labels=forward(sample);loss=null_balanced_parent_loss(scores,labels,null_weight)')
    source=replace_once(source,"status='completed_focus_head_adaptation',settings=SETTINGS", "status='completed_focus_null_balanced_head',null_weight=null_weight,supervision_counts=counts,settings=SETTINGS")
    source=replace_once(source,"'checkpoint','features-root'", "'checkpoint','features-root','extra-features-root'")
    ast.parse(source);return source


def make_spec():
    spec=EXPANDED['make_spec']()
    old=json.loads(parent()[-1]['training_spec.json'])
    spec['feature_groups'][0]['feature_records']=old['feature_records']
    spec['contract']['diagnostic_stems']=old['contract']['diagnostic_stems']
    for key in ('replay_summaries','unchanged_diagnostic_summaries'):spec.pop(key)
    from research.focus_null_balanced_loss import fitting_null_weight
    counts={role:{key:sum(r['counts'][key] for g in spec['feature_groups'] for r in g['feature_records'] if r['role']==role) for key in ('known_parent','known_absent')} for role in ('fitting','diagnostic')}
    if counts!={'fitting':{'known_parent':10754,'known_absent':161},'diagnostic':{'known_parent':2645,'known_absent':27}}:raise ValueError('Prespecified counts changed')
    smoke_path=ROOT/'reports/experiments/focus-null-loss-cpu-v1-result.json'
    if sha(smoke_path)!='1cd972746085aee0ba0d252c473d4594a5142a06a244762c087f59401db039a9':raise ValueError('Verified CPU smoke required')
    controls=json.loads((ROOT/'reports/experiments/focus-adaptation-head-v1-result.json').read_text())['worker']
    spec.update(supervision_counts=counts,null_weight=fitting_null_weight(**counts['fitting']),
        initial_controls={k:controls[k] for k in ('physical_diagnostic','initial_diagnostic')},
        cpu_loss_smoke_sha256=sha(smoke_path),design_sha256=sha(ROOT/f'reports/experiments/{RUN}-design.md'))
    return spec


def build(spec=None):
    if spec is None:spec=make_spec()
    path,nb,source,node,runtime=parent();runtime.pop('source_hashes.json')
    loss=ROOT/'research/focus_null_balanced_loss.py'
    if sha(loss)!='a17d0c33a2eab536e4b3e5ade3d543ea29e2478e0420bf0179ed5b988ed36c4b':raise ValueError('CPU-tested loss required')
    runtime.update({'run_pilot.py':worker(),'focus_null_balanced_loss.py':loss.read_text(encoding='utf-8'),'training_spec.json':json.dumps(spec,allow_nan=False)})
    runtime['source_hashes.json']=json.dumps({k:hashlib.sha256(v.encode()).hexdigest() for k,v in runtime.items()})
    nb['cells'][1]['source']=source.replace(ast.get_source_segment(source,node),'runtime_sources = '+repr(runtime)).splitlines(keepends=True)
    for index in (0,len(nb['cells'])-1):
        source=''.join(nb['cells'][index]['source']).replace('focus-adaptation-head-v1',RUN).replace('focus_adaptation_head','focus_null_balanced_head')
        if index==len(nb['cells'])-1:
            locator="""extra_candidates=[Path('/kaggle/input')/prefix/'biohub-focus-extra-fit-features-v1/focus_extra_fit_features' for prefix in ('','notebooks/indarkarhana','kernels/indarkarhana')]
extra_features_root=next((p for p in extra_candidates if p.is_dir()),None)
if extra_features_root is None:raise RuntimeError('Verified additional fitting features missing')
"""
            source=replace_once(source,'command = [',locator+'command = [')
            source=replace_once(source,"'--features-root', str(features_root),", "'--features-root', str(features_root), '--extra-features-root',str(extra_features_root),")
        nb['cells'][index]['source']=source.splitlines(keepends=True)
    for cell in nb['cells']:ast.parse(''.join(cell['source']))
    nb['metadata']['codex'].update(run_id=RUN,contract=spec['contract'],scope='Twelve fitting/four diagnostic; fixed fitting-count null weighting; four-step real smoke then 800 head steps',parent_notebook_sha256=PARENT_SHA)
    meta=json.loads((path.parent/'kernel-metadata.json').read_text())
    meta.update(id='indarkarhana/'+SLUG,title=SLUG,code_file=SLUG+'.ipynb')
    meta['kernel_sources'].append('indarkarhana/biohub-focus-extra-fit-features-v1/1')
    return nb,meta


if __name__=='__main__':
    target=ROOT/'kaggle'/SLUG
    if target.exists():raise ValueError('Never overwrite staged experiment')
    nb,meta=build();target.mkdir()
    (target/meta['code_file']).write_text(json.dumps(nb),encoding='utf-8')
    (target/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    (target/'staged_identity.json').write_text(json.dumps(dict(run_id=RUN,notebook_sha256=sha(target/meta['code_file']),metadata_sha256=sha(target/'kernel-metadata.json'),builder_sha256=sha(Path(__file__)),status='staged_not_launched'),indent=2),encoding='utf-8')
    print(target)
