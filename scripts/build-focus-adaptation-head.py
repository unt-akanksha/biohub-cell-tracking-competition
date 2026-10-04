"""Immutable head-only adaptation from verified feature evidence."""
import ast
import hashlib
import json
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.focus_adaptation_training import SETTINGS
SLUG='biohub-focus-adaptation-head-v1'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def make_spec():
    receipt=runpy.run_path(str(ROOT/'scripts/verify-focus-adaptation-features.py'))['verify'](ROOT/'.biohub/cache/kernel-outputs/focus-adaptation-features-v1')
    report=ROOT/'reports/experiments/focus-adaptation-features-v1-result.json'
    if json.loads(report.read_text())!=receipt:raise ValueError('Actual complete verified feature receipt required')
    fit=ROOT/'reports/experiments/focus-motion-residual-fit-v1.json'
    if sha(fit)!='fcef805eabaa616102a4588437d0ea7cdcd5ef1cb7a7eeaa6d77138a7834c066':raise ValueError('Exact previous training-only motion calibration required')
    return dict(settings=SETTINGS,contract=receipt['contract'],feature_records=receipt['records'],
        feature_receipt_sha256=sha(report),feature_worker_result_sha256=receipt['worker_result_sha256'],
        feature_spec_file_sha256=sha(ROOT/'.biohub/cache/kernel-outputs/focus-adaptation-features-v1/focus_adaptation_features/outputs/features_spec.json'),
        checkpoint_sha256='76f7da6e32c901e3e3f9d2ab870b8a22235b0b91086cfcf909aa41674896a144',
        model_tensor_sha256='41e82ccfd2e049abbc3d7d11e5eb08b60361b3673b606f848787597de6d62327',
        motion_parameters=json.loads(fit.read_text())['parameters'],motion_fit_sha256=sha(fit),
        design_sha256=sha(ROOT/'reports/experiments/focus-adaptation-head-v1-design.md'))


def build(spec=None):
    if spec is None:spec=make_spec()
    nb,meta=runpy.run_path(str(ROOT/'scripts/build-association-calibration-probe.py'))['build']()
    paths={'run_pilot.py':'scripts/train-focus-adaptation-head.py','split.json':'research/independent_real_baseline_v1_split.json'}
    for name in ('focus_cached_pair','focus_adaptation_training','focus_indexed_parent_loss','sparse_parent_loss','sparse_parent_missing_null','independent_real_baseline'):
        paths[name+'.py']='research/'+name+'.py'
    runtime={name:(ROOT/path).read_text(encoding='utf-8') for name,path in paths.items()}
    runtime['training_spec.json']=json.dumps(spec,allow_nan=False)
    runtime['source_hashes.json']=json.dumps({k:hashlib.sha256(v.encode()).hexdigest() for k,v in runtime.items()})
    source=''.join(nb['cells'][1]['source'])
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='runtime_sources')
    nb['cells'][1]['source']=source.replace(ast.get_source_segment(source,node),'runtime_sources = '+repr(runtime)).splitlines(keepends=True)
    for index in (0,len(nb['cells'])-1):
        source=''.join(nb['cells'][index]['source']).replace('association-calibration-probe-v1','focus-adaptation-head-v1').replace('association_calibration_probe','focus_adaptation_head')
        source=source.replace("subprocess.run([sys.executable,'-m','pytest',str(runtime/'tests'),'-q'],cwd=runtime,timeout=180,check=True)\n",'')
        if index==len(nb['cells'])-1:
            locator="""feature_candidates=[Path('/kaggle/input')/prefix/'biohub-focus-adaptation-features-v1/focus_adaptation_features' for prefix in ('','notebooks/indarkarhana','kernels/indarkarhana')]
features_root=next((p for p in feature_candidates if p.is_dir()),None)
if features_root is None:raise RuntimeError('Verified frozen feature input missing')
"""
            source=source.replace('command = [',locator+'command = [').replace("'--checkpoint', str(checkpoint),","'--features-root', str(features_root), '--checkpoint', str(checkpoint),")
        nb['cells'][index]['source']=source.splitlines(keepends=True)
    for cell in nb['cells']:ast.parse(''.join(cell['source']))
    nb['metadata']['codex'].update(run_id='focus-adaptation-head-v1',scope='800 head-only steps; four-step smoke gate; four fitting/four diagnostic training movies',
        contract=spec['contract'],declared_budget_seconds=3600)
    nb['metadata']['codex'].pop('windows_per_movie',None)
    meta.update(id='indarkarhana/'+SLUG,title=SLUG,code_file=SLUG+'.ipynb',is_private=True,enable_internet=False,
        enable_gpu=True,enable_tpu=False,kernel_sources=['indarkarhana/biohub-image-motion-linker-v1/2','indarkarhana/biohub-focus-adaptation-features-v1/1'])
    return nb,meta


if __name__=='__main__':
    target=ROOT/'kaggle'/SLUG
    if target.exists():raise ValueError('Never overwrite a staged or launched experiment')
    nb,meta=build();target.mkdir()
    (target/meta['code_file']).write_text(json.dumps(nb),encoding='utf-8')
    (target/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8');print(target)
