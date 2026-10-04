"""Immutable one-hour offline ensemble smoke with GPU tests before images."""
import ast
import hashlib
import json
from pathlib import Path
import runpy

ROOT=Path(__file__).resolve().parents[1]
RUN='owned-detector-ensemble-probe-v1'
TARGET=ROOT/'kaggle'/('biohub-'+RUN)


def build():
    nb,meta=runpy.run_path(str(ROOT/'scripts/build-detector-spatial-tta-probe.py'))['build']()
    source=''.join(nb['cells'][1]['source'])
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='runtime_sources')
    runtime=ast.literal_eval(node.value)
    runtime['run_pilot.py']=(ROOT/'scripts/run-owned-detector-ensemble-probe.py').read_text()
    runtime['selection_contract.py']=(ROOT/'scripts/run-independent-selection-inference.py').read_text()
    for name in ('owned_detector_ensemble','owned_detector_logit_targets','owned_detector_pu'):
        runtime[name+'.py']=(ROOT/f'research/{name}.py').read_text()
    runtime['spotiflow_biohub/__init__.py']=''
    runtime['spotiflow_biohub/pu_targets.py']=(ROOT/'research/spotiflow_biohub/pu_targets.py').read_text()
    runtime['tests/test_owned_detector_ensemble.py']=(ROOT/'tests/test_owned_detector_ensemble.py').read_text().replace(
        "str(Path(__file__).resolve().parents[1]/'research')","str(Path(__file__).resolve().parents[1])")
    probe=ROOT/'.biohub/cache/kernel-outputs/owned-detector-logit-probe-v1/owned_detector_logit_probe/outputs/result.json'
    if hashlib.sha256(probe.read_bytes()).hexdigest()!='889a94d0a7bb60d40bde3a42a1343072a33cc68a0680dd0d083c41814efaf2ba':
        raise ValueError('Frozen probe provenance changed')
    runtime['owned_probe.json']=probe.read_bytes().decode('utf-8')
    nb['cells'][1]['source']=source.replace(ast.get_source_segment(source,node),'runtime_sources = '+repr(runtime)).splitlines(keepends=True)
    for index in (0,len(nb['cells'])-1):
        source=''.join(nb['cells'][index]['source']).replace('detector-spatial-tta-probe-v1',RUN)
        source=source.replace('detector_spatial_tta_probe','owned_detector_ensemble_probe')
        if index==len(nb['cells'])-1:
            locator="""secondary = next((Path('/kaggle/input')/p/'owned_detector_fit_pair/outputs/pu/last.pt'
    for p in ('biohub-owned-detector-fit-pair-v1','notebooks/indarkarhana/biohub-owned-detector-fit-pair-v1',
              'kernels/indarkarhana/biohub-owned-detector-fit-pair-v1')
    if (Path('/kaggle/input')/p/'owned_detector_fit_pair/outputs/pu/last.pt').is_file()),None)
if secondary is None: raise RuntimeError('Frozen owned PU input missing')
"""
            source=source.replace('command = [',locator+'command = [')
            source=source.replace("'--checkpoint', str(checkpoint)","'--secondary', str(secondary), '--checkpoint', str(checkpoint)")
        nb['cells'][index]['source']=source.splitlines(keepends=True)
    nb['metadata']['codex']=dict(run_id=RUN,declared_budget_seconds=3600,training_only=True,
        frames=3,weights=[.5,.5],detector_views_per_model=8,selection_opened=False,
        target_audit_opened=False,authorized_for_submission=False)
    meta.update(id='indarkarhana/'+TARGET.name,title='Biohub Owned Detector Ensemble Probe v1',code_file=TARGET.name+'.ipynb',
        kernel_sources=['indarkarhana/biohub-image-motion-linker-v1/2','indarkarhana/biohub-owned-detector-fit-pair-v1/1'])
    for cell in nb['cells']: ast.parse(''.join(cell['source']))
    return nb,meta


if __name__=='__main__':
    if TARGET.exists(): raise ValueError('Refuse to overwrite ensemble staging')
    nb,meta=build(); TARGET.mkdir(parents=True)
    (TARGET/meta['code_file']).write_text(json.dumps(nb),encoding='utf-8')
    (TARGET/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(TARGET)
