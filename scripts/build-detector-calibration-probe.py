"""Prepare, but do not launch, the bounded training-confidence collection."""
import ast
import json
from pathlib import Path
import runpy

ROOT=Path(__file__).resolve().parents[1]
RUN='owned-detector-calibration-probe-v1'
TARGET=ROOT/'kaggle'/('biohub-'+RUN)


def build():
    nb,meta=runpy.run_path(str(ROOT/'scripts/build-detector-spatial-tta-selection.py'))['build']()
    source=''.join(nb['cells'][1]['source'])
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='runtime_sources')
    runtime=ast.literal_eval(node.value)
    runtime['run_selection.py']=(ROOT/'scripts/collect-detector-calibration-probe.py').read_text()
    runtime['detector_calibration_records.py']=(ROOT/'research/detector_calibration_records.py').read_text()
    nb['cells'][1]['source']=source.replace(ast.get_source_segment(source,node),'runtime_sources = '+repr(runtime)).splitlines(keepends=True)
    for index in (0,len(nb['cells'])-1):
        source=''.join(nb['cells'][index]['source']).replace('detector-spatial-tta-selection-v1',RUN)
        source=source.replace('detector_spatial_tta_selection','owned_detector_calibration_probe')
        if index==len(nb['cells'])-1:
            start=source.index('command ='); end=source.index('process = subprocess.Popen')
            source=source[:start]+'''candidate_paths = [Path('/kaggle/input')/p/'owned_detector_fit_pair/outputs/sparse/last.pt'
    for p in ('biohub-owned-detector-fit-pair-v1','notebooks/indarkarhana/biohub-owned-detector-fit-pair-v1',
              'kernels/indarkarhana/biohub-owned-detector-fit-pair-v1')]
candidate = next((p for p in candidate_paths if p.is_file()),None)
if candidate is None: raise RuntimeError('Completed sparse-control checkpoint not mounted')
command = [sys.executable,'-u',str(runtime/'run_selection.py'),
    '--repo',str(repo),'--runtime',str(runtime),'--manifest',str(runtime/'split.json'),
    '--data',str(data),'--output',str(work/'outputs'),'--parent',str(checkpoint),'--candidate',str(candidate)]
import signal
'''+source[end:]
        nb['cells'][index]['source']=source.splitlines(keepends=True)
    nb['metadata']['codex']=dict(run_id=RUN,declared_budget_seconds=3600,training_only=True,
        probe_movies=6,frames_per_movie=3,calibration_fitted=False,selection_opened=False,
        target_audit_opened=False,authorized_for_submission=False)
    meta.update(id='indarkarhana/'+TARGET.name,title='Biohub Owned Detector Calibration Probe v1',code_file=TARGET.name+'.ipynb',
        kernel_sources=['indarkarhana/biohub-image-motion-linker-v1/2','indarkarhana/biohub-owned-detector-fit-pair-v1/1'])
    for cell in nb['cells']: ast.parse(''.join(cell['source']))
    return nb,meta


if __name__=='__main__':
    if TARGET.exists(): raise ValueError('Refuse to overwrite calibration staging')
    nb,meta=build(); TARGET.mkdir(parents=True)
    (TARGET/meta['code_file']).write_text(json.dumps(nb),encoding='utf-8')
    (TARGET/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(TARGET)
