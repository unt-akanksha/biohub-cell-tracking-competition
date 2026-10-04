"""Extend the immutable successful collector to the declared120 training movies."""
import ast
import hashlib
import json
from pathlib import Path
import runpy

ROOT=Path(__file__).resolve().parents[1]
RUN='owned-detector-calibration-full-v1'
TARGET=ROOT/'kaggle'/('biohub-'+RUN)
PROBE_NB_SHA='d3ba9d43f0daeedd3eef086c63665ae3b525bca694173536e8a31630a32021d1'


def full_runtime(source):
    replacements={
        'Only six training movies:':'Full120 training movies:',
        'Run only in a separately staged small GPU probe after CPU matching tests pass.':'Requires the completed verified small probe; no model updates.',
        'groups=calibration_scope(split,probe=True)':'groups=calibration_scope(split,probe=False)',
        'records=[]; started=time.monotonic()':'records=[]; replay_count=0; started=time.monotonic()',
        "from image_motion_residual import flow_hash":"from image_motion_residual import flow_hash\n    from detector_calibration_full_contract import verify_probe,verify_replay_row,PROBE_SHA\n    replay=verify_probe((args.runtime/'frozen_calibration_probe_report.json').read_bytes())",
        'records.append(row); print(json.dumps(row),flush=True)':'replay_count += int(verify_replay_row(row,replay))\n                records.append(row); print(json.dumps(row),flush=True)',
        'if len(records)!=18 or':'if len(records)!=360 or replay_count!=18 or',
        'Exactly18 frame records':'Exactly360 frame records and18 probe replays',
        'completed_training_calibration_collection_probe':'completed_training_calibration_collection_full',
        'diagnostic_only=True,calibration_fitted=False,':'diagnostic_only=True,calibration_fitted=False,probe_report_sha256=PROBE_SHA,replayed_frames=replay_count,',
        'Bounded six-movie probe exceeded40minutes':'Bounded full training collection exceeded40minutes',
        'Six neural-training movies only; functionality records, not independent validation or full calibration':'120 neural-training movies; confidence collection only, not independent validation or submission',
    }
    for old,new in replacements.items():
        if source.count(old)!=1: raise ValueError('Immutable collector marker changed: '+old)
        source=source.replace(old,new)
    ast.parse(source)
    return source


def build():
    folder=ROOT/'kaggle/biohub-owned-detector-calibration-probe-v1'
    path=folder/'biohub-owned-detector-calibration-probe-v1.ipynb'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=PROBE_NB_SHA: raise ValueError('Successful probe notebook changed')
    gate=runpy.run_path(str(ROOT/'research/detector_calibration_full_contract.py'))
    report=(ROOT/'reports/experiments/detector-calibration-probe-v1-result.json').read_bytes()
    gate['verify_probe'](report)
    nb=json.loads(path.read_text()); meta=json.loads((folder/'kernel-metadata.json').read_text())
    source=''.join(nb['cells'][1]['source'])
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='runtime_sources')
    runtime=ast.literal_eval(node.value)
    runtime['run_selection.py']=full_runtime(runtime['run_selection.py'])
    runtime['detector_calibration_full_contract.py']=(ROOT/'research/detector_calibration_full_contract.py').read_text()
    runtime['frozen_calibration_probe_report.json']=report.decode()
    nb['cells'][1]['source']=source.replace(ast.get_source_segment(source,node),'runtime_sources = '+repr(runtime)).splitlines(keepends=True)
    for index in (0,len(nb['cells'])-1):
        source=''.join(nb['cells'][index]['source']).replace('owned-detector-calibration-probe-v1',RUN)
        source=source.replace('owned_detector_calibration_probe','owned_detector_calibration_full')
        nb['cells'][index]['source']=source.splitlines(keepends=True)
    nb['metadata']['codex'].pop('probe_movies')
    nb['metadata']['codex'].update(run_id=RUN,training_movies=120,fitting_movies=96,diagnostic_movies=24,
        probe_report_sha256=gate['PROBE_SHA'])
    meta.update(id='indarkarhana/'+TARGET.name,title='Biohub Owned Detector Calibration Full v1',code_file=TARGET.name+'.ipynb')
    for cell in nb['cells']: ast.parse(''.join(cell['source']))
    return nb,meta


if __name__=='__main__':
    if TARGET.exists(): raise ValueError('Refuse to overwrite full collector staging')
    nb,meta=build(); TARGET.mkdir(parents=True)
    (TARGET/meta['code_file']).write_text(json.dumps(nb),encoding='utf-8')
    (TARGET/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(TARGET)
