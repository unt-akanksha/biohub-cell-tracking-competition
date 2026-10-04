"""Verify small training collection and exact rematched recall, never deploy."""
import hashlib
import argparse
import json
from pathlib import Path
import runpy
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.detector_calibration_records import calibration_scope,annotated_confidences,temporal_context,select_frames
from research.detector_recall_calibration import fit_threshold

NB_SHA='d3ba9d43f0daeedd3eef086c63665ae3b525bca694173536e8a31630a32021d1'
FULL_NB_SHA='a96c62e370344eb84f37f5bea4b9623112df773cfa06c0371e09e44d51fa2fc9'


def evaluate_records(rows, cutoff):
    fitting=[]; matched=[]; details=[]
    for row,payload in rows:
        truth=payload['truth_coords']
        parent=annotated_confidences(payload['parent_coords'],payload['parent_probabilities'],truth)>0
        candidate=annotated_confidences(payload['candidate_coords'],payload['candidate_probabilities'],truth)
        if row['group']=='fitting': fitting.extend(candidate); matched.extend(parent)
        elif row['group']!='diagnostic': raise ValueError('Unregistered training calibration group')
        details.append((row,payload,parent,candidate))
    fit=fit_threshold(np.asarray(matched,dtype=bool),fitting,baseline_threshold=cutoff)
    threshold=fit['threshold']; report=[]
    for row,payload,parent,candidate in details:
        exact=None
        if threshold is not None:
            keep=payload['candidate_probabilities']>np.float32(threshold)
            exact=int(np.sum(annotated_confidences(payload['candidate_coords'][keep],
                payload['candidate_probabilities'][keep],payload['truth_coords'])>0))
        report.append(dict(stem=row['stem'],t=row['t'],group=row['group'],annotations=len(parent),
            parent_matched=int(parent.sum()),candidate_baseline_matched=int(np.sum(candidate>0)),
            candidate_calibrated_matched=exact))
    per_movie=[]
    for stem in dict.fromkeys(r['stem'] for r in report):
        movie=[r for r in report if r['stem']==stem]; n=sum(r['annotations'] for r in movie)
        a=sum(r['parent_matched'] for r in movie)
        b=None if threshold is None else sum(r['candidate_calibrated_matched'] for r in movie)
        per_movie.append(dict(stem=stem,group=movie[0]['group'],annotations=n,parent_matched=a,
            candidate_calibrated_matched=b,recall_delta=None if b is None else (b-a)/n,
            recall_preserved=b is not None and b>=a-int(np.floor(.005*n))))
    exact_fit=None if threshold is None else sum(r['candidate_calibrated_matched'] for r in report if r['group']=='fitting')
    if exact_fit is not None and exact_fit<fit['required_matched']:
        raise ValueError('Exact rematching invalidated the fitted recall constraint')
    diagnostic=[r for r in per_movie if r['group']=='diagnostic']
    return dict(fit=fit,per_frame=report,per_movie=per_movie,
        probe_diagnostic_recall_preserved=bool(diagnostic) and all(r['recall_preserved'] for r in diagnostic),
        authorized_for_submission=False,independent_validation=False,full_calibration_completed=False)


def main(full=False):
    import tracksdata as td
    import polars as pl
    kind='full' if full else 'probe'; expected_frames=360 if full else 18
    run_id=f'owned-detector-calibration-{kind}-v1'
    base=ROOT/f'.biohub/cache/kernel-outputs/{run_id}/owned_detector_calibration_{kind}'
    nb=ROOT/f'kaggle/biohub-{run_id}/biohub-{run_id}.ipynb'
    expected_nb=FULL_NB_SHA if full else NB_SHA
    if hashlib.sha256(nb.read_bytes()).hexdigest()!=expected_nb: raise ValueError('Frozen collector notebook changed')
    pilot=runpy.run_path(str(ROOT/'scripts/verify-independent-real-pilot.py'))
    bundles=pilot['embedded_sources'](nb)
    for key,file in [('sources','source_hashes.json'),('runtime_sources','runtime_hashes.json')]:
        expected={p:hashlib.sha256(s.encode()).hexdigest() for p,s in bundles[key].items()}
        if json.loads((base/file).read_text())!=expected: raise ValueError('Collector source receipt changed')
    result=json.loads((base/'outputs/collection_result.json').read_text())
    terminal=json.loads((base/'launcher_terminal.json').read_text())
    collector=runpy.run_path(str(ROOT/'scripts/collect-detector-calibration-probe.py'))
    split_bytes=(ROOT/'research/independent_real_baseline_v1_split.json').read_bytes()
    if hashlib.sha256(split_bytes).hexdigest()!=collector['SPLIT_SHA']: raise ValueError('Original split changed')
    split=json.loads(split_bytes)
    groups=calibration_scope(split,probe=not full)
    if (terminal['status']!='completed' or terminal['run_id']!=run_id
        or not 0<terminal['elapsed_seconds']<=3600
        or result['status']!=f'completed_training_calibration_collection_{kind}'
        or result['groups']!=groups or len(result['records'])!=expected_frames
        or result['parent_sha256']!=collector['PARENT_SHA'] or result['candidate_sha256']!=collector['SPARSE_SHA']
        or result['split_sha256']!=collector['SPLIT_SHA'] or result['models_unchanged'] is not True
        or result['precision']!='FP32 exact inference'
        or any(result[k] is not False for k in ('selection_opened','target_audit_opened','authorized_for_submission','calibration_fitted'))):
        raise ValueError('Successful exact small training collector required')
    replay={}; replay_count=0; maximum_replay_probability_delta=0.
    if full:
        from research.detector_calibration_full_contract import verify_probe,verify_replay_row,PROBE_SHA
        replay=verify_probe((ROOT/'reports/experiments/detector-calibration-probe-v1-result.json').read_bytes())
        if result.get('probe_report_sha256')!=PROBE_SHA or result.get('replayed_frames')!=18:
            raise ValueError('Full collector lacks all18 small-probe input replays')
    rows=[]; seen=set()
    for group,stems in groups.items():
        for stem in stems:
            truth=td.graph.IndexedRXGraph.from_geff(str(ROOT/'.biohub/cache/competition-train-geffs-packed-v1/train'/(stem+'.geff')))[0]
            attrs=truth.node_attrs(attr_keys=['node_id','t','z','y','x'])
            selected=select_frames(stem,attrs['t'].to_list(),100)
            records=[r for r in result['records'] if r['stem']==stem]
            if [r['t'] for r in records]!=selected: raise ValueError('Frame choice changed from deterministic training rule')
            for row in records:
                if (row['group']!=group or row['context']!=list(temporal_context(row['t'],100)[0])
                    or row['artifact']!=f"{stem}_{row['t']:03d}.npz" or row['artifact'] in seen):
                    raise ValueError('Invalid or duplicate training-frame artifact')
                seen.add(row['artifact']); path=base/'outputs'/row['artifact']
                if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']: raise ValueError('Frame artifact changed')
                with np.load(path,allow_pickle=False) as payload: values={k:payload[k] for k in payload.files}
                if full and verify_replay_row(row,replay):
                    replay_count+=1
                    previous=ROOT/'.biohub/cache/kernel-outputs/owned-detector-calibration-probe-v1/owned_detector_calibration_probe/outputs'/row['artifact']
                    if hashlib.sha256(previous.read_bytes()).hexdigest()!=replay[(stem,row['t'])]['sha256']:
                        raise ValueError('Original small-probe artifact changed')
                    with np.load(previous,allow_pickle=False) as old:
                        for key,value in values.items():
                            if key.endswith('_probabilities'):
                                delta=float(np.max(np.abs(value-old[key]),initial=0.))
                                maximum_replay_probability_delta=max(maximum_replay_probability_delta,delta)
                                if delta>1e-6: raise ValueError('Material probability drift in probe replay')
                            elif not np.array_equal(value,old[key]):
                                raise ValueError('Coordinates or annotation identity changed in probe replay')
                annotations=attrs.filter(pl.col('t')==row['t']).sort('node_id')
                if (not np.array_equal(values['truth_ids'],annotations['node_id'].to_numpy())
                    or not np.array_equal(values['truth_coords'],annotations.select('t','z','y','x').to_numpy())
                    or row['annotation_count']!=len(annotations)):
                    raise ValueError('Missing or changed original training annotations')
                for name in ('parent','candidate'):
                    probs=values[name+'_probabilities']; coords=values[name+'_coords']
                    if (len(coords)!=row[name+'_peaks'] or probs.dtype!=np.float32
                        or np.any(probs<=np.float32(result['baseline_threshold']))
                        or np.any(coords[:,0]!=row['t'])):
                        raise ValueError('Peak confidence or original-frame scope changed')
                rows.append((row,values))
    if len(rows)!=expected_frames or full and replay_count!=18:
        raise ValueError('Complete declared collection and probe replay required')
    report=evaluate_records(rows,result['baseline_threshold'])
    report.update(status=f'verified_training_calibration_collection_{kind}',collection=result,
        notebook_sha256=expected_nb,selection_opened=False,target_audit_opened=False)
    if full:
        report.update(full_calibration_completed=True,
            full_diagnostic_recall_preserved=report.pop('probe_diagnostic_recall_preserved'),
            replayed_frames=replay_count,maximum_replay_probability_delta=maximum_replay_probability_delta)
    (ROOT/f'reports/experiments/detector-calibration-{kind}-v1-result.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('collection','per_frame')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--full',action='store_true')
    main(parser.parse_args().full)
