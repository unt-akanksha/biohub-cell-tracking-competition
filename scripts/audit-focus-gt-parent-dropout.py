"""Recover exact training GT geometry, replay labels and audit realistic dropout."""
import hashlib
import json
from pathlib import Path
import runpy
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.focus_gt_parent_dropout import augment
from research.focus_adaptation_labels import node_matches
from research.focus_predicted_training_labels import targets
RUN='focus-gt-parent-dropout-v1'
FILES=['scripts/audit-focus-gt-parent-dropout.py','research/focus_gt_parent_dropout.py',
       'research/focus_parent_dropout.py','research/focus_adaptation_labels.py','research/focus_predicted_training_labels.py',
       f'reports/experiments/{RUN}-design.md']


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import tracksdata as td
    started=time.monotonic();out=ROOT/'.biohub/cache'/RUN;target=ROOT/f'reports/experiments/{RUN}-audit.json'
    if out.exists() or target.exists():raise ValueError('Never overwrite any partial or completed audit')
    frozen={p:sha(ROOT/p) for p in FILES}
    difficulty_path=ROOT/'reports/experiments/focus-null-difficulty-v1-result.json'
    if sha(difficulty_path)!='686ba7bc396c23740f1bf273c243b3d22d11dc22cdac22ad33a182d0248a433a':raise ValueError('Actual fitting difficulty evidence required')
    previous=json.loads(difficulty_path.read_text());difficulty=runpy.run_path(str(ROOT/'scripts/audit-focus-null-difficulty.py'))
    if sha(ROOT/'scripts/audit-focus-null-difficulty.py')!=previous['source_sha256']:raise ValueError('Prior difficulty formula changed')
    receipt=ROOT/'reports/experiments/focus-parent-dropout-training-v1-result.json'
    if sha(receipt)!=previous['training_result_sha256']:raise ValueError('Verified prior training required')
    proof=json.loads(receipt.read_text());sp=ROOT/'.biohub/cache/kernel-outputs/focus-parent-dropout-training-v1/focus_parent_dropout_training/runtime/training_spec.json'
    if sha(sp)!=proof['worker']['training_spec_sha256']:raise ValueError('Actual original specification changed')
    spec=json.loads(sp.read_text());oldpath=ROOT/'reports/experiments/focus-parent-dropout-v1-audit.json'
    if sha(oldpath)!=spec['dropout_audit_sha256']:raise ValueError('Previous full dropout audit required')
    old=json.loads(oldpath.read_text());prior={(r['stem'],r['source_frame']):r for r in old['records']}
    runpy.run_path(str(ROOT/'scripts/score-independent-selection.py'))['load_scorer'](ROOT/'.biohub/vendor/kaggle-cell-tracking-competition/src/tracking_cellmot')
    roots=[ROOT/'.biohub/cache/kernel-outputs/focus-adaptation-features-v1/focus_adaptation_features/outputs',ROOT/'.biohub/cache/kernel-outputs/focus-extra-fit-features-v1/focus_extra_fit_features/outputs']
    out.mkdir();records=[];movie_rows=[];hardness=[];parents=0;natural=0;synthetic=0;eligible=0
    for group,root in zip(spec['feature_groups'],roots):
        for record in group['feature_records']:
            if record['role']!='fitting':continue
            stem=record['stem']
            if stem not in old['training_stems'] or stem in spec['contract']['diagnostic_stems']:raise ValueError('Fitting GT access only')
            raw=root/'raw_detections'/(stem+'.npz');mp=root/stem/'manifest.json'
            if sha(raw)!=record['raw_sha256'] or sha(mp)!=record['manifest_sha256']:raise ValueError('Verified raw/manifest changed')
            with np.load(raw,allow_pickle=False) as data:coords=data['coords'].copy()
            truth=td.graph.IndexedRXGraph.from_geff(str(ROOT/'.biohub/cache/competition-train-geffs-packed-v1/train'/(stem+'.geff')))[0]
            mapping=node_matches(coords,truth);keys=td.DEFAULT_ATTR_KEYS
            nodes={int(r[0]):list(r[1:]) for r in truth.node_attrs().select(keys.NODE_ID,'t','z','y','x').iter_rows()}
            edges=list(truth.edge_attrs().select(keys.EDGE_SOURCE,keys.EDGE_TARGET).iter_rows())
            local=[];label_replays=0
            for pair in json.loads(mp.read_text())['pairs']:
                path=root/stem/pair['file']
                if sha(path)!=pair['sha256']:raise ValueError('Frozen fitting packet changed')
                with np.load(path,allow_pickle=False) as data:p={k:data[k].copy() for k in data.files}
                expected=targets(coords,mapping,nodes,edges,int(p['source_frame']))
                if any(not np.array_equal(p[k],expected[k]) for k in ('labels','source_indices','target_indices')):raise ValueError('Every original label must replay against fresh GT matching')
                label_replays+=1;ns=len(p['source_indices']);known=np.unique(p['labels'][(p['labels']>=0)&(p['labels']<ns)])
                centers={str(int(i)):nodes[int(mapping[int(p['source_indices'][i])])][1:] for i in known}
                changed=augment(dict(packet=p,stem=stem,role='fitting'),centers)
                if changed is None:continue
                previous_row=prior.pop((stem,int(p['source_frame'])))
                if changed['provenance']['selected_parent_row']!=previous_row['provenance']['selected_parent_row']:raise ValueError('Parent identity selection changed')
                actual=changed['packet'];new_ns=len(actual['source_indices']);columns=changed['provenance']['synthetic_null_columns']
                for c in columns:
                    center=np.asarray(centers[str(int(p['labels'][c]))],float)
                    if np.any(np.linalg.norm((actual['source_coords'].astype(float)-center)*[1.625,.40625,.40625],axis=1)<=7.):raise ValueError('A newly null parent remains geometrically representable')
                if any(not np.array_equal(actual[k],p[k]) for k in p if k not in ('source_indices','source_coords','source_features','source_pos','labels')):raise ValueError('Target arrays must remain exact')
                row=dict(stem=stem,source_frame=int(p['source_frame']),packet_sha256=pair['sha256'],parent_gt_coords=centers,
                    provenance=changed['provenance'],labels_sha256=hashlib.sha256(actual['labels'].tobytes()).hexdigest(),
                    source_ids_sha256=hashlib.sha256(actual['source_indices'].tobytes()).hexdigest(),
                    retained_known_parent=int(((actual['labels']>=0)&(actual['labels']<new_ns)).sum()),
                    original_known_null=int((p['labels']==ns).sum()),total_known_null=int((actual['labels']==new_ns).sum()))
                local.append(row);records.append(row)
                if changed['provenance']['eligible_nonempty_source']:
                    eligible+=1;parents+=row['retained_known_parent'];natural+=row['original_known_null'];synthetic+=len(columns)
                    hardness.extend([dict(stem=stem,source_frame=row['source_frame'],**r) for r in difficulty['measure'](actual,np.asarray(columns,np.int64),spec['motion_parameters'])])
            if label_replays!=99:raise ValueError('Complete original movie label replay required')
            path=out/(stem+'.json');path.write_text(json.dumps(local,indent=2,allow_nan=False))
            movie_rows.append(dict(stem=stem,records=len(local),label_replays=label_replays,sha256=sha(path),raw_sha256=sha(raw)))
            print(json.dumps(dict(stage='exact_gt_training_movie',**movie_rows[-1])),flush=True)
    if prior or [r['stem'] for r in movie_rows]!=old['training_stems']:raise ValueError('All and only original fitting pairs required')
    measured=difficulty['summary'](hardness);before=previous['summary']['synthetic']
    conditions=dict(eligible_pairs_over100=eligible>100,
        more_wrong_parent_cases=measured['examples']-measured['physical_correct_null']>before['examples']-before['physical_correct_null'],
        higher_mean_null_nll=measured['mean_null_nll']>before['mean_null_nll'])
    if frozen!={p:sha(ROOT/p) for p in FILES}:raise ValueError('Frozen audit design changed')
    result=dict(status='completed_exact_gt_parent_dropout_audit',run_id=RUN,training_stems=old['training_stems'],source_hashes=frozen,
        previous_difficulty_sha256=sha(difficulty_path),prior_dropout_audit_sha256=sha(oldpath),per_movie=movie_rows,
        eligible_pairs=eligible,eligible_retained_parent_labels=parents,eligible_natural_nulls=natural,eligible_synthetic_nulls=synthetic,
        synthetic_difficulty=measured,previous_synthetic_difficulty=before,feasibility_conditions=conditions,
        eligible_for_small_training_smoke=all(conditions.values()),original_label_replays=12*99,
        diagnostic_movies_opened=0,source_movies_opened=0,new_target_movies_opened=0,gpu_seconds=0,
        authorized_for_submission=False,elapsed_seconds=time.monotonic()-started)
    target.write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps({k:v for k,v in result.items() if k not in ('source_hashes','per_movie')},indent=2))


if __name__=='__main__':main()
