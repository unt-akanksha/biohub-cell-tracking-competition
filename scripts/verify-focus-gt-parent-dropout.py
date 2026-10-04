"""Replay every serialized exact-GT dropout against the actual fitting packet."""
import hashlib
import json
from pathlib import Path
import runpy
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.focus_gt_parent_dropout import augment
RUN='focus-gt-parent-dropout-v1'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    report=ROOT/f'reports/experiments/{RUN}-audit.json'
    if sha(report)!='d27222f2c768105a149226b2b864fe3931a3f1ff5f4f4a415cd079c87c4afd2f':raise ValueError('Original completed exact-GT audit required')
    audit=json.loads(report.read_text())
    if any(sha(ROOT/p)!=v for p,v in audit['source_hashes'].items()):raise ValueError('Audited source code/design changed')
    previous_path=ROOT/'reports/experiments/focus-null-difficulty-v1-result.json'
    if sha(previous_path)!=audit['previous_difficulty_sha256']:raise ValueError('Original difficulty comparison required')
    prior=json.loads(previous_path.read_text());old_path=ROOT/'reports/experiments/focus-parent-dropout-v1-audit.json'
    if sha(old_path)!=audit['prior_dropout_audit_sha256']:raise ValueError('Original fitting partition required')
    old=json.loads(old_path.read_text())
    if audit['training_stems']!=old['training_stems'] or len(audit['per_movie'])!=12:raise ValueError('Exact twelve fitting movies required')
    sp=ROOT/'.biohub/cache/kernel-outputs/focus-parent-dropout-training-v1/focus_parent_dropout_training/runtime/training_spec.json'
    if sha(sp)!='811dc9187cb587cbc803a0514c92cadb47e41704ef473c5532c269cb5790a36e':raise ValueError('Actual original physical/feature specification required')
    spec=json.loads(sp.read_text());difficulty=runpy.run_path(str(ROOT/'scripts/audit-focus-null-difficulty.py'))
    if sha(ROOT/'scripts/audit-focus-null-difficulty.py')!=prior['source_sha256']:raise ValueError('Frozen difficulty formula required')
    roots=[ROOT/'.biohub/cache/kernel-outputs/focus-adaptation-features-v1/focus_adaptation_features/outputs',ROOT/'.biohub/cache/kernel-outputs/focus-extra-fit-features-v1/focus_extra_fit_features/outputs']
    locations={r['stem']:root for g,root in zip(spec['feature_groups'],roots) for r in g['feature_records'] if r['role']=='fitting'}
    records=[];hardness=[];counts=dict(eligible_pairs=0,eligible_retained_parent_labels=0,eligible_natural_nulls=0,eligible_synthetic_nulls=0)
    for movie in audit['per_movie']:
        stem=movie['stem'];path=ROOT/'.biohub/cache'/RUN/(stem+'.json')
        if stem not in locations or sha(path)!=movie['sha256'] or movie['label_replays']!=99:raise ValueError('Actual complete fitting sidecar required')
        local=json.loads(path.read_text())
        if len(local)!=movie['records']:raise ValueError('Sidecar row count changed')
        for row in local:
            pp=locations[stem]/stem/f"{row['source_frame']:03d}.npz"
            if row['stem']!=stem or sha(pp)!=row['packet_sha256']:raise ValueError('Actual original fitting pair required')
            with np.load(pp,allow_pickle=False) as data:packet={k:data[k].copy() for k in data.files}
            changed=augment(dict(packet=packet,stem=stem,role='fitting'),row['parent_gt_coords']);p=changed['packet']
            if changed['provenance']!=row['provenance'] or hashlib.sha256(p['labels'].tobytes()).hexdigest()!=row['labels_sha256'] or hashlib.sha256(p['source_indices'].tobytes()).hexdigest()!=row['source_ids_sha256']:raise ValueError('Exact restored augmentation replay required')
            ns=len(p['source_indices']);parent=int(((p['labels']>=0)&(p['labels']<ns)).sum());natural=int((packet['labels']==len(packet['source_indices'])).sum())
            if parent!=row['retained_known_parent'] or natural!=row['original_known_null'] or int((p['labels']==ns).sum())!=row['total_known_null']:raise ValueError('Actual transformed supervision counts changed')
            records.append(row)
            if changed['provenance']['eligible_nonempty_source']:
                cols=np.asarray(changed['provenance']['synthetic_null_columns'],np.int64)
                counts['eligible_pairs']+=1;counts['eligible_retained_parent_labels']+=parent;counts['eligible_natural_nulls']+=natural;counts['eligible_synthetic_nulls']+=len(cols)
                hardness.extend(difficulty['measure'](p,cols,spec['motion_parameters']))
    measured=difficulty['summary'](hardness);before=prior['summary']['synthetic']
    conditions=dict(eligible_pairs_over100=counts['eligible_pairs']>100,
        more_wrong_parent_cases=measured['examples']-measured['physical_correct_null']>before['examples']-before['physical_correct_null'],
        higher_mean_null_nll=measured['mean_null_nll']>before['mean_null_nll'])
    if any(audit[k]!=v for k,v in counts.items()) or measured!=audit['synthetic_difficulty'] or conditions!=audit['feasibility_conditions'] or not all(conditions.values()):raise ValueError('Complete counts/difficulty gate replay required')
    if audit['original_label_replays']!=1188 or any(audit[k]!=0 for k in ('diagnostic_movies_opened','source_movies_opened','new_target_movies_opened','gpu_seconds')) or audit['authorized_for_submission'] is not False:raise ValueError('Original label replay and fitting-only scope required')
    receipt=dict(status='verified_exact_gt_dropout_sidecars',audit_sha256=sha(report),exact_augmented_pair_replays=len(records),counts=counts,
        synthetic_difficulty=measured,eligible_for_small_training_smoke=True,authorized_for_submission=False)
    bundle=dict(records=records,training_stems=audit['training_stems'],**counts,exact_gt_audit_sha256=sha(report))
    return receipt,bundle


if __name__=='__main__':
    target=ROOT/f'reports/experiments/{RUN}-verification.json'
    if target.exists():raise ValueError('Never overwrite actual verification')
    receipt,_=verify();target.write_text(json.dumps(receipt,indent=2,allow_nan=False));print(json.dumps(receipt,indent=2))
