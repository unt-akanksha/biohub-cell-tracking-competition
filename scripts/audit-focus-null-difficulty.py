"""Fitting-only physical difficulty audit; no new model or threshold selection."""
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.special import logsumexp

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.focus_parent_dropout import augment
RUN='focus-null-difficulty-v1'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def measure(packet,columns,parameters):
    source=np.asarray(packet['source_coords'],float)
    target=np.asarray(packet['target_coords'],float)[columns]
    flow=np.asarray(packet['backward_um'],float)[columns]
    delta=(source[:,None]-target[None])*[1.625,.40625,.40625]-flow[None]-parameters['mean_um']
    scores=-.5*(delta**2/parameters['variance_um2']).sum(-1)
    if not len(source):raise ValueError('This fitting audit requires nonempty candidate sets')
    margin=scores.max(0)+4.5
    loss=np.logaddexp(logsumexp(scores,axis=0),-4.5)+4.5
    return [dict(column=int(c),best_parent_minus_null=float(m),null_nll=float(l)) for c,m,l in zip(columns,margin,loss)]


def summary(rows):
    margin=np.array([r['best_parent_minus_null'] for r in rows]);loss=np.array([r['null_nll'] for r in rows])
    if not len(rows) or not np.isfinite(margin).all() or not np.isfinite(loss).all():raise ValueError('Finite nonempty null evidence required')
    return dict(examples=len(rows),physical_correct_null=int((margin<0).sum()),physical_correct_fraction=float((margin<0).mean()),
        mean_null_nll=float(loss.mean()),median_best_parent_minus_null=float(np.median(margin)),
        parent_margin_quantiles=dict(zip(['q0','q25','q50','q75','q100'],np.quantile(margin,[0,.25,.5,.75,1]).tolist())))


def main():
    started=time.monotonic();target=ROOT/f'reports/experiments/{RUN}-result.json'
    if target.exists():raise ValueError('Never overwrite an actual difficulty audit')
    receipt=ROOT/'reports/experiments/focus-parent-dropout-training-v1-result.json'
    if sha(receipt)!='854d43703018ff135cfe464c5597903df484fa9ac52d4fc3fc6eaa71f4c1c623':raise ValueError('Verified completed training result required')
    verified=json.loads(receipt.read_text());work=ROOT/'.biohub/cache/kernel-outputs/focus-parent-dropout-training-v1/focus_parent_dropout_training'
    sp=work/'runtime/training_spec.json'
    if sha(sp)!=verified['worker']['training_spec_sha256']:raise ValueError('Actual frozen experiment specification required')
    spec=json.loads(sp.read_text());audit_path=ROOT/'reports/experiments/focus-parent-dropout-v1-audit.json'
    if sha(audit_path)!=spec['dropout_audit_sha256']:raise ValueError('Original completed dropout audit required')
    audit=json.loads(audit_path.read_text());expected={(r['stem'],r['source_frame']):r for r in audit['records']}
    if sha(ROOT/'research/focus_parent_dropout.py')!=audit['source_hashes']['research/focus_parent_dropout.py']:raise ValueError('Frozen augmentation code changed')
    parameters=spec['motion_parameters'];records={'natural':[],'synthetic':[]};movies=[]
    roots=[ROOT/'.biohub/cache/kernel-outputs/focus-adaptation-features-v1/focus_adaptation_features/outputs',ROOT/'.biohub/cache/kernel-outputs/focus-extra-fit-features-v1/focus_extra_fit_features/outputs']
    for group,root in zip(spec['feature_groups'],roots):
        for record in group['feature_records']:
            if record['role']!='fitting':continue
            stem=record['stem'];mp=root/stem/'manifest.json'
            if sha(mp)!=record['manifest_sha256']:raise ValueError('Actual fitting manifest changed')
            manifest=json.loads(mp.read_text());local={'natural':[],'synthetic':[]}
            for pair in manifest['pairs']:
                if not pair['known_parent']+pair['known_absent']:continue
                path=root/stem/pair['file']
                if sha(path)!=pair['sha256']:raise ValueError('Actual fitting packet changed')
                with np.load(path,allow_pickle=False) as data:p={k:data[k].copy() for k in data.files}
                basecols=np.flatnonzero(p['labels']==len(p['source_indices']))
                if len(basecols):local['natural'].extend([dict(stem=stem,source_frame=int(p['source_frame']),**r) for r in measure(p,basecols,parameters)])
                changed=augment(dict(packet=p,stem=stem,role='fitting'))
                if changed is None:continue
                row=expected.pop((stem,int(p['source_frame'])))
                if row['provenance']!=changed['provenance'] or row['labels_sha256']!=hashlib.sha256(changed['packet']['labels'].tobytes()).hexdigest():raise ValueError('Exact original augmentation required')
                cols=np.asarray(changed['provenance']['synthetic_null_columns'],np.int64)
                if changed['provenance']['eligible_nonempty_source'] and len(cols):
                    local['synthetic'].extend([dict(stem=stem,source_frame=int(p['source_frame']),**r) for r in measure(changed['packet'],cols,parameters)])
            movies.append(dict(stem=stem,**{k:summary(v) if v else None for k,v in local.items()}))
            for k in records:records[k].extend(local[k])
            print(json.dumps(dict(stage='fitting_difficulty',**movies[-1])),flush=True)
    if expected or [r['stem'] for r in movies]!=audit['training_stems'] or len(records['natural'])!=161 or len(records['synthetic'])!=1136:raise ValueError('Complete fitting-only null inventory required')
    result=dict(status='completed_fitting_null_difficulty_audit',run_id=RUN,summary={k:summary(v) for k,v in records.items()},
        per_movie=movies,records=records,training_result_sha256=sha(receipt),audit_sha256=sha(audit_path),
        source_sha256=sha(Path(__file__)),diagnostic_movies_opened=0,source_movies_opened=0,new_target_movies_opened=0,
        gpu_seconds=0,thresholds_changed=False,authorized_for_submission=False,elapsed_seconds=time.monotonic()-started,
        caveat='Physical prior on fitting examples only; not the neural score and not independent model-quality evidence.')
    target.write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result['summary'],indent=2))


if __name__=='__main__':main()
