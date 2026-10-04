"""Audit all fitting-only counterfactual labels before any GPU training."""
import hashlib
import json
from pathlib import Path
import runpy
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.focus_cached_pair import validate_pair
from research.focus_parent_dropout import augment
RUN='focus-parent-dropout-v1'
FILES=['scripts/audit-focus-parent-dropout.py','research/focus_parent_dropout.py',
       'research/focus_cached_pair.py',f'reports/experiments/{RUN}-design.md']


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    started=time.monotonic();target=ROOT/f'reports/experiments/{RUN}-audit.json'
    if target.exists():raise ValueError('Never overwrite an actual audit')
    frozen={p:sha(ROOT/p) for p in FILES}
    spec=runpy.run_path(str(ROOT/'scripts/build-focus-expanded-summary.py'))['make_spec']()
    roots=[ROOT/'.biohub/cache/kernel-outputs/focus-adaptation-features-v1/focus_adaptation_features/outputs',
           ROOT/'.biohub/cache/kernel-outputs/focus-extra-fit-features-v1/focus_extra_fit_features/outputs']
    movies=[];records=[];eligible=0;synthetic=0;natural=0;parents=0;seen=[]
    for group,root in zip(spec['feature_groups'],roots):
        for record in group['feature_records']:
            if record['role']!='fitting':continue
            stem=record['stem'];seen.append(stem)
            raw=root/'raw_detections'/(stem+'.npz');mp=root/stem/'manifest.json'
            if sha(raw)!=record['raw_sha256'] or sha(mp)!=record['manifest_sha256']:raise ValueError('Frozen fitting geometry/manifest changed')
            with np.load(raw,allow_pickle=False) as data:coords=data['coords'].copy()
            manifest=json.loads(mp.read_text());local=[]
            for pair in manifest['pairs']:
                path=root/stem/pair['file']
                if sha(path)!=pair['sha256']:raise ValueError('Frozen fitting packet changed')
                with np.load(path,allow_pickle=False) as data:packet={k:data[k].copy() for k in data.files}
                counts=validate_pair(packet,coords)
                if any(counts[k]!=pair[k] for k in ('source_nodes','target_nodes','known_parent','known_absent','unknown')):raise ValueError('Original label inventory changed')
                result=augment(dict(packet=packet,stem=stem,role='fitting'))
                if result is None:continue
                new=result['packet'];proof=result['provenance'];ns=proof['source_nodes_after']
                # Original packet must remain bitwise unchanged after augmentation.
                with np.load(path,allow_pickle=False) as data:
                    if any(not np.array_equal(packet[k],data[k]) for k in data.files):raise ValueError('Augmentation mutated input')
                row=dict(stem=stem,source_frame=int(packet['source_frame']),packet_sha256=pair['sha256'],
                         provenance=proof,labels_sha256=hashlib.sha256(new['labels'].tobytes()).hexdigest(),
                         source_ids_sha256=hashlib.sha256(new['source_indices'].tobytes()).hexdigest(),
                         retained_known_parent=int(((new['labels']>=0)&(new['labels']<ns)).sum()),
                         total_known_null=int((new['labels']==ns).sum()),original_known_null=counts['known_absent'])
                local.append(row);records.append(row)
                if proof['eligible_nonempty_source']:
                    eligible+=1;synthetic+=len(proof['synthetic_null_columns']);natural+=counts['known_absent'];parents+=row['retained_known_parent']
            movies.append(dict(stem=stem,pairs_with_known_parent=len(local),
                               eligible_pairs=sum(r['provenance']['eligible_nonempty_source'] for r in local),
                               synthetic_nulls=sum(len(r['provenance']['synthetic_null_columns']) for r in local if r['provenance']['eligible_nonempty_source'])))
            print(json.dumps(dict(stage='fitting_movie_audited',**movies[-1])),flush=True)
    if seen!=spec['contract']['fitting_stems'] or len(seen)!=12 or set(seen)&set(spec['contract']['unchanged_diagnostic_stems']):raise ValueError('Exact twelve fitting movies only')
    if frozen!={p:sha(ROOT/p) for p in FILES}:raise ValueError('Frozen augmentation design/code changed')
    result=dict(status='completed_fitting_parent_dropout_audit',source_hashes=frozen,training_stems=seen,
        feature_spec=spec,per_movie=movies,records=records,eligible_pairs=eligible,eligible_synthetic_nulls=synthetic,
        eligible_natural_nulls=natural,eligible_retained_parent_labels=parents,
        eligible_for_small_training_smoke=eligible>=100 and synthetic>=100,
        original_packets_unchanged=True,diagnostic_samples_augmented=0,source_movies_opened=0,new_target_movies_opened=0,
        gpu_seconds=0,authorized_for_submission=False,elapsed_seconds=time.monotonic()-started)
    target.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in result.items() if k not in ('feature_spec','records','source_hashes','per_movie')}),flush=True)


if __name__=='__main__':main()
