"""Summarize immutable source-audit output and reconcile prior retained triplets."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root=ROOT/'.biohub/cache/native-division-proposal-audit-v4-full-output';path=root/'RESULT.json'
    result=json.loads(path.read_text())
    if result['status']!='proposal_audit_complete' or result['selection_opened'] or result['target_pilot_labels_used']:
        raise ValueError('Incomplete or ineligible source audit')
    for artifact in result['point_artifacts']:
        if sha(root/artifact['path'])!=artifact['sha256']:raise ValueError('Image proposal cache changed')
    merged=ROOT/'.biohub/cache/native-division-v3-data-bundle';data_path=merged/'DATA.json'
    if sha(data_path)!='7e0e7b1ca3c11b27319824fad3746b49f8a1cf519a73d3dad38e12d2ef8a0e06':raise ValueError('Prior data changed')
    prior=set()
    for r in json.loads(data_path.read_text())['records']:
        if r['role']!='optimization' or not r['positive']:continue
        if sha(merged/r['path'])!=r['sha256']:raise ValueError('Prior optimization packet changed')
        with np.load(merged/r['path'],allow_pickle=False) as packet:
            for parent in packet['truth_ids'][packet['labels']==1,0]:prior.add((r['stem'],r['transition'],int(parent)))
    paired={}
    for record in result['records']:
        for event in record['events']:
            key=(record['stem'],record['transition'],event['parent'])
            if record['variant'] in paired.setdefault(key,{}):raise ValueError('Duplicate event')
            paired[key][record['variant']]=event
    if len(paired)!=112 or any(set(v)!={'baseline','xy2'} for v in paired.values()):raise ValueError('Incomplete paired event set')
    if not prior.issubset(paired):raise ValueError('Prior positives outside frozen source inventory')
    comparison={}
    for embryo in ('44b6','6bba'):
        selected={k:v for k,v in paired.items() if k[0].startswith(embryo)}
        c=Counter(('both_eligible' if v['baseline']['eligible'] and v['xy2']['eligible'] else
                   'gained_xy2' if v['xy2']['eligible'] else 'lost_xy2' if v['baseline']['eligible'] else 'neither_eligible') for v in selected.values())
        c.update(prior_v3_positives=sum(k in prior for k in selected),
                 prior_v3_preserved_xy2=sum(k in prior and v['xy2']['eligible'] for k,v in selected.items()))
        comparison[embryo]=dict(c)
    frame_counts=dict(frames=len(result['frames']),baseline_points=sum(f['baseline_points'] for f in result['frames']),
                      xy2_points=sum(f['xy2_points'] for f in result['frames']),
                      xy2_additional_seconds=sum(f['xy2_after_normalization_seconds'] for f in result['frames']))
    frame_counts['point_count_increase_fraction']=frame_counts['xy2_points']/frame_counts['baseline_points']-1
    summary=dict(status='verified_source_proposal_audit',result_sha256=sha(path),totals=result['totals'],comparison=comparison,
                 frame_summary=frame_counts,seconds=result['seconds'],prior_optimization_positives=len(prior),
                 selection_opened=False,target_pilot_labels_used=False,authorized_for_submission=False)
    destination=ROOT/'reports/experiments/native-division-proposal-audit-v4-summary.json'
    destination.write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))


if __name__=='__main__':main()
