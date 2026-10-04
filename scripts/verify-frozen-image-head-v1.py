"""Independently replay source-only selection and verify local model/data receipts."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from research.frozen_image_head import morphology,predict,standardizer
from research.image_context_quality import metrics,utility


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    base=ROOT/'.biohub/cache/frozen-image-head-v1-output'
    root=base/'frozen-head-v1-full'
    data_root=ROOT/'.biohub/cache/image-division-context-v2'
    output=ROOT/'reports/experiments/frozen-image-head-v1-verification.json'
    if output.exists():raise ValueError('Preserve earlier verification')
    contract=ROOT/'.biohub/cache/frozen-image-head-v1-code/CONTRACT.json'
    if sha(contract)!='a80e0e80bcf6103f78f10bdda658a737b00b90a839797c03f4941617f62db48f':
        raise ValueError('Frozen source contract changed')
    for name,digest in json.loads(contract.read_text())['files'].items():
        if sha(contract.parent/name)!=digest:raise ValueError('Frozen code changed')
    inventories=list(base.glob('inventory-*.json'))
    if not inventories:raise ValueError('No verified remote inventory')
    for inventory in inventories:
        for row in json.loads(inventory.read_text())['files']:
            path=base/row['path']
            if sha(path)!=row['sha256'] or path.stat().st_size!=row['bytes']:
                raise ValueError('Backed-up artifact changed')
    result=json.loads((root/'result.json').read_text())
    if result['held_out_embryo_scores_opened'] or result['authorized_for_submission']:
        raise ValueError('Unexpected quality-gate transition')
    evidence={}
    for source in ('44b6','6bba'):
        folder=root/f'source-{source}'
        terminal=json.loads((folder/'terminal.json').read_text())
        if sha(folder/'best-head.npz')!=terminal['head_sha256']:raise ValueError('Head changed')
        with np.load(folder/'best-head.npz',allow_pickle=False) as a:state=dict(a)
        with np.load(folder/'optimization-features.npz',allow_pickle=False) as a:optimization=dict(a)
        with np.load(folder/'selection-features.npz',allow_pickle=False) as a:selection=dict(a)
        fitted=standardizer(optimization['features'])
        np.testing.assert_allclose(state['mean'],fitted['mean'],rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(state['scale'],fitted['scale'],rtol=1e-12,atol=1e-12)
        all_movies=[]
        for role,packet in (('optimization',optimization),('selection',selection)):
            records=json.loads((data_root/f'{source}-{role}-inventory.json').read_text())
            if any(r['role']!=role or r['embryo']!=source for r in records):raise ValueError('Source exclusion failed')
            all_movies.append({r['stem'] for r in records})
            with np.load(data_root/f'{source}-{role}.npz',allow_pickle=False) as a:
                for key in ('targets','eligible','weights'):np.testing.assert_array_equal(packet[key],a[key])
                np.testing.assert_allclose(packet['features'][:,-45:],morphology(a['context'],a['mask']),rtol=0,atol=1e-12)
        if all_movies[0]&all_movies[1]:raise ValueError('Source movie overlap')
        scores=predict(selection['features'],state)
        with np.load(folder/'source-selection.npz',allow_pickle=False) as a:
            np.testing.assert_allclose(scores,a['scores'],rtol=1e-12,atol=1e-12)
        mask=selection['eligible'].astype(bool)
        row=metrics(selection['targets'][mask],scores[mask])
        if utility(row)!=utility(terminal['metrics']):
            # BLAS versions may change only the last few BCE bits, never ranking/counts.
            for key in ('average_precision','zero_fp_tp','source_gate_passed'):
                if row[key]!=terminal['metrics'][key]:raise ValueError('Source utility changed')
            np.testing.assert_allclose(row['binary_cross_entropy'],terminal['metrics']['binary_cross_entropy'],rtol=1e-12)
        history=json.loads((folder/'selection-history.json').read_text())['rows']
        best=max((x for x in history if x['status']=='converged'),key=lambda x:utility(x['metrics']))
        if best['penalty']!=terminal['penalty']:raise ValueError('Wrong source-selected penalty')
        records=json.loads((data_root/f'{source}-selection-inventory.json').read_text())
        ordered=np.flatnonzero(mask)[np.argsort(-scores[mask],kind='stable')]
        evidence[source]=dict(source_metrics=row,source_movie_counts=list(map(len,all_movies)),
            penalty=terminal['penalty'],head_parameters=len(state['coefficients'])+1,
            eligible_source_ranking=[dict(row=int(i),**records[i],target=float(selection['targets'][i]),
                                         score=float(scores[i])) for i in ordered])
    payload=dict(status='verified_rejection',run_id='frozen-image-head-v1',
        terminal_sha256=sha(root/'result.json'),held_out_embryo_scores_opened=False,
        authorized_for_submission=False,source_normalization_replay=True,
        source_scores_replay=True,image_morphology_replay=True,source_exclusion_verified=True,
        actual_leaderboard_score=None,evidence=evidence)
    output.write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(dict(status=payload['status'],result_sha256=sha(output),
                         source_metrics={k:v['source_metrics'] for k,v in evidence.items()})))


if __name__=='__main__':main()
