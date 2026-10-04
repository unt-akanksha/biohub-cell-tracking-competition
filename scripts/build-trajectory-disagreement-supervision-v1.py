"""Freeze source-only prediction features before attaching verified sparse GT."""
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.public_d4_full_movie import sha,csv_equivalent_graph
from research.trajectory_disagreement_data_v1 import FEATURES,extract,supervision


def main():
    started=time.monotonic()
    target=ROOT/'.biohub/cache/trajectory-disagreement-source-v1-supervision'
    assert not target.exists()
    plan_path=ROOT/'.biohub/cache/trajectory-disagreement-source-v1-plan/MOVIES.json'
    assert sha(plan_path)=='5c4b64793068565598537db6bb3af439513bde3d46e357ffeab59164bda020bf'
    plan=json.loads(plan_path.read_text());stems=[m['stem'] for m in plan['movies']]
    assert len(stems)==8 and all(m['role']=='optimization' for m in plan['movies'])
    roles_path=ROOT/'.biohub/cache/native-correspondence-v2-plan/MOVIES.json'
    assert sha(roles_path)==plan['source_plan_sha256']
    roles={m['stem']:m['role'] for m in json.loads(roles_path.read_text())['movies']}
    assert all(roles[s]=='optimization' for s in stems)
    base=ROOT/'.biohub/cache/trajectory-disagreement-source-v1-full-output'
    terminal=json.loads((base/'result.json').read_text())
    assert terminal['status']=='complete_prelabel_predictions' and terminal['mode']=='full'
    assert terminal['contract_sha256']=='ba27bfeb00ab800507275e0d9e5864e53033cb4350505a41331f574e7f9ab80c'
    assert terminal['inputs_unchanged'] and not terminal['ground_truth_opened'] and set(terminal['movies'])==set(stems)
    receipt=json.loads((ROOT/'reports/experiments/trajectory-disagreement-source-v1-full-harvest.json').read_text())
    assert receipt['status']=='verified_backup'
    for r in receipt['records']:assert sha(base/r['path'])==r['sha256']
    all_records=[];matrices=[];slices={};graphs={};counts={}
    for stem in stems:
        row=terminal['movies'][stem]['original'];assert row['frames']==100
        folder=base/(stem+'-original')
        initial=json.loads((folder/'pre-postprocess.json').read_text())
        final=json.loads((folder/'repaired-prediction.json').read_text())
        assert sha(folder/'repaired-prediction.json')==row['repaired_sha256']
        assert csv_equivalent_graph({int(k):v for k,v in final['nodes'].items()},final['edges'],100)==final
        with np.load(folder/'raw-candidates.npz',allow_pickle=False) as raw:
            records,matrix,summary=extract(initial,final,raw['coords'],raw['edges'])
        start=len(all_records);all_records.extend(dict(r,stem=stem,embryo=stem.split('_')[0],role='optimization') for r in records)
        slices[stem]=(start,len(all_records));matrices.append(matrix);graphs[stem]=final['nodes'];counts[stem]=summary
    target.mkdir()
    records_path=target/'records.json';records_path.write_text(json.dumps(all_records)+'\n',encoding='utf-8')
    feature_path=target/'features.npz'
    np.savez_compressed(feature_path,features=np.concatenate(matrices),feature_names=np.asarray(FEATURES))
    frozen=dict(records_sha256=sha(records_path),features_sha256=sha(feature_path),source_scope_sha256=sha(plan_path))
    (target/'PRELABEL.json').write_text(json.dumps(frozen,indent=2)+'\n')
    # All decision features are immutable before any source annotation is opened.
    truth_root=ROOT/'.biohub/cache/competition-train-geffs-packed-v1'
    manifest=truth_root/'train_geff_cache_manifest.json'
    assert sha(manifest)=='744f06f75388a7c9199179e5a90bfae4fb3aac835d4e5181893f3745601a9cb9'
    files=json.loads(manifest.read_text())['files']
    import zarr
    labels=np.full(len(all_records),-1,np.int64);reasons=['']*len(all_records);label_counts={};annotations={}
    for stem in stems:
        checks=[r for r in files if r['relative_path'].startswith(stem+'.geff/')]
        assert len(checks)==21
        for r in checks:assert sha(truth_root/'train'/r['relative_path'])==r['sha256']
        group=zarr.open_group(str(truth_root/'train'/(stem+'.geff')),mode='r')
        ids=np.asarray(group['nodes/ids']);times=np.asarray(group['nodes/props/t/values'])
        xyz={a:np.asarray(group[f'nodes/props/{a}/values']) for a in ('z','y','x')}
        nodes={int(i):dict(t=int(times[j]),**{a:float(xyz[a][j]) for a in xyz}) for j,i in enumerate(ids)}
        edges=np.asarray(group['edges/ids']).astype(np.int64).tolist()
        assert len(nodes)==len(ids) and len({b for a,b in edges})==len(edges)
        a,b=slices[stem]
        labels[a:b],reasons[a:b],label_counts[stem]=supervision(all_records[a:b],graphs[stem],nodes,edges)
        annotations[stem]=dict(nodes=len(nodes),edges=len(edges))
        print(json.dumps(dict(stem=stem,features=counts[stem],labels=label_counts[stem])),flush=True)
    assert sha(records_path)==frozen['records_sha256'] and sha(feature_path)==frozen['features_sha256']
    np.savez_compressed(target/'labels.npz',labels=labels,reasons=np.asarray(reasons))
    result=dict(status='source_supervision_complete',source_sha256=sha(Path(__file__)),
                helper_sha256=sha(ROOT/'research/trajectory_disagreement_data_v1.py'),
                matching_helper_sha256=sha(ROOT/'research/native_correspondence_data_v2.py'),
                **frozen,labels_sha256=sha(target/'labels.npz'),terminal_sha256=sha(base/'result.json'),
                feature_names=FEATURES,per_movie_features=counts,per_movie_labels=label_counts,annotations=annotations,
                candidates=len(labels),eligible_labels=int((labels>=0).sum()),
                class_counts={name:int((labels==i).sum()) for i,name in enumerate(('neural','motion','neither'))},
                source_only=True,holdout_opened=False,model_fitted=False,authorized_for_submission=False,
                public_backbone_training_overlap=True,seconds=time.monotonic()-started)
    (target/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    (ROOT/'reports/experiments/trajectory-disagreement-source-v1-supervision.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','candidates','eligible_labels','class_counts','seconds')}))


if __name__=='__main__':main()
