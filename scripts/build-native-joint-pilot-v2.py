"""Apply one frozen joint-link policy to verified image-only posterior caches."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.native_joint_linking_v2 import joint_repair,links
from research.trajectory_runtime_v1 import validate_graph


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    start=time.monotonic();old=ROOT/'.biohub/cache/native-repair-pilot-v2-bundle'
    original=json.loads((old/'PILOT.json').read_text())
    cache_path=ROOT/'reports/experiments/native-posterior-v2-cache-result.json'
    cache=json.loads(cache_path.read_text());attribution=json.loads((ROOT/'reports/experiments/native-repair-pilot-v2-attribution.json').read_text())
    if cache['status']!='label_free_attribution_complete' or cache['ground_truth_opened'] or cache['model_or_graph_changed'] or cache['pilot_sha256']!=sha(old/'PILOT.json'):
        raise ValueError('Require exact label-free image posterior cache')
    if [r['counts'] for r in cache['movies']]!=[r['counts'] for r in attribution['movies']]:raise ValueError('Cached inference changed original decision counts')
    bundle=ROOT/'.biohub/cache/native-joint-pilot-v2-bundle';output=ROOT/'.biohub/cache/native-joint-pilot-v2-output'
    bundle.mkdir(exist_ok=False);output.mkdir(exist_ok=False)
    for relative in ('research/native_joint_linking_v2.py','scripts/build-native-joint-pilot-v2.py','reports/experiments/native-joint-linking-v2-design.md'):
        target=bundle/relative;target.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(ROOT/relative,target)
    (bundle/'graphs').mkdir();(bundle/'posteriors').mkdir()
    shutil.copy2(cache_path,bundle/'CACHE_MANIFEST.json');movies=[];prepared=[]
    for item,record in zip(original['movies'],cache['movies']):
        stem=item['stem']
        if record['stem']!=stem:raise ValueError('Cache movie order changed')
        source=old/item['graph']
        expected=next(r['sha256'] for r in original['files'] if r['path']==item['graph'])
        if sha(source)!=expected:raise ValueError('Original graph changed')
        graph=json.loads(source.read_text());validate_graph(graph,100)
        cache_record=record['prediction_cache'];p=ROOT/'.biohub/cache/native-posterior-v2-cache'/cache_record['file']
        if p.stat().st_size!=cache_record['bytes'] or sha(p)!=cache_record['sha256']:raise ValueError('Posterior cache transfer failed verification')
        with np.load(p,allow_pickle=False) as arrays:ids=arrays['ids'];probabilities=arrays['probabilities']
        expected_queries={n['node_id'] for n in graph['nodes'].values() if n['t']>0}
        if set(ids[:,0])!=expected_queries or len(ids)!=len(expected_queries) or len(ids)!=record['counts']['queries']:raise ValueError('Incomplete full-movie posterior queries')
        for row in ids:
            for parent in row[1:][row[1:]>=0]:
                if graph['nodes'][str(int(parent))]['t']+1!=graph['nodes'][str(int(row[0]))]['t']:raise ValueError('Nonadjacent cached parent')
        graph_path='graphs/'+stem+'.json';posterior_path='posteriors/'+cache_record['file']
        shutil.copy2(source,bundle/graph_path);shutil.copy2(p,bundle/posterior_path)
        movies.append(dict(stem=stem,graph=graph_path,posteriors=posterior_path));prepared.append((graph,ids,probabilities))
    contract=dict(run_id='native-joint-pilot-v2',movies=movies,models=original['models'],
        original_native_pilot_sha256=sha(old/'PILOT.json'),posterior_manifest_sha256=sha(cache_path),
        ground_truth_included=False,authorized_for_submission=False,
        files=[dict(path=p.relative_to(bundle).as_posix(),sha256=sha(p)) for p in sorted(bundle.rglob('*')) if p.is_file()])
    (bundle/'PILOT.json').write_text(json.dumps(contract,indent=2)+'\n')
    result=dict(status='complete_prelabel_predictions',pilot_sha256=sha(bundle/'PILOT.json'),smoke=False,
                ground_truth_opened=False,authorized_for_submission=False,movies=[])
    for item,(graph,ids,probabilities) in zip(movies,prepared):
        tick=time.monotonic();stem=item['stem'];repaired,events=joint_repair(graph,ids,probabilities);validate_graph(repaired,100)
        _,old_out=links(graph);_,new_out=links(repaired)
        if repaired['nodes']!=graph['nodes'] or any(sorted(new_out[p])!=sorted(children) for p,children in old_out.items() if len(children)==2):raise ValueError('Original nodes/divisions changed')
        destination=output/(stem+'.json');destination.write_text(json.dumps(repaired))
        details=output/(stem+'-repairs.json');details.write_text(json.dumps(events,indent=2)+'\n')
        record=dict(stem=stem,frames_evaluated=100,groups=len(ids),repairs=len(events),
                    additions=sum(e['kind']=='persistent_division' for e in events),
                    rewires=sum(len(e['changes']) for e in events if e['kind']=='joint_identity_cycle'),
                    elapsed_seconds=time.monotonic()-tick,prediction_sha256=sha(destination),repairs_sha256=sha(details),
                    parent_sha256=sha(bundle/item['graph']))
        result['movies'].append(record);print(json.dumps(record),flush=True)
    result['elapsed_seconds']=time.monotonic()-start
    (output/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],pilot_sha256=result['pilot_sha256'],seconds=result['elapsed_seconds'])))


if __name__=='__main__':main()
