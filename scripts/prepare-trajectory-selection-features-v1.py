"""Freeze selection candidate geometry/features without opening any annotations."""
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.public_d4_full_movie import sha,csv_equivalent_graph
from research.trajectory_candidate_inventory_v1 import candidates
from research.trajectory_candidate_ranker_v1 import FEATURES,features


def main():
    started=time.monotonic();target=ROOT/'.biohub/cache/trajectory-selection-features-v1'
    assert not target.exists()
    scope=ROOT/'.biohub/cache/trajectory-ranker-selection-v1-plan/MOVIES.json'
    assert sha(scope)=='1d9cbb244bbdc23d631c45811ccd222bb2e29d5b70a2cd6f05fa87aeb5ff297e'
    plan=json.loads(scope.read_text());stems=[m['stem'] for m in plan['movies']]
    assert len(stems)==10 and all(m['role']=='selection' for m in plan['movies'])
    base=ROOT/'.biohub/cache/trajectory-ranker-selection-v1-full-output'
    terminal=json.loads((base/'result.json').read_text())
    assert terminal['status']=='complete_prelabel_predictions' and terminal['mode']=='full'
    assert terminal['contract_sha256']=='286b0eb2d0e2632e9914e3edfbb8c31f74a5429c9fa04f0a99e396a9e123dfab'
    assert terminal['inputs_unchanged'] and not terminal['ground_truth_opened'] and set(terminal['movies'])==set(stems)
    backup=json.loads((ROOT/'reports/experiments/trajectory-ranker-selection-v1-full-harvest.json').read_text())
    assert backup['status']=='verified_backup'
    actual={p.relative_to(base).as_posix() for p in base.rglob('*') if p.is_file()}
    assert actual=={r['path'] for r in backup['records']}
    for r in backup['records']:assert sha(base/r['path'])==r['sha256']
    source=ROOT/'.biohub/cache/trajectory-candidate-ranker-v1/CONTRACT.json'
    assert sha(ROOT/'research/trajectory_candidate_ranker_v1.py')==json.loads(source.read_text())['helper_sha256']
    target.mkdir();records={}
    for stem in stems:
        proof=terminal['movies'][stem]['original'];assert proof['frames']==100
        folder=base/(stem+'-original')
        initial=json.loads((folder/'pre-postprocess.json').read_text())
        path=folder/'repaired-prediction.json';assert sha(path)==proof['repaired_sha256']
        final=json.loads(path.read_text())
        assert csv_equivalent_graph({int(k):v for k,v in final['nodes'].items()},final['edges'],100)==final
        groups=candidates(initial,final)
        with np.load(folder/'raw-candidates.npz',allow_pickle=False) as raw:matrix=features(initial,final,groups,raw['edges'])
        gp=target/(stem+'-groups.npz');fp=target/(stem+'-features.npz')
        np.savez_compressed(gp,**groups);np.savez_compressed(fp,features=matrix,feature_names=np.asarray(FEATURES))
        records[stem]=dict(groups_sha256=sha(gp),features_sha256=sha(fp),baseline_sha256=sha(path),
                           queries=len(groups['children']),edges=len(groups['parents']))
        print(json.dumps(dict(stem=stem,**records[stem])),flush=True)
    result=dict(status='selection_features_frozen_no_labels',per_movie=records,scope_sha256=sha(scope),
                terminal_sha256=sha(base/'result.json'),source_sha256=sha(Path(__file__)),
                candidate_helper_sha256=sha(ROOT/'research/trajectory_candidate_inventory_v1.py'),
                feature_helper_sha256=sha(ROOT/'research/trajectory_candidate_ranker_v1.py'),
                ground_truth_opened=False,model_applied=False,authorized_for_submission=False,
                seconds=time.monotonic()-started)
    (target/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    (ROOT/'reports/experiments/trajectory-selection-features-v1.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],seconds=result['seconds'])))


if __name__=='__main__':main()
