"""Measure source-only headroom, without constructing/exporting oracle graphs."""
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.public_d4_full_movie import sha
from research.trajectory_assignment_feasibility_v1 import counts_only


def main():
    started=time.monotonic()
    report=ROOT/'reports/experiments/trajectory-assignment-feasibility-v1.json';assert not report.exists()
    inv_path=ROOT/'reports/experiments/trajectory-candidate-supervision-v1.json'
    inv=json.loads(inv_path.read_text());assert inv['source_only'] and not inv['selection_or_validation_opened']
    plan=ROOT/'.biohub/cache/trajectory-disagreement-source-v1-plan/MOVIES.json'
    assert sha(plan)==inv['source_scope_sha256']=='5c4b64793068565598537db6bb3af439513bde3d46e357ffeab59164bda020bf'
    source=ROOT/'.biohub/cache/trajectory-disagreement-source-v1-full-output'
    backup=json.loads((ROOT/'reports/experiments/trajectory-disagreement-source-v1-full-harvest.json').read_text())
    assert backup['status']=='verified_backup'
    for r in backup['records']:assert sha(source/r['path'])==r['sha256']
    base=ROOT/'.biohub/cache/trajectory-candidate-supervision-v1';rows={}
    for stem in inv['per_movie']:
        group_path=base/(stem+'-candidates.npz');labels_path=base/(stem+'-labels.npz')
        assert sha(group_path)==inv['candidate_sha256'][stem] and sha(labels_path)==inv['label_sha256'][stem]
        final=json.loads((source/(stem+'-original')/'repaired-prediction.json').read_text())
        with np.load(group_path,allow_pickle=False) as f:groups=dict(f)
        with np.load(labels_path,allow_pickle=False) as f:rows[stem]=counts_only(final,groups,f['target'],f['safe'])
        print(json.dumps(dict(stem=stem,**rows[stem])),flush=True)
    result=dict(status='source_assignment_feasibility_complete',per_movie=rows,source_only=True,
                source_inventory_sha256=sha(inv_path),source_sha256=sha(Path(__file__)),
                helper_sha256=sha(ROOT/'research/trajectory_assignment_feasibility_v1.py'),
                source_selection_or_validation_used=False,oracle_predictions_exported=False,
                model_fitted=False,authorized_for_submission=False,seconds=time.monotonic()-started)
    report.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],seconds=result['seconds'])))


if __name__=='__main__':main()
