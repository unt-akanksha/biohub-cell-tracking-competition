"""Verify terminal artifacts and recompute screening gates independently."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.dense_warp_screen_v1 import checked_gate


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root=ROOT/'.biohub/cache/dense-warp-v1-full-output';bundle=ROOT/'.biohub/cache/dense-warp-v1-full-bundle'
    result=json.loads((root/'RESULT.json').read_text())
    assert result['status']=='dense_warp_training_complete' and result['successful_updates']==4000
    assert result['contract_sha256']==sha(bundle/'CONTRACT.json') and result['plan_sha256']==sha(bundle/'plan/MOVIES.json')
    contract=json.loads((bundle/'CONTRACT.json').read_text())
    for row in contract['files']:assert sha(bundle/row['path'])==row['sha256']
    assert sha(root/'features.pt')==result['features_sha256']
    assert result['public_backbone_training_overlap'] and not result['independently_held_out'] and not result['authorized_for_submission']
    assert [f['source'] for f in result['folds']]==['44b6','6bba']
    qualified=[]
    for row in result['folds']:
        assert row['steps']==2000
        assert sha(root/(row['source']+'-final.pt'))==row['sha256']
        passed=checked_gate(row['before'],row['after']);assert passed==row['source_pass']
        assert row['opposite_opened']==passed
        if passed:
            opposite=checked_gate(row['opposite_before'],row['opposite_after']);assert opposite==row['opposite_pass']
            if opposite:qualified.append(row['source'])
        else:assert 'opposite_after' not in row and 'opposite_pass' not in row
    assert len(qualified)==result['individual_experts_qualified']
    verification=dict(status='verified_real_pair_screen',qualified_sources=qualified,
                      result_sha256=sha(root/'RESULT.json'),full_movie_validated=False,
                      independently_held_out=False,authorized_for_submission=False)
    path=ROOT/'reports/experiments/dense-warp-v1-verification.json';assert not path.exists()
    path.write_text(json.dumps(verification,indent=2)+'\n');print(json.dumps(verification))


if __name__=='__main__':main()
