"""Measure available correction labels in the first source batch; no fitting."""
from collections import Counter
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.trajectory_runtime_v1 import sha
from research.trajectory_correction_supervision_v1 import labels


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def arrays(p):
    with np.load(p,allow_pickle=False) as data:return dict(data)


def main():
    started=time.monotonic();name='trajectory-correction-supervision-v1'
    target=ROOT/'.biohub/cache'/name;receipt=ROOT/'reports/experiments'/(name+'.json')
    assert not target.exists() and not receipt.exists()
    inputs=ROOT/'.biohub/cache/trajectory-correction-components-v1';proof=read(inputs/'RESULT.json')
    assert proof['status']=='source_atomic_corrections_verified' and len(proof['records'])==8
    feature_root=ROOT/'.biohub/cache/trajectory-event-source-v1-b0-features';frozen=read(feature_root/'RESULT.json')
    label_root=ROOT/'.biohub/cache/trajectory-event-source-v1-b0-supervision';supervision=read(label_root/'RESULT.json')
    assert supervision['source_only'] and supervision['feature_receipt_sha256']==sha(feature_root/'RESULT.json')
    candidate_root=ROOT/'.biohub/cache/trajectory-event-anchor-source-v1'
    records={};pooled=Counter();target.mkdir()
    for stem,item in proof['records'].items():
        part_path=inputs/(stem+'-components.json');assert sha(part_path)==item['components_sha256']
        pp=feature_root/(stem+'-prediction.json');cp=candidate_root/(stem+'-prediction.json')
        gp=feature_root/(stem+'-groups.npz');lp=label_root/(stem+'-labels.npz')
        assert sha(pp)==item['baseline_sha256'] and sha(cp)==item['candidate_sha256']
        assert sha(gp)==frozen['per_movie'][stem]['groups_sha256']
        assert sha(lp)==supervision['per_movie'][stem]['labels_sha256']
        known=arrays(lp)
        rows=labels(read(part_path),read(pp),read(cp),arrays(gp),known['target'],known['safe'])
        assert len(rows)==item['components']
        counts=Counter()
        for row in rows:
            counts[{1:'partial_improvement',0:'partial_regression',-1:'unlabeled_or_neutral'}[row['label']]]+=1
            counts.update({k:v for k,v in row.items() if k!='label'})
        out=target/(stem+'-labels.json');out.write_text(json.dumps(rows)+'\n',encoding='utf-8')
        records[stem]=dict(counts=counts,labels_sha256=sha(out));pooled.update(counts)
    result=dict(status='source_partial_correction_labels_audited',records=records,pooled=pooled,
        components_sha256=sha(inputs/'RESULT.json'),supervision_sha256=sha(label_root/'RESULT.json'),
        helper_sha256=sha(ROOT/'research/trajectory_correction_supervision_v1.py'),source_sha256=sha(Path(__file__)),
        source_only=True,selection_or_validation_opened=False,model_fitted=False,
        labels_measure_known_links_not_full_component_score=True,
        unknown_children_are_not_negatives=True,ambiguous_matches_excluded=True,
        seconds=time.monotonic()-started,candidate_changed=False)
    text=json.dumps(result,indent=2)+'\n';(target/'RESULT.json').write_text(text,encoding='utf-8');receipt.write_text(text,encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
