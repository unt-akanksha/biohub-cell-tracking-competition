"""Use exactly replayed vectorized geometry for subsequent source batches."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    base=ROOT/'scripts/prepare-trajectory-event-cases-v1.py'
    assert sha(base)=='259359d59978a8a254cd3ec98ca35786ff157dbe2ff47345abe1aface25be35e'
    audit=ROOT/'reports/experiments/trajectory-event-fast-features-v1-audit.json'
    proof=json.loads(audit.read_text(encoding='utf-8'))
    assert proof['status']=='exact_source_feature_replay_passed' and proof['elements_compared']==102432780
    assert sha(ROOT/'research/trajectory_event_fast_features_v1.py')==proof['accelerated_features_sha256']
    assert sha(ROOT/'research/trajectory_event_features_v1.py')==proof['reference_features_sha256']
    source=base.read_text(encoding='utf-8')
    changes=[
        ('from research.trajectory_event_features_v1 import FEATURES, features',
         'from research.trajectory_event_features_v1 import FEATURES\nfrom research.trajectory_event_fast_features_v1 import FeatureContext'),
        ("graph=read(pp);initial=read(ip);groups=arrays(gp);labels=arrays(lp);edge=arrays(fp)",
         "graph=read(pp);initial=read(ip);groups=arrays(gp);labels=arrays(lp);edge=arrays(fp)\n            context=FeatureContext(graph,edge['features'])"),
        ("x=features(problem,graph,edge['features'])",'x=context.features(problem)'),
        ("'trajectory_event_features_v1.py','trajectory_event_supervision_v1.py','trajectory_event_assignment_v1.py')",
         "'trajectory_event_features_v1.py','trajectory_event_fast_features_v1.py','trajectory_event_supervision_v1.py','trajectory_event_assignment_v1.py')"),
        ("source_sha256=sha(Path(__file__)),feature_names=list(FEATURES),",
         "source_sha256=sha(Path(__file__)),feature_names=list(FEATURES),\n                exact_feature_replay_audit_sha256="+repr(sha(audit))+",")]
    for old,new in changes:
        assert source.count(old)==1,old
        source=source.replace(old,new)
    exec(compile(source,str(base),'exec'),dict(__name__='__main__',__file__=__file__))


if __name__=='__main__':main()
