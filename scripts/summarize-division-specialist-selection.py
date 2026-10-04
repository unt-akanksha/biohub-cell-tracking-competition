"""Compare a division specialist with its parent and both earlier controls."""
import hashlib
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
BASE = runpy.run_path(str(ROOT/'scripts/summarize-known-null-selection.py'))
INITIAL = 'b64254aece75ef709e517621757a5313ac665e0803050ce4f13fa2a76602d6ef'


def compare(candidate,manifest,training,native,causal,parent):
    if (manifest.get('division_specialist_training') != dict(version=1,division_window_mass=.5)
        or manifest.get('division_sampling') != training.get('division_sampling')
        or training.get('division_columns',0) <= 0 or parent['checkpoint_sha256'] != INITIAL):
        raise ValueError('Verified specialist training and exact learned-parent reference required')
    report = BASE['compare'](candidate,manifest,training,native,causal)
    if [r['stem'] for r in parent['per_movie']] != [r['stem'] for r in candidate['per_movie']]:
        raise ValueError('Parent comparison movie coverage mismatch')
    for current,previous in zip(candidate['per_movie'],parent['per_movie']):
        for key in ('num_pred_nodes','node_recall','total_node_ratio'):
            if current[key] != previous[key]:
                raise ValueError('Parent detection metrics changed: '+key)
    report.update(status='verified_division_specialist_selection',parent_summary=parent['summary'],
        delta_vs_parent=candidate['summary']['score']-parent['summary']['score'],
        adjusted_edge_regressions_vs_parent=[r['stem'] for r,p in zip(candidate['per_movie'],parent['per_movie'])
                                            if r['adj_edge_jaccard'] < p['adj_edge_jaccard']-1e-12],
        caveat='Case-balanced sampling plus positive-class balancing with retained true-null supervision; source selection only')
    report['counts']['parent'] = {key:sum(r[key] for r in parent['per_movie']) for key in report['counts']['candidate']}
    return report


if __name__ == '__main__':
    paths = dict(candidate=ROOT/'.biohub/cache/kernel-outputs/division-specialist-scoring-v1/division_specialist_score/selection_score.json',
        manifest=ROOT/'.biohub/cache/kernel-outputs/division-specialist-selection-v1/division_specialist_selection/outputs/selection_manifest.json',
        training=ROOT/'reports/experiments/division-specialist-v2-training.json',
        native=ROOT/'reports/experiments/independent-joint-selection-v1-score.json',
        causal=ROOT/'reports/experiments/causal-motion-selection-v1-score.json',
        parent=ROOT/'reports/experiments/independent-known-null-selection-v1-score.json')
    values = {key:json.loads(path.read_text()) for key,path in paths.items()}
    for key in ('native','causal','parent'):
        values[key] = values[key]['result']
    report = compare(**values)
    report['source_sha256'] = {key:hashlib.sha256(path.read_bytes()).hexdigest() for key,path in paths.items()}
    (ROOT/'reports/experiments/division-specialist-selection-v1-score.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({key:report[key] for key in ('status','delta_vs_parent','delta_vs_native','delta_vs_causal',
        'counts','adjusted_edge_regressions_vs_parent')},indent=2))
