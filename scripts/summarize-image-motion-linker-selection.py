"""Full-movie comparison against the parent linker and strongest motion model."""
import hashlib
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
BASE = runpy.run_path(str(ROOT/'scripts/summarize-known-null-selection.py'))
TRAIN = runpy.run_path(str(ROOT/'scripts/summarize-image-motion-linker.py'))


def compare(candidate,manifest,training,native,causal,parent,flow):
    if (manifest.get('image_motion_training') != TRAIN['EXPECTED']
        or training.get('image_motion') != TRAIN['EXPECTED'] or training.get('flow_unchanged') is not True
        or manifest.get('frozen_flow_sha256') != training.get('frozen_flow_sha256')
        or parent['checkpoint_sha256'] != TRAIN['INITIAL']
        or flow['flow_checkpoint_sha256'] != TRAIN['EXPECTED']['flow_checkpoint_sha256']):
        raise ValueError('Exact integrated training, parent and standalone flow references required')
    report = BASE['compare'](candidate,manifest,training,native,causal)
    for name,rows in [('parent',parent['per_movie']),('flow',flow['per_movie']['motion'])]:
        if [r['stem'] for r in rows] != [r['stem'] for r in candidate['per_movie']]:
            raise ValueError('Incomplete reference coverage: '+name)
        for row,previous in zip(candidate['per_movie'],rows):
            if any(row[k] != previous[k] for k in ('num_pred_nodes','node_recall','total_node_ratio')):
                raise ValueError('Reference detections changed: '+name)
        report['counts'][name] = {k:sum(r[k] for r in rows) for k in report['counts']['candidate']}
    score = candidate['summary']['score']
    report.update(status='verified_image_motion_linker_selection',
        delta_vs_parent=score-parent['summary']['score'],delta_vs_flow=score-flow['summaries']['motion']['score'],
        parent_summary=parent['summary'],flow_summary=flow['summaries']['motion'],
        regressions_vs_parent=[r['stem'] for r,p in zip(candidate['per_movie'],parent['per_movie'])
                               if r['adj_edge_jaccard'] < p['adj_edge_jaccard']-1e-12],
        regressions_vs_flow=[r['stem'] for r,p in zip(candidate['per_movie'],flow['per_movie']['motion'])
                             if r['adj_edge_jaccard'] < p['adj_edge_jaccard']-1e-12],
        caveat='Frozen image-flow prior plus extra learned-linker training; source selection only, not an isolated prior swap')
    report['decision'] = ('source_selection_gain_requires_embryo_audit'
        if min(report[k] for k in ('delta_vs_parent','delta_vs_flow','delta_vs_native','delta_vs_causal')) > 0
        else 'not_strongest_standalone')
    return report


if __name__ == '__main__':
    paths = dict(candidate=ROOT/'.biohub/cache/kernel-outputs/image-motion-linker-scoring-v1/image_motion_linker_score/selection_score.json',
        manifest=ROOT/'.biohub/cache/kernel-outputs/image-motion-linker-selection-v1/image_motion_linker_selection/outputs/selection_manifest.json',
        training=ROOT/'reports/experiments/image-motion-linker-v2-training.json',
        native=ROOT/'reports/experiments/independent-joint-selection-v1-score.json',
        causal=ROOT/'reports/experiments/causal-motion-selection-v1-score.json',
        parent=ROOT/'reports/experiments/independent-known-null-selection-v1-score.json',
        flow=ROOT/'reports/experiments/backward-flow-selection-v1-score.json')
    values = {k:json.loads(p.read_text()) for k,p in paths.items()}
    for name in ('native','causal','parent','flow'):
        values[name] = values[name]['result']
    report = compare(**values)
    report['source_sha256'] = {k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()}
    (ROOT/'reports/experiments/image-motion-linker-selection-v1-score.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('status','delta_vs_parent','delta_vs_flow','delta_vs_native','delta_vs_causal','decision','counts')},indent=2))
