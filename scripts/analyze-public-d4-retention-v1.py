"""Separate membership and coordinate losses using already saved predictions."""
import contextlib
import io
import json
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.public_d4_full_movie import sha, STEMS


def main():
    v1=runpy.run_path(str(ROOT/'scripts/score-public-d4-full-movie-v1.py'))
    v2=runpy.run_path(str(ROOT/'scripts/score-public-d4-full-movie-v2.py'))
    import tracksdata as td
    from tracksdata.metrics import DistanceMatching
    root=ROOT/'.biohub/cache/public-d4-full-movie-v1-output'
    output=ROOT/'reports/experiments/public-d4-retention-v1-result.json'
    if output.exists():raise ValueError('Preserve completed retention analysis')
    _,prepared=v1['validate_predictions'](root,'61429f28d6fa3c42ee761d38456f9daca66986bd5845e38162f8baefcaf45683')
    truth_root=ROOT/'.biohub/cache/competition-train-geffs-packed-v1'
    v2['verify_truth_inventory'](truth_root,'744f06f75388a7c9199179e5a90bfae4fb3aac835d4e5181893f3745601a9cb9')
    manifest=json.loads((ROOT/'reports/experiments/public-d4-full-movie-v1-artifact-manifest.json').read_text())
    for row in manifest['files']:
        if row['path'].endswith('/pre-postprocess.json') and sha(root/row['path'])!=row['sha256']:
            raise ValueError('Pre-postprocess graph changed')
    result={}
    for stem in STEMS:
        result[stem]={}
        for arm in ('original','corrected'):
            pre=json.loads((root/f'{stem}-{arm}'/'pre-postprocess.json').read_text())
            final=prepared[stem][arm]
            retained={i:n for i,n in pre['nodes'].items() if i in final['nodes']}
            mixed={i:pre['nodes'].get(i,n) for i,n in final['nodes'].items()}
            states={'post_ilp':pre['nodes'],'retained_at_detector_positions':retained,
                    'final_with_detector_positions':mixed,'final':final['nodes']}
            matches={};counts={}
            for stage,nodes in states.items():
                graph=v1['prediction_graph'](dict(nodes=nodes,edges=[]))
                truth=td.graph.IndexedRXGraph.from_geff(str(truth_root/'train'/(stem+'.geff')))[0]
                with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                    graph.match(truth,matching=DistanceMatching(max_distance=7.,scale=(1.625,.40625,.40625)))
                ids={int(i) for i in graph.node_attrs()[td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID].to_list() if i is not None and i!=-1}
                matches[stage]=ids;counts[stage]=dict(nodes=len(nodes),matched_annotations=len(ids))
            row=dict(stages=counts,
                annotations_lost_with_deletion=len(matches['post_ilp']-matches['retained_at_detector_positions']),
                annotations_lost_with_coordinate_changes=len(matches['final_with_detector_positions']-matches['final']),
                annotations_gained_with_coordinate_changes=len(matches['final']-matches['final_with_detector_positions']))
            result[stem][arm]=row
            print(json.dumps(dict(stem=stem,arm=arm,**row)),flush=True)
    output.write_text(json.dumps(dict(run_id='public-d4-retention-v1',status='complete',
        source_sha256=sha(Path(__file__)),per_movie=result,authorized_for_submission=False,
        independently_held_out=False,gpu_hours=0),indent=2)+'\n')


if __name__=='__main__':main()
