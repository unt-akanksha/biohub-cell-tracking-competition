"""Fixed stage attribution of a failed D4 gate; diagnostics only, no GPU."""
import json
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.public_d4_full_movie import sha,STEMS
V1=runpy.run_path(str(ROOT/'scripts/score-public-d4-full-movie-v1.py'))
V2=runpy.run_path(str(ROOT/'scripts/score-public-d4-full-movie-v2.py'))


def main():
    import numpy as np
    import tracksdata as td
    from tracksdata.metrics import DistanceMatching
    folder=ROOT/'.biohub/cache/public-d4-full-movie-v1-output'
    output=ROOT/'reports/experiments/public-d4-full-movie-v1-stage-attribution.json'
    if output.exists():raise ValueError('Preserve completed attribution')
    terminal,prepared=V1['validate_predictions'](folder,'61429f28d6fa3c42ee761d38456f9daca66986bd5845e38162f8baefcaf45683')
    truth_root=ROOT/'.biohub/cache/competition-train-geffs-packed-v1'
    V2['verify_truth_inventory'](truth_root,'744f06f75388a7c9199179e5a90bfae4fb3aac835d4e5181893f3745601a9cb9')
    results={}
    for stem in STEMS:
        results[stem]={}
        for arm in ('original','corrected'):
            root=folder/f'{stem}-{arm}'
            with np.load(root/'raw-candidates.npz',allow_pickle=False) as cache:
                coords=cache['coords']
            raw=dict(nodes={str(i):dict(node_id=i,t=int(p[0]),z=int(p[1]),y=int(p[2]),x=int(p[3]))
                            for i,p in enumerate(coords)},edges=[])
            pre=json.loads((root/'pre-postprocess.json').read_text())
            final=prepared[stem][arm]
            records={}; sets={}
            for stage,data in (('raw_detector',raw),('post_ilp',pre),('final',final)):
                # Node matching only; never treat the raw multi-candidate graph
                # as a submission or change the actual saved predictions.
                graph=V1['prediction_graph'](dict(nodes=data['nodes'],edges=[]))
                truth=td.graph.IndexedRXGraph.from_geff(str(truth_root/'train'/(stem+'.geff')))[0]
                graph.match(truth,matching=DistanceMatching(max_distance=7.,scale=(1.625,.40625,.40625)))
                matched=graph.node_attrs()[td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID].to_list()
                ids={int(i) for i in matched if i is not None and i!=-1}
                sets[stage]=ids
                records[stage]=dict(predicted_nodes=graph.num_nodes(),matched_truth_nodes=len(ids),
                                     annotated_nodes=truth.num_nodes(),recall=len(ids)/truth.num_nodes())
            records['stage_losses']=dict(detector_matches_lost_after_ilp=len(sets['raw_detector']-sets['post_ilp']),
                ilp_matches_lost_after_repairs_pruning_smoothing=len(sets['post_ilp']-sets['final']),
                new_matches_after_repairs_pruning_smoothing=len(sets['final']-sets['post_ilp']))
            results[stem][arm]=records
            print(json.dumps(dict(stem=stem,arm=arm,**records)),flush=True)
    result=dict(run_id='public-d4-full-movie-v1-stage-attribution',status='complete',
        source_sha256=sha(Path(__file__)),per_movie=results,annotations_previously_exposed=True,
        purpose='Explain failed embryo gate, not train or select an output',
        independently_held_out=False,authorized_for_submission=False,new_gpu_hours=0)
    output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
