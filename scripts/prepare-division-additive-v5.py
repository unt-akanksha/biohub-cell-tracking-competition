"""Freeze 21 positive-only image extraction recipes from the completed source audit."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.native_correspondence_data_v2 import VOXEL
from research.native_division_matches_v5 import isolated_additive_matches


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if sha(ROOT/'research/native_division_matches_v5.py')!='a4283493608f0aa3867d6ff79d400c6a52c7cc1151f4a2e0ecd7106c6d0a6fee':raise ValueError('Matcher changed')
    source=ROOT/'.biohub/cache/native-division-proposal-audit-v4-full-output'
    audit_path=source/'RESULT.json'
    if sha(audit_path)!='0f12f8a011c0575f63f9160e97db2dcc2dc8ccab6cb8c898e3963cc18e6a2d4e':raise ValueError('Audit changed')
    audit=json.loads(audit_path.read_text());plan_path=ROOT/'.biohub/cache/native-division-proposal-audit-v4-plan/MOVIES.json'
    if sha(plan_path)!=audit['movie_plan_sha256']:raise ValueError('Source roles changed')
    movies={m['stem']:m for m in json.loads(plan_path.read_text())['movies']}
    dry_path=ROOT/'reports/experiments/native-division-additive-matches-v5-dry-run.json';dry=json.loads(dry_path.read_text())
    selected=[r for r in dry['events'] if r['additive'] and not r['strict']]
    artifacts={Path(r['path']).stem:r for r in audit['point_artifacts']}
    frames={(r['stem'],r['time']):r['raw_sha256'] for r in audit['frames']}
    recipes=[]
    for r in selected:
        stem=r['stem'];movie=movies[stem];t=r['transition']
        if movie['role']!='optimization':raise ValueError('Only optimization positives')
        artifact=artifacts[stem];path=source/artifact['path']
        if sha(path)!=artifact['sha256']:raise ValueError('Proposals changed')
        coords=[];indices=[]
        with np.load(path,allow_pickle=False) as packet:
            for time,ids in ((t,[r['parent']]),(t+1,sorted(r['children']))):
                nodes=[n for n in movie['nodes'] if int(n[1])==time]
                points=packet[f'baseline_{time:03d}']
                truth=np.array([n[2:] for n in nodes],np.float32).reshape(-1,3)*VOXEL
                matches=isolated_additive_matches(truth,points)
                lookup={int(n[0]):i for i,n in enumerate(nodes)}
                for node in ids:
                    index=matches[lookup[node]];indices.append(index);coords.append(points[index].tolist())
        if indices[1]==indices[2] or any(np.linalg.norm(np.array(coords[i])-coords[0])>20 for i in (1,2)):raise ValueError('Invalid positive triplet')
        recipes.append(dict(stem=stem,embryo=movie['embryo'],role='optimization',transition=t,parent=r['parent'],
                            children=sorted(r['children']),coords=coords,proposal_indices=indices,
                            raw_sha256={str(time):frames[(stem,time)] for time in (t,t+1)}))
    if len(recipes)!=21 or sum(r['embryo']=='44b6' for r in recipes)!=3:raise ValueError('Unexpected positive extension')
    output=ROOT/'.biohub/cache/native-division-additive-v5-plan';output.mkdir(exist_ok=False)
    result=dict(run_id='native-division-additive-v5',recipes=recipes,source_audit_sha256=sha(audit_path),dry_run_sha256=sha(dry_path),
                target_pilot_labels_used=False,selection_opened=False,authorized_for_submission=False)
    (output/'RECIPES.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='positive_recipes_frozen',events=len(recipes),recipe_sha256=sha(output/'RECIPES.json'))))


if __name__=='__main__':main()
