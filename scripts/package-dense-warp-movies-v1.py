"""Stage existing complete-movie runtime, two fixed learned linkers and controls."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.dense_warp_movie_adapter_v1 import build_runtime


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    old=ROOT/'.biohub/cache/trajectory-division-full-movie-v1-bundle'
    contract=json.loads((old/'CONTRACT.json').read_text())
    assert sha(old/'CONTRACT.json')=='7d4d1bc9f81f37fc2a8d23bd2449fc53f99bfb23eff39d02460b884661f8bdd1'
    output=ROOT/'.biohub/cache/dense-warp-movie-v1-bundle';output.mkdir(exist_ok=False)
    for name,digest in contract['bundle_sha256'].items():
        assert sha(old/name)==digest
        shutil.copy2(old/name,output/name)
    adapted=build_runtime((old/'run-trajectory-division-full-movie-v1.py').read_text())
    (output/'run-dense-warp-movies-v1.py').write_text(adapted)
    shutil.copy2(ROOT/'research/dense_warp_movie_adapter_v1.py',output/'dense_warp_movie_adapter_v1.py')
    proof=ROOT/'reports/experiments/dense-warp-v1-verification.json'
    assert json.loads(proof.read_text())['status']=='verified_real_pair_screen'
    result=json.loads((ROOT/'.biohub/cache/dense-warp-v1-full-output/RESULT.json').read_text())
    for row in result['folds']:
        name=row['source']+'-final.pt';path=ROOT/'.biohub/cache/dense-warp-v1-full-output'/name
        assert sha(path)==row['sha256'];shutil.copy2(path,output/name)
    prior=ROOT/'.biohub/cache/trajectory-division-full-v1-output'
    prior_result=json.loads((prior/'result.json').read_text())
    assert sha(prior/'result.json')=='40c35432a298d22f347cbdd111b1cd6d21ea3965acc35b4fc63369d3147544b1'
    for stem in contract['stems']:
        path=prior/(stem+'-original')/'repaired-prediction.json'
        assert sha(path)==prior_result['movies'][stem]['original']['repaired_sha256']
        shutil.copy2(path,output/(stem+'-control.json'))
    pins={p.name:sha(p) for p in output.iterdir()}
    contract.update(run_id='dense-warp-complete-movie-v1',bundle_sha256=pins,arms=['original','warp44','warp6'],
                    scientific_delta='Primary linker checkpoint only; detector/encoder/secondary/DeepCenter/D4/thresholds/trajectory repair unchanged',
                    partial_screen_passed=False,previously_exposed_diagnostic_movies=True,
                    complete_movie_gate=dict(pooled_score_strict=True,raw_edge_jaccard_strict=True,each_embryo_nonregression=True,each_movie_nonregression=True),
                    ensemble_authorized=False,all_eight_confirmation_required=True)
    contract['full']['arms']=3
    (output/'CONTRACT.json').write_text(json.dumps(contract,indent=2)+'\n')
    archive=output.with_suffix('.tar')
    with tarfile.open(archive,'w') as tf:tf.add(output,arcname=output.name)
    receipt=dict(status='staged_diagnostic',contract_sha256=sha(output/'CONTRACT.json'),archive_sha256=sha(archive),bytes=archive.stat().st_size)
    (ROOT/'reports/experiments/dense-warp-movies-v1-build.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()
