"""Freeze an encoder-only AMP runtime screen; linker and DeepCenter stay FP32."""
import ast
import json
from pathlib import Path
import shutil
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.trajectory_runtime_v1 import sha


def main():
    source=ROOT/'.biohub/cache/trajectory-division-full-movie-v1-bundle'
    if sha(source/'CONTRACT.json')!='7d4d1bc9f81f37fc2a8d23bd2449fc53f99bfb23eff39d02460b884661f8bdd1':
        raise ValueError('Validated baseline changed')
    contract=json.loads((source/'CONTRACT.json').read_text())
    out=ROOT/'.biohub/cache/trajectory-amp-encode-v1-bundle';out.mkdir(exist_ok=False)
    for name,digest in contract['bundle_sha256'].items():
        if sha(source/name)!=digest:raise ValueError('Baseline runtime changed')
        shutil.copyfile(source/name,out/name)
    name='run-trajectory-division-full-movie-v1.py';code=(out/name).read_text()
    anchor="    post_source = (args.bundle / 'public-postprocess-d4-corrected.py').read_text()"
    if code.count(anchor)!=1:raise ValueError('Unexpected model load structure')
    addition='''    # Benchmark-supported new hypothesis: encoder-only FP16 arithmetic.
    # Preserve FP32 TTA accumulation, edge transformer, ILP and DeepCenter.
    for candidate_model in models:
        original_encode = candidate_model.encode
        def amp_encode(imgs, _original=original_encode):
            with torch.autocast('cuda', dtype=torch.float16):
                features, logits = _original(imgs)
            return features.float(), [value.float() for value in logits]
        candidate_model.encode = amp_encode
'''
    code=code.replace(anchor,addition+anchor).replace("run_id='trajectory-division-full-movie-v1'","run_id='trajectory-amp-encode-v1'")
    ast.parse(code);(out/name).write_text(code)
    contract.update(run_id='trajectory-amp-encode-v1',
        baseline_contract_sha256=sha(source/'CONTRACT.json'),
        precision_change='UNetNodeTransformer.encode only; output casts FP32; all other math remains baseline',
        purpose='Runtime headroom screen; precision changes require paired full-movie quality validation',
        authorized_for_submission=False)
    contract['bundle_sha256']={p.name:sha(p) for p in out.iterdir() if p.is_file()}
    (out/'CONTRACT.json').write_text(json.dumps(contract,indent=2))
    archive=out.with_suffix('.tar.gz')
    with tarfile.open(archive,'w:gz') as tar:
        for p in sorted(out.iterdir()):tar.add(p,arcname=p.name,recursive=False)
    result=dict(status='staged',contract_sha256=sha(out/'CONTRACT.json'),archive_sha256=sha(archive),
                archive_bytes=archive.stat().st_size,smoke_frames=8,full_movies=4,maximum_full_seconds=3600,
                no_weights_changed=True,source_model_ensemble_unchanged=True,submission_performed=False)
    (ROOT/'reports/experiments/trajectory-amp-encode-v1-build.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
