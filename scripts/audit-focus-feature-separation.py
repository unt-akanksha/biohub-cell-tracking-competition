"""Use only four fitting movies to assess image-feature discrimination."""
import hashlib
import json
from pathlib import Path
import runpy
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.focus_feature_separation import compare,summarize
RUN='focus-feature-separation-v1'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    started=time.monotonic();target=ROOT/f'reports/experiments/{RUN}-result.json'
    if target.exists():raise ValueError('Never overwrite an observed diagnostic')
    frozen={p:sha(ROOT/p) for p in ('research/focus_feature_separation.py','scripts/audit-focus-feature-separation.py','reports/experiments/focus-feature-separation-v1-design.md')}
    receipt_path=ROOT/'reports/experiments/focus-adaptation-features-v1-result.json'
    receipt=json.loads(receipt_path.read_text())
    if receipt!=runpy.run_path(str(ROOT/'scripts/verify-focus-adaptation-features.py'))['verify'](ROOT/'.biohub/cache/kernel-outputs/focus-adaptation-features-v1'):
        raise ValueError('Exact verified feature cache required')
    fit=ROOT/'reports/experiments/focus-motion-residual-fit-v1.json'
    if sha(fit)!='fcef805eabaa616102a4588437d0ea7cdcd5ef1cb7a7eeaa6d77138a7834c066':raise ValueError('Frozen training-only motion calibration required')
    parameters=json.loads(fit.read_text())['parameters'];records=[];all_values=[]
    output=ROOT/'.biohub/cache/kernel-outputs/focus-adaptation-features-v1/focus_adaptation_features/outputs'
    for stem in receipt['contract']['fitting_stems']:
        manifest=json.loads((output/stem/'manifest.json').read_text());rows=[]
        for pair in manifest['pairs']:
            path=output/stem/pair['file']
            if sha(path)!=pair['sha256']:raise ValueError('Frozen fitting feature packet changed')
            with np.load(path,allow_pickle=False) as data:packet={k:data[k].copy() for k in data.files}
            rows.append(compare(packet,parameters))
        values=np.concatenate(rows);all_values.append(values);records.append(dict(stem=stem,**summarize(values)))
    result=dict(status='completed_fitting_only_feature_separation_audit',run_id=RUN,records=records,
        pooled=summarize(np.concatenate(all_values)),source_hashes=frozen,feature_receipt_sha256=sha(receipt_path),
        diagnostic_representation_statistics_computed=False,source_selection_opened=False,new_target_movies_opened=0,
        predictions_changed=False,optimizer_run=False,gpu_seconds=0,authorized_for_submission=False,
        elapsed_seconds=time.monotonic()-started,
        caveat='Pairwise cosine/L2 versus closest incorrect physical candidate; not the transformer, tracking metric, or proof of feature collapse')
    if any(sha(ROOT/p)!=v for p,v in frozen.items()):raise ValueError('Audit implementation changed')
    target.write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
