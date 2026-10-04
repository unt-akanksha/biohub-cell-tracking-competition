"""Immutable additional-fitting raw FOCUS cache; no diagnostic reuse for fitting."""
import ast
import hashlib
import json
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.focus_extra_fit_scope import scope
RUN='focus-extra-fit-cache-v1';SLUG='biohub-'+RUN
PARENT_SHA='efd1cdef0e762c8ed7bb3ce045365dfbfd854c04a3dbe79e0ef99d969a20a8bd'


def build():
    policy=scope((ROOT/'research/independent_real_baseline_v1_split.json').read_bytes())
    parent=ROOT/'kaggle/biohub-focus-adaptation-cache-v1/biohub-focus-adaptation-cache-v1.ipynb'
    if hashlib.sha256(parent.read_bytes()).hexdigest()!=PARENT_SHA:raise ValueError('Exact completed eight-movie detector implementation required')
    receipt=runpy.run_path(str(ROOT/'scripts/verify-focus-adaptation-cache.py'))['verify'](ROOT/'.biohub/cache/kernel-outputs/focus-adaptation-cache-v1')
    if receipt!=json.loads((ROOT/'reports/experiments/focus-adaptation-cache-v1-result.json').read_text()):raise ValueError('Actual prior detector cache must verify')
    nb=json.loads(parent.read_text());old=nb['metadata']['codex']['contract']
    before_all=repr(old['replay_stems']+old['training_stems']);after_all=repr(policy['replay_stems']+policy['training_stems'])
    all_count=subset_count=0
    for cell in nb['cells']:
        source=''.join(cell['source']);all_count+=source.count(before_all);source=source.replace(before_all,after_all)
        subset_count+=source.count(repr(old['training_stems']));source=source.replace(repr(old['training_stems']),repr(policy['training_stems']))
        source=source.replace('focus-adaptation-cache-v1',RUN).replace('focus_adaptation_cache_terminal.json','focus_extra_fit_cache_terminal.json')
        cell['source']=source.splitlines(keepends=True)
        if cell['cell_type']=='code':ast.parse(source)
    if all_count!=2 or subset_count!=1:raise ValueError('Unexpected detector scope substitution counts')
    nb['cells'][0]['source']=['# Additional fitting-only raw FOCUS cache\nEight new original fitting movies plus six replay frames. Existing diagnostic set unchanged; no labels, postprocessing or submission.\n']
    nb['metadata']['codex'].update(run_id=RUN,contract=policy,parent_notebook_sha256=PARENT_SHA,declared_budget_seconds=3600)
    meta=json.loads((parent.parent/'kernel-metadata.json').read_text());meta.update(id='indarkarhana/'+SLUG,title=SLUG,code_file=SLUG+'.ipynb')
    return nb,meta


if __name__=='__main__':
    target=ROOT/'kaggle'/SLUG
    if target.exists():raise ValueError('Never overwrite staged or launched experiment')
    nb,meta=build();target.mkdir()
    (target/meta['code_file']).write_text(json.dumps(nb),encoding='utf-8')
    (target/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps(nb['metadata']['codex']['contract'],indent=2))
