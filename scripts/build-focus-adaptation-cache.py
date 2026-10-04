"""Extend the verified FOCUS inference to eight predetermined training movies."""
import ast
import hashlib
import json
from pathlib import Path
import runpy

ROOT=Path(__file__).resolve().parents[1]
RUN='focus-adaptation-cache-v1'
SLUG='biohub-'+RUN
PARENT_SHA='d65eac4ce6354829b13fd92176c3163b45a50efabe2c326f5b5322d7d73f322b'


def build():
    payload=(ROOT/'research/independent_real_baseline_v1_split.json').read_bytes()
    policy=runpy.run_path(str(ROOT/'research/focus_adaptation_cache_contract.py'))['scope'](payload)
    parent=ROOT/'kaggle/biohub-focus-source-cache-v1/biohub-focus-source-cache-v1.ipynb'
    if hashlib.sha256(parent.read_bytes()).hexdigest()!=PARENT_SHA:raise ValueError('Exact completed full-cache implementation required')
    probe=runpy.run_path(str(ROOT/'scripts/verify-focus-source-probe.py'))['verify'](ROOT/'.biohub/cache/kernel-outputs/focus-source-probe-v1')
    if probe!=json.loads((ROOT/'reports/experiments/focus-source-probe-v1-result.json').read_text()):raise ValueError('Successful actual detector smoke required')
    nb=json.loads(parent.read_text());old=json.loads(payload)['folds'][0]['selection']
    before_all=repr(policy['replay_stems']+old);after_all=repr(policy['replay_stems']+policy['training_stems'])
    changed_all=changed_subset=0
    for cell in nb['cells']:
        text=''.join(cell['source'])
        changed_all+=text.count(before_all);text=text.replace(before_all,after_all)
        changed_subset+=text.count(repr(old));text=text.replace(repr(old),repr(policy['training_stems']))
        text=text.replace('focus-source-cache-v1',RUN).replace('focus_source_cache_terminal.json','focus_adaptation_cache_terminal.json')
        text=text.replace('selection_stems','training_stems').replace("'source_selection'","'training_cache'")
        cell['source']=text.splitlines(keepends=True)
        if cell['cell_type']=='code':ast.parse(text)
    if changed_all!=2 or changed_subset!=1:raise ValueError(f'Expected exact scope substitutions, got {changed_all}/{changed_subset}')
    nb['cells'][0]['source']=['# FOCUS adaptation training cache\nFour fitting/four diagnostic original training movies plus six replay frames. No source-selection/target images or labels. No postprocessing or submission.\n']
    nb['metadata']['codex'].pop('source_stems',None)
    nb['metadata']['codex'].update(run_id=RUN,contract=policy,parent_notebook_sha256=PARENT_SHA,declared_budget_seconds=3600)
    meta=json.loads((parent.parent/'kernel-metadata.json').read_text())
    meta.update(id='indarkarhana/'+SLUG,title=SLUG,code_file=SLUG+'.ipynb')
    return nb,meta


if __name__=='__main__':
    target=ROOT/'kaggle'/SLUG
    if target.exists():raise ValueError('Never overwrite staged or launched cache')
    nb,meta=build();target.mkdir()
    (target/meta['code_file']).write_text(json.dumps(nb))
    (target/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(nb['metadata']['codex']['contract'],indent=2))
