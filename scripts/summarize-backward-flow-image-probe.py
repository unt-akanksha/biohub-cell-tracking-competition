"""Paired100step sparse/image supervision evidence, not a tracking candidate."""
import ast
import hashlib
import json
from pathlib import Path
import runpy
ROOT=Path(__file__).resolve().parents[1]
BASE=runpy.run_path(str(ROOT/'scripts/summarize-backward-flow-probe.py'))
CONTROL_RESULT_SHA='4dddfaf9f8bb14e90056e0ecf6dd3b65dfb4ed2fa001d8f9ae3ff3f12f2feb38'


def compare(result,terminal,split_bytes,control,control_terminal):
    report=BASE['verify'](result,terminal,split_bytes,run_id='backward-flow-image-probe-v1')
    baseline=BASE['verify'](control,control_terminal,split_bytes)
    if (result['identity'].get('loss_profile')!=dict(image_weight=.25,smoothness_weight=.01)
        or control['checkpoint_sha256']!='c1203ffb03ac98b4aa5b332ec26b4469ba5b9e784e6e095f03a41c533bba5884'
        or result['input_hashes']!=control['input_hashes'] or result['before']!=control['before']
        or report['parameters']!=baseline['parameters'] or report['fitting_windows']!=baseline['fitting_windows']
        or report['diagnostic_links']!=baseline['diagnostic_links']):
        raise ValueError('Exact frozen sparse-probe inputs, initialization and diagnostic scope required')
    delta={key:report['errors'][key]-baseline['errors'][key] for key in ('mae_um','endpoint_um')}
    passed=report['small_fit_gate_passed'] and all(v<0 for v in delta.values())
    report.update(status='verified_image_motion_probe_not_tracking_validation',
        errors_vs_sparse_probe=delta,input_replay_identical=True,
        decision='paired_probe_gain_requires_full_fit_and_tracking_validation' if passed else 'no_paired_probe_gain',
        image_profile=result['identity']['loss_profile'],control_checkpoint_sha256=control['checkpoint_sha256'])
    return report


if __name__=='__main__':
    cache=ROOT/'.biohub/cache/kernel-outputs/backward-flow-image-probe-v1/backward_flow_image_probe'
    original=ROOT/'.biohub/cache/kernel-outputs/backward-flow-probe-v1/backward_flow_probe'
    paths=dict(result=cache/'outputs/result.json',terminal=cache/'launcher_terminal.json',
        control=original/'outputs/result.json',control_terminal=original/'launcher_terminal.json')
    if hashlib.sha256(paths['control'].read_bytes()).hexdigest()!=CONTROL_RESULT_SHA:
        raise ValueError('Original sparse control result hash mismatch')
    values={k:json.loads(p.read_text()) for k,p in paths.items()}
    report=compare(split_bytes=(ROOT/'research/independent_real_baseline_v1_split.json').read_bytes(),**values)
    notebook=ROOT/'kaggle/biohub-backward-flow-image-probe-v1/biohub-backward-flow-image-probe-v1.ipynb'
    source=''.join(json.loads(notebook.read_text())['cells'][1]['source'])
    for name,filename in [('sources','source_hashes.json'),('runtime_sources','runtime_hashes.json')]:
        assignment=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
            and isinstance(n.targets[0],ast.Name) and n.targets[0].id==name)
        expected={k:hashlib.sha256(v.encode()).hexdigest() for k,v in ast.literal_eval(assignment.value).items()}
        if json.loads((cache/filename).read_text())!=expected:
            raise ValueError('Immutable image-probe source hashes differ')
    report['source_sha256']={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()}
    report['notebook_sha256']=hashlib.sha256(notebook.read_bytes()).hexdigest()
    (ROOT/'reports/experiments/backward-flow-image-probe-v1-result.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
