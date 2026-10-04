"""Verify actual CPU smoke source and results before any real GPU training."""
import ast
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
WRAPPER_SHA='2a4572f7a7f2de7c1ea147796514260d0f3bd98fb25e4bb6004a40f3b61bbbb2'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    target=ROOT/'reports/experiments/focus-null-loss-cpu-v1-result.json'
    if target.exists():raise ValueError('Never overwrite verified smoke evidence')
    wrapper=ROOT/'kaggle/biohub-focus-null-loss-cpu-v1/run.py'
    if sha(wrapper)!=WRAPPER_SHA:raise ValueError('Exact launched CPU script required')
    sources=ast.literal_eval(next(n.value for n in ast.parse(wrapper.read_text()).body
        if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='SOURCES'))
    folder=ROOT/'.biohub/cache/kernel-outputs/focus-null-loss-cpu-v1';runtime=folder/'focus_null_loss_runtime'
    if any((runtime/p).read_bytes()!=v.encode() for p,v in sources.items()):raise ValueError('Actual CPU runtime changed')
    if json.loads((folder/'focus_null_loss_source_hashes.json').read_text())!={p:hashlib.sha256(v.encode()).hexdigest() for p,v in sources.items()}:
        raise ValueError('Actual runtime source identities differ')
    terminal=json.loads((folder/'focus_null_loss_terminal.json').read_text());result=json.loads((runtime/'result.json').read_text())
    if (terminal['status']!='completed' or terminal['error'] is not None or terminal['gpu_used'] is not False
        or terminal['declared_budget_seconds']!=900 or not 0<terminal['elapsed_seconds']<=900):raise ValueError('Complete bounded CPU-only terminal required')
    if (result['status']!='passed_cpu_null_weighted_loss_smoke' or result['null_weight']!=math.sqrt(10754/161)
        or result['unit_weight_loss_delta']!=0 or result['empty_source_null_loss']!=0
        or not 0<=result['synthetic_final']<result['synthetic_initial']
        or any(result[k] is not True for k in ('unknown_gradient_exact_zero','known_null_gradient_direction','same_parent_daughters_supported','integer_labels_guarded'))
        or any(result[k] is not False for k in ('cuda_initialized','real_training_performed','authorized_for_submission'))):
        raise ValueError('Required loss/gradient/scope checks did not pass')
    receipt=dict(status='verified_cpu_null_weighted_loss_smoke',wrapper_sha256=WRAPPER_SHA,worker_result_sha256=sha(runtime/'result.json'),
        source_hashes={p:sha(runtime/p) for p in sources},terminal=terminal,worker=result,
        real_gpu_four_step_smoke_still_required=True,authorized_for_submission=False)
    target.write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
