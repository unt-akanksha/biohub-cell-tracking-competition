"""Same exact pruned inference, with one BLAS thread in this process only."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    base=ROOT/'scripts/audit-trajectory-event-fast-runtime-v1.py'
    assert sha(base)=='428235e82a204bd79f8384c15740ae95238911998bd4a32bd92f4111896899b0'
    benchmark=ROOT/'reports/experiments/trajectory-event-dominance-v1-audit.json';proof=json.loads(benchmark.read_text())
    assert proof['status']=='source_dominance_benchmark_complete' and proof['all_reduced_optimal']
    assert sha(ROOT/'research/trajectory_event_dominance_v1.py')==proof['reduction_sha256']
    source=base.read_text(encoding='utf-8')
    source=source.replace('trajectory_event_fast_inference_v1','trajectory_event_pruned_inference_v1')
    source=source.replace('trajectory-event-fast-runtime-v1','trajectory-event-pruned-runtime-v2')
    old="fast_inference_sha256=sha(ROOT/'research/trajectory_event_pruned_inference_v1.py'),model_sha256=sha(model))"
    assert source.count(old)==1
    source=source.replace(old,"fast_inference_sha256=sha(ROOT/'research/trajectory_event_pruned_inference_v1.py'),\n                blas_thread_limit=1,dominance_benchmark_sha256="+repr(sha(benchmark))+",model_sha256=sha(model))")
    namespace=dict(__name__='runtime_replay_library',__file__=__file__)
    exec(compile(source,str(base),'exec'),namespace)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1,user_api='blas'):namespace['main']()


if __name__=='__main__':main()
