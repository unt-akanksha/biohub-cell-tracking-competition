"""Verified engineering retry: unchanged source, objective, epochs and seed."""
import ctypes
from ctypes import wintypes
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    base=ROOT/'scripts/train-trajectory-event-source-pilot-v1.py'
    assert sha(base)=='de9232bcdf8e0a2a606976377ad90b7bd432c9829f2ad04f9539bd81930381b4'
    previous=ROOT/'reports/experiments/trajectory-event-source-pilot-v1.json'
    old=json.loads(previous.read_text());assert old['status']=='timeout'
    proof_path=ROOT/'reports/experiments/trajectory-event-fast-training-v1-audit.json'
    proof=json.loads(proof_path.read_text())
    assert proof['status']=='real_fast_training_functionality_passed' and proof['exact_saved_case_reconstructions']==673
    assert proof['cache_items']==673 and proof['cache_bytes']<1536*1024**2
    for name,digest in proof['modules_sha256'].items():assert sha(ROOT/'research'/name)==digest
    class MemoryStatus(ctypes.Structure):
        _fields_=[('length',wintypes.DWORD),('load',wintypes.DWORD)]+[(n,ctypes.c_ulonglong) for n in
            ('total_phys','available_phys','total_page','available_page','total_virtual','available_virtual','extended')]
    memory=MemoryStatus();memory.length=ctypes.sizeof(memory)
    assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory))
    assert memory.available_phys>=4*1024**3,'Preserve memory for other local work'
    source=base.read_text(encoding='utf-8')
    source=source.replace('trajectory-event-source-pilot-v1','trajectory-event-source-pilot-v2')
    source=source.replace('from research.trajectory_event_case_store_v1 import CaseStore',
                          'from research.trajectory_event_fast_case_store_v1 import CaseStore')
    source=source.replace('from research.trajectory_event_training_v1 import prior,fit',
                          'from research.trajectory_event_training_v1 import prior\nfrom research.trajectory_event_fast_training_v1 import fit')
    source=source.replace('store=DeadlineStore(cases_root,records,cache_size=2)',
                          'store=DeadlineStore(cases_root,records,cache_size=len(records),max_cache_bytes=1536*1024**2)')
    old_marker="'trajectory_event_assignment_v1.py','trajectory_event_case_store_v1.py')"
    assert source.count(old_marker)==1
    source=source.replace(old_marker,"'trajectory_event_assignment_v1.py','trajectory_event_case_store_v1.py',\n            'trajectory_event_fast_case_store_v1.py','trajectory_event_fast_training_v1.py','trajectory_event_dominance_v1.py')")
    marker='no_source_score_based_epoch_selection=True,selection_or_validation_used_for_training=False,'
    assert source.count(marker)==1
    source=source.replace(marker,marker+'\n        blas_thread_limit=1,cache_limit_bytes=1536*1024**2,available_memory_before='+str(memory.available_phys)+',\n        previous_timeout_sha256='+repr(sha(previous))+',fast_training_proof_sha256='+repr(sha(proof_path))+',\n        engineering_retry_not_hyperparameter_search=True,')
    namespace=dict(__name__='fixed_source_training_retry',__file__=__file__)
    exec(compile(source,str(base),'exec'),namespace)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1,user_api='blas'):namespace['main']()


if __name__=='__main__':main()
