"""One bounded successor diagnostic after exact full-training completion."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
OWNED=Path('/tmp/biohub-image-context-v2.ScdSdY')
EXPECTED_CONTRACT='bda044511d3f37b33caa76d5f226fa64c4542d2705d304749e60f95c52869593'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    training=OWNED/'native-correspondence-v2-training-full'
    destination=OWNED/'native-correspondence-v2-cross-embryo-result.json'
    receipt=OWNED/'native-correspondence-v2-cross-embryo-queue-result.json'
    if destination.exists() or receipt.exists(): raise ValueError('Preserve existing diagnostic')
    manifest=json.loads((ROOT/'SUCCESSOR.json').read_text())
    for row in manifest['files']:
        if sha(ROOT/row['path'])!=row['sha256']: raise ValueError('Successor code changed')
    if sha(ROOT/'TRAINING.json')!=EXPECTED_CONTRACT: raise ValueError('Unexpected training contract')
    begin=time.monotonic(); result=dict(status='waiting_for_exact_training',training_contract_sha256=EXPECTED_CONTRACT,pid=os.getpid())
    try:
        while time.monotonic()-begin<5700:
            terminal=training/'RESULT.json'
            if terminal.exists():
                state=json.loads(terminal.read_text())
                if state['status']!='training_complete' or state['training_contract_sha256']!=EXPECTED_CONTRACT:
                    result.update(status='not_launched_training_failed',training_status=state['status']); break
                active=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=10).stdout
                gpu_pids={int(line) for line in active.splitlines() if line.strip()}
                training_pid=json.loads((OWNED/'native-correspondence-v2-full-training-launch.json').read_text())['pid']
                if gpu_pids=={training_pid}:
                    time.sleep(5); continue
                if gpu_pids:
                    result['status']='not_launched_foreign_gpu_occupied'; break
                log=OWNED/'native-correspondence-v2-cross-embryo.log'
                with log.open('xb') as output:
                    completed=subprocess.run(['/home/ubuntu/venv/bin/python',str(ROOT/'scripts/evaluate-native-correspondence-v2.py'),
                        '--data','/dev/shm/biohub-native-correspondence-v2-full','--training',str(training),
                        '--output',str(destination)],stdout=output,stderr=subprocess.STDOUT,timeout=300)
                result.update(status='diagnostic_completed' if completed.returncode==0 else 'diagnostic_failed',returncode=completed.returncode)
                if destination.exists(): result['diagnostic_sha256']=sha(destination)
                break
            time.sleep(30)
        else: result['status']='not_launched_training_wait_timeout'
    except Exception as error:
        result.update(status='failed',error_type=type(error).__name__)
    result['elapsed_seconds']=time.monotonic()-begin
    receipt.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result),flush=True)


if __name__=='__main__':main()
