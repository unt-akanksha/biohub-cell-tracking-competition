"""Local CPU fallback for the rejected remote scorer upload; no GPU or push."""
import hashlib
import json
from pathlib import Path
import runpy
import time

ROOT=Path(__file__).resolve().parents[1]
RUN='owned-detector-ensemble-selection-v1'


def main():
    notebook=ROOT/f'kaggle/biohub-{RUN}/biohub-{RUN}.ipynb'
    expected='5dde4d126d150fb9c861df451e2b813f361b052c6bcaed93948df3aa4ee03bc1'
    if hashlib.sha256(notebook.read_bytes()).hexdigest()!=expected:
        raise ValueError('Frozen ensemble inference notebook changed')
    root=ROOT/f'.biohub/cache/kernel-outputs/{RUN}/owned_detector_ensemble_selection'
    output=ROOT/f'.biohub/cache/local-evaluations/{RUN}'
    output.mkdir(parents=True,exist_ok=True)
    with (output/'started.json').open('x') as handle:
        json.dump(dict(platform='local_cpu',notebook_sha256=expected),handle)
    scorer_path=ROOT/'scripts/score-independent-selection.py'
    started=time.monotonic()
    print('Scoring all eight saved graphs locally with pinned official metric',flush=True)
    result=runpy.run_path(str(scorer_path))['score'](root,notebook,
        ROOT/'.biohub/cache/competition-train-geffs-packed-v1/train')
    score_path=output/'selection_score.json'
    score_path.write_text(json.dumps(result,indent=2))
    execution_path=output/'execution.json'
    execution_path.write_text(json.dumps(dict(status='completed',platform='local_cpu',
        elapsed_seconds=time.monotonic()-started,notebook_sha256=expected,
        scorer_wrapper_sha256=hashlib.sha256(scorer_path.read_bytes()).hexdigest(),
        score_sha256=hashlib.sha256(score_path.read_bytes()).hexdigest(),
        remote_cpu_upload='HTTP400; exact remote absence verified; no scorer version created',
        gpu_used=False,submission_performed=False),indent=2))
    runpy.run_path(str(ROOT/'scripts/summarize-owned-detector-ensemble-selection.py'))['summarize'](score_path,execution_path)


if __name__=='__main__': main()
