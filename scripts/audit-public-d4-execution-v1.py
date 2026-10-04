"""Persist actual extracted-code CPU execution evidence; no CUDA/labels."""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
from public_d4_numpy_execution import encoder_check, deepcenter_check


if __name__ == "__main__":
    started = time.monotonic()
    output = ROOT / "reports/experiments/public-d4-execution-v1-audit.json"
    if output.exists():
        raise ValueError("Do not overwrite completed execution evidence")
    cache = ROOT / ".biohub/cache/public-d4-correction-v1"
    result = dict(status="extracted_public_arithmetic_verified", run_id="public-d4-execution-v1",
                  encoders=encoder_check(cache), deepcenter=deepcenter_check(cache),
                  labels_opened=False, gpu_seconds=0, trained_models_executed=False,
                  peak_counting_stubbed=True, quality_gain_established=False,
                  authorized_for_submission=False,
                  source_sha256=hashlib.sha256((ROOT/"research/public_d4_numpy_execution.py").read_bytes()).hexdigest(),
                  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  elapsed_seconds=time.monotonic()-started)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
