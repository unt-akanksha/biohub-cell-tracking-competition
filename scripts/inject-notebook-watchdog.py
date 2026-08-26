from __future__ import annotations

import argparse
import json
from pathlib import Path


MARKER = "# Biohub quota watchdog"


def watchdog_source(run_id: str, budget_seconds: int, safety_margin_seconds: int) -> str:
    hard_stop_seconds = budget_seconds - safety_margin_seconds
    return f'''{MARKER}: {budget_seconds} s declared budget, {safety_margin_seconds} s safety margin.
import atexit as _biohub_atexit
import hashlib as _biohub_hashlib
import json as _biohub_json
import os as _biohub_os
import threading as _biohub_threading
import time as _biohub_time
from pathlib import Path as _BiohubPath

_BIOHUB_RUN_ID = {run_id!r}
_BIOHUB_STARTED = _biohub_time.monotonic()
_BIOHUB_FINISHED = False
_BIOHUB_TERMINAL = _BiohubPath("/kaggle/working/watchdog-terminal.json")

def _biohub_write_terminal(status):
    submission = _BiohubPath("/kaggle/working/submission.csv")
    payload = {{
        "run_id": _BIOHUB_RUN_ID,
        "status": status,
        "elapsed_seconds": round(_biohub_time.monotonic() - _BIOHUB_STARTED, 3),
        "declared_budget_seconds": {budget_seconds},
        "safety_margin_seconds": {safety_margin_seconds},
        "submission_exists": submission.is_file(),
    }}
    if submission.is_file():
        payload["submission_bytes"] = submission.stat().st_size
        payload["submission_sha256"] = _biohub_hashlib.sha256(submission.read_bytes()).hexdigest()
    temporary = _BIOHUB_TERMINAL.with_suffix(".tmp")
    temporary.write_text(_biohub_json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(_BIOHUB_TERMINAL)

def _biohub_abort():
    if not _BIOHUB_FINISHED:
        _biohub_write_terminal("aborted")

_biohub_atexit.register(_biohub_abort)

def _biohub_budget_expired():
    _biohub_write_terminal("budget_expired")
    _biohub_os._exit(124)

_BIOHUB_TIMER = _biohub_threading.Timer({hard_stop_seconds}, _biohub_budget_expired)
_BIOHUB_TIMER.daemon = True
_BIOHUB_TIMER.start()
print("Biohub watchdog armed: hard stop after {hard_stop_seconds} seconds.")
'''


def completion_source() -> str:
    return '''# Persist terminal evidence only after every notebook cell succeeded.
_BIOHUB_FINISHED = True
_BIOHUB_TIMER.cancel()
_biohub_write_terminal("completed")
print("Biohub watchdog terminal evidence:", _BIOHUB_TERMINAL)
'''


def code_cell(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def inject(notebook_path: Path, run_id: str, budget_seconds: int, safety_margin_seconds: int) -> None:
    if budget_seconds <= 0 or safety_margin_seconds <= 0:
        raise ValueError("budget and safety margin must be positive")
    if safety_margin_seconds >= budget_seconds:
        raise ValueError("safety margin must be smaller than the declared budget")

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = notebook.get("cells")
    if not isinstance(cells, list):
        raise ValueError("notebook cells must be a list")
    if any(MARKER in str(cell.get("source", "")) for cell in cells if isinstance(cell, dict)):
        raise ValueError("notebook already contains the Biohub watchdog")

    cells.insert(0, code_cell(watchdog_source(run_id, budget_seconds, safety_margin_seconds)))
    cells.append(code_cell(completion_source()))

    kaggle = notebook.setdefault("metadata", {}).setdefault("kaggle", {})
    kaggle["accelerator"] = "gpu"
    kaggle["isGpuEnabled"] = True
    kaggle["isInternetEnabled"] = False

    temporary = notebook_path.with_suffix(f"{notebook_path.suffix}.tmp")
    temporary.write_text(
        json.dumps(notebook, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    temporary.replace(notebook_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Inject a hard-stop watchdog into a Kaggle notebook")
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--budget-seconds", type=int, required=True)
    parser.add_argument("--safety-margin-seconds", type=int, required=True)
    args = parser.parse_args()
    inject(args.notebook, args.run_id, args.budget_seconds, args.safety_margin_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
