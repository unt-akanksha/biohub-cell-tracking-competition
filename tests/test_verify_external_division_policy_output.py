from __future__ import annotations

import hashlib
import json
from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
VERIFIER = ROOT / "scripts/verify-external-division-policy-output.py"


def test_atomic_copy_is_idempotent_only_for_same_policy(tmp_path: Path) -> None:
    module = runpy.run_path(str(VERIFIER))
    source = tmp_path / "source.json"
    destination = tmp_path / "published.json"
    source.write_text('{"status":"accepted"}', encoding="utf-8")

    module["atomic_copy"](source, destination)
    module["atomic_copy"](source, destination)

    assert hashlib.sha256(destination.read_bytes()).hexdigest() == hashlib.sha256(
        source.read_bytes()
    ).hexdigest()


def test_atomic_copy_rejects_a_different_existing_policy(tmp_path: Path) -> None:
    module = runpy.run_path(str(VERIFIER))
    source = tmp_path / "source.json"
    destination = tmp_path / "published.json"
    source.write_text(json.dumps({"threshold": 1}), encoding="utf-8")
    destination.write_text(json.dumps({"threshold": 2}), encoding="utf-8")

    try:
        module["atomic_copy"](source, destination)
    except FileExistsError as error:
        assert "different external policy" in str(error)
    else:
        raise AssertionError("different policy unexpectedly overwrote destination")
