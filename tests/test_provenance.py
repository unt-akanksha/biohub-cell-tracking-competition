from __future__ import annotations

import subprocess

from biohub_tracker.kaggle import KaggleRunner
from biohub_tracker.provenance import (
    ProvenanceClass,
    audit_source_tree,
    classify_notebook,
    load_policy,
)


def policies():
    return (
        load_policy("policies/notebook_audits.json"),
        load_policy("policies/metric_hack_patterns.json"),
    )


def test_known_and_titled_metric_hacks_are_excluded():
    registry, patterns = policies()
    known = classify_notebook(
        "xiaoleilian/biohub-ct-mix-divaug", "Biohub CT", None, registry, patterns
    )
    titled = classify_notebook("someone/kernel", "My metric hack", None, registry, patterns)
    assert known.provenance == ProvenanceClass.EXPLICIT_METRIC_HACK
    assert titled.provenance == ProvenanceClass.EXPLICIT_METRIC_HACK
    assert known.disposition == titled.disposition == "excluded_metric_hack"


def test_unknown_requires_source_before_automated_no_signature(tmp_path):
    registry, patterns = policies()
    unknown = classify_notebook("new/cleanish", "ordinary title", None, registry, patterns)
    source = tmp_path / "kernel.py"
    source.write_text("print('ordinary tracking code')", encoding="utf-8")
    scanned = classify_notebook("new/cleanish", "ordinary title", [source], registry, patterns)
    assert unknown.provenance == ProvenanceClass.UNKNOWN
    assert scanned.provenance == ProvenanceClass.AUTOMATED_NO_KNOWN_SIGNATURE
    assert "not_a_clean_score_claim" in scanned.evidence


def test_curated_hash_change_invalidates_review(tmp_path):
    _, patterns = policies()
    source = tmp_path / "kernel.py"
    source.write_text("x = 1", encoding="utf-8")
    source_hash = audit_source_tree([source], patterns).source_sha256
    registry = {
        "notebooks": [
            {
                "ref": "owner/kernel",
                "classification": "reproduced_post_patch",
                "disposition": "research_candidate",
                "reviewed_sha256": source_hash,
                "score_reproduced": True,
            }
        ]
    }
    valid = classify_notebook("owner/kernel", "title", [source], registry, patterns)
    source.write_text("x = 2", encoding="utf-8")
    changed = classify_notebook("owner/kernel", "title", [source], registry, patterns)
    assert valid.provenance == ProvenanceClass.REPRODUCED_POST_PATCH
    assert changed.provenance == ProvenanceClass.UNKNOWN
    assert changed.evidence == "unknown_source_changed"


def test_source_audit_never_executes_notebook_or_python(tmp_path):
    registry, patterns = policies()
    marker = tmp_path / "EXECUTED"
    source = tmp_path / "kernel.py"
    source.write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('bad')", encoding="utf-8"
    )
    notebook = tmp_path / "kernel.ipynb"
    notebook.write_text(
        '{"cells":[{"cell_type":"code","source":["raise RuntimeError(\\"executed\\")"]}]}',
        encoding="utf-8",
    )
    result = classify_notebook("owner/kernel", "ordinary", [tmp_path], registry, patterns)
    assert result.source_sha256
    assert not marker.exists()


def test_kernel_pull_uses_validated_argument_array(tmp_path):
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return subprocess.CompletedProcess(command, 0, "pulled", "")

    target = KaggleRunner(run_process=fake_run).pull_kernel_source(
        "owner/kernel", tmp_path / "cache"
    )
    assert target == (tmp_path / "cache" / "owner" / "kernel").resolve()
    assert observed["command"][:4] == ["kaggle", "kernels", "pull", "owner/kernel"]
    assert observed["kwargs"]["shell"] is False
