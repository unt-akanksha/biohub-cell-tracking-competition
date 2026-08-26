from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.public_notebook_lineage import audit_lineage, normalized_code_lines


def _notebook(path: Path, code: str) -> Path:
    path.write_text(
        json.dumps(
            {
                "cells": [
                    {"cell_type": "markdown", "source": ["ignored"]},
                    {"cell_type": "code", "source": code.splitlines(keepends=True)},
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


def test_lineage_audit_normalizes_whitespace_and_ignores_markdown(tmp_path: Path) -> None:
    left = _notebook(tmp_path / "left.ipynb", "a = 1\nb = 2\nc = 3\n")
    right = _notebook(tmp_path / "right.ipynb", "  a = 1\nb = 2\nd = 4\n")

    result = audit_lineage([left, right], cluster_threshold=0.49)

    assert result["pairs"][0]["line_jaccard"] == pytest.approx(0.5)
    assert result["pairs"][0]["shared_lineage"] is True
    assert normalized_code_lines(left) == frozenset({"a = 1", "b = 2", "c = 3"})


def test_lineage_audit_rejects_empty_and_duplicate_inputs(tmp_path: Path) -> None:
    empty = _notebook(tmp_path / "empty.ipynb", "\n")
    with pytest.raises(ValueError, match="no nonempty code"):
        normalized_code_lines(empty)
    valid = _notebook(tmp_path / "valid.ipynb", "print(1)\n")
    with pytest.raises(ValueError, match="unique"):
        audit_lineage([valid, valid])

