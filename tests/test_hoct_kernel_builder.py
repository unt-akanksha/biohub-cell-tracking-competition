from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "kaggle" / "biohub-hoct-multibackbone-probe-v1"


def test_multibackbone_notebook_has_verified_topology_cache_and_fallback() -> None:
    notebook = json.loads(
        (KERNEL / "biohub-hoct-multibackbone-probe-v1.ipynb").read_text(
            encoding="ascii"
        )
    )
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    compile(code, str(KERNEL / "biohub-hoct-multibackbone-probe-v1.ipynb"), "exec")
    assert "validated_cached_topology" in code
    assert "EXPECTED_DEEPCENTER_SHA256" in code
    assert 'report.get("processed_validation_sha256") == csv_sha256' in code
    assert 'report.get("ground_truth_read_for_postprocessing") is False' in code
    assert "No valid cached topology found; using the exact in-kernel materializer." in code
    assert 'subprocess.run(materialize_command, check=True)' in code


def test_multibackbone_kernel_attaches_only_expected_preceding_kernel() -> None:
    metadata = json.loads(
        (KERNEL / "kernel-metadata.json").read_text(encoding="ascii")
    )

    assert metadata["kernel_sources"] == [
        "indarkarhana/biohub-trackastra-raw-confidence-acceptance-v2"
    ]
    assert metadata["enable_gpu"] is True
    assert metadata["enable_tpu"] is False
    assert metadata["enable_internet"] is False
