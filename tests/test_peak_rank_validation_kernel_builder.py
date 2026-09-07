import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build-peak-rank-validation-kernel.py"


def test_builder_creates_private_two_gpu_non_submitting_kernel(tmp_path, monkeypatch) -> None:
    spec = importlib.util.spec_from_file_location("peak_rank_kernel_builder", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    target = tmp_path / "kernel"
    monkeypatch.setattr(module, "TARGET", target)
    monkeypatch.setattr(module, "NOTEBOOK", target / "candidate.ipynb")
    module.main()

    metadata = json.loads((target / "kernel-metadata.json").read_text())
    notebook = json.loads((target / "candidate.ipynb").read_text())
    source = "".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is True
    assert metadata["enable_tpu"] is False
    assert metadata["enable_internet"] is False
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert "torch.cuda.device_count() != 2" in source
    assert '"--devices", "0,1"' in source
    assert "selection_passed" in source
    assert "acceptance_opened" in source
    assert "submission.csv" in source
    assert "kaggle competitions submit" not in source
    assert "api.competition_submit" not in source


def test_builder_keeps_full_runtime_watchdog() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"declared_budget_seconds": 43200' in source
    assert '"hard_stop_seconds": 42000' in source
    assert "threading.Timer(42000" in source
