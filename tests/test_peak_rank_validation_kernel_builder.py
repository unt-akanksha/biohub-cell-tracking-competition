import importlib.util
import json
from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build-peak-rank-validation-kernel.py"
DEPTH_PU_SCRIPT = ROOT / "scripts" / "build-peak-rank-depth-pu-validation-kernel.py"
CAPACITY_PU_SCRIPT = ROOT / "scripts" / "build-peak-rank-capacity-pu-validation-kernel.py"
ENSEMBLE_SCRIPT = ROOT / "scripts" / "build-peak-rank-logit-ensemble-validation-kernel.py"


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


def test_depth_pu_kernel_wrapper_has_distinct_private_identity() -> None:
    wrapper = runpy.run_path(str(DEPTH_PU_SCRIPT), run_name="depth_pu_probe")
    globals_ = wrapper["module"]["main"].__globals__
    assert globals_["TARGET_ID"] == "biohub-peak-rank-depth-pu-validation-v2"
    assert globals_["RUNTIME_REF"].endswith("depth-pu-validation-runtime-v2")
    assert globals_["NOTEBOOK"].name == "biohub-peak-rank-depth-pu-validation-v2.ipynb"


def test_capacity_pu_kernel_wrapper_binds_larger_parameter_contract(tmp_path: Path) -> None:
    wrapper = runpy.run_path(str(CAPACITY_PU_SCRIPT), run_name="capacity_pu_probe")
    globals_ = wrapper["module"]["main"].__globals__
    assert globals_["TARGET_ID"] == "biohub-peak-rank-capacity-pu-validation-v3"
    assert globals_["EXPECTED_PARAMETER_COUNT"] == 66_977_670
    target = globals_["TARGET"]
    original = (globals_["TARGET"], globals_["NOTEBOOK"])
    try:
        root = tmp_path / "kernel"
        globals_["TARGET"] = root
        globals_["NOTEBOOK"] = root / "capacity.ipynb"
        globals_["main"]()
        code = (root / "capacity.ipynb").read_text(encoding="utf-8")
        assert "66977670" in code
        assert "__EXPECTED_PARAMETER_COUNT__" not in code
    finally:
        globals_["TARGET"], globals_["NOTEBOOK"] = original


def test_logit_ensemble_kernel_wrapper_binds_total_parameter_contract() -> None:
    wrapper = runpy.run_path(str(ENSEMBLE_SCRIPT), run_name="ensemble_probe")
    globals_ = wrapper["module"]["main"].__globals__
    assert globals_["TARGET_ID"] == "biohub-peak-rank-logit-ensemble-validation-v4"
    assert globals_["EXPECTED_PARAMETER_COUNT"] == 76_762_956
