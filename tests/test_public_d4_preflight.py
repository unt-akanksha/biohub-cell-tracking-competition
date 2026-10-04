import ast
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("preflight", ROOT / "research/public_d4_preflight.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_busy_gpu_rejected_without_killing():
    module.verify_idle_gpu_query("\n")
    with pytest.raises(ValueError, match="other projects"):
        module.verify_idle_gpu_query("12345\n")


def test_only_selected_definitions_extracted():
    result = module.named_definitions("raise RuntimeError('never execute')\ndef kept(): return 1\ndef ignored(): return 2", ("kept",))
    assert "ignored" not in result and "never execute" not in result
    with pytest.raises(ValueError, match="one reviewed"):
        module.named_definitions("x=1", ("missing",))


@pytest.mark.parametrize("name", ["public-predictor-original.py", "public-predictor-d4-corrected.py"])
def test_actual_encoder_only_source(name):
    path = ROOT / ".biohub/cache/public-d4-correction-v1" / name
    if not path.exists():
        pytest.skip("Source-audit cache not installed")
    source = module.encoder_only_function(path.read_text(encoding="utf-8"))
    compile(source, "encoder-only", "exec")
    tree = ast.parse(source)
    assert len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef)
    assert "model.encode(imgs)" in source and "secondary_model.encode(imgs)" in source
    assert "seen_frames = set(frame_indices)" in source
    for excluded in ("predict_edges(", "zarr_arr", "open_dataset", "submission", "subprocess", "torch.load"):
        assert excluded not in source


def test_encoder_drift_fails_closed():
    with pytest.raises(ValueError):
        module.encoder_only_function("def predict_video(): pass")
