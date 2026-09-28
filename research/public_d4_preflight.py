"""Extract reviewed encoder-only public code for a label-free GPU preflight.

No notebook cells, dependency installers, scorers, graph repairs, or submission
writers are executed. The caller must hash-check all source files beforehand.
"""
from __future__ import annotations

import ast
import copy


def named_definitions(source: str, names: tuple[str, ...]) -> str:
    tree = ast.parse(source)
    result = []
    for name in names:
        found = [n for n in ast.walk(tree)
                 if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name]
        if len(found) != 1:
            raise ValueError(f"Expected one reviewed definition: {name}")
        result.append(copy.deepcopy(found[0]))
    module = ast.fix_missing_locations(ast.Module(body=result, type_ignores=[]))
    return "from __future__ import annotations\n" + ast.unparse(module) + "\n"


def encoder_only_function(source: str) -> str:
    tree = ast.parse(source)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "predict_video"]
    if len(functions) != 1:
        raise ValueError("Expected one public predict_video")
    loops = [n for n in functions[0].body if isinstance(n, ast.For)
             and isinstance(n.target, ast.Name) and n.target.id == "ws"]
    if len(loops) != 1:
        raise ValueError("Expected one inference window loop")
    body = loops[0].body
    start = [i for i, n in enumerate(body) if isinstance(n, ast.Assign)
             and ast.unparse(n.targets[0]) == "(unet_out, det_logits)"]
    stop = [i for i, n in enumerate(body) if isinstance(n, ast.Delete)
            and [ast.unparse(t) for t in n.targets] == ["imgs"]]
    if len(start) != 1 or len(stop) != 1 or stop[0] - start[0] != 4:
        raise ValueError("Public encoder block structure changed")
    template = ast.parse(
        "def public_encoder_pass(imgs, model, secondary_model):\n"
        "    W = imgs.shape[1]\n"
        "    frame_indices = list(range(W))\n"
        "    seen_frames = set(frame_indices)  # suppress only public disk logging\n"
        "    secondary_detection_weight = 0.80\n"
        "    return unet_out, secondary_unet_out, det_logits\n"
    ).body[0]
    template.body[-1:-1] = copy.deepcopy(body[start[0]:stop[0]])
    module = ast.fix_missing_locations(ast.Module(body=[template], type_ignores=[]))
    return ast.unparse(module) + "\n"


def verify_idle_gpu_query(text: str) -> None:
    """Only a previously freed GPU may be used; never stop another process."""
    if text.strip():
        raise ValueError("GPU has compute processes; leave other projects untouched")
