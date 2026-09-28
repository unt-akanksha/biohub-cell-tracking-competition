"""One evidence-isolated correction to the public LF-DCTTA inference geometry.

This does not import or execute downloaded notebook code. Literal replacement
tables are read with ast.literal_eval and applied to a SHA-pinned support file.
The final anti-diagonal pass must be R180 followed by transpose, not R90 then
transpose (which is merely the already-used horizontal flip).
"""
from __future__ import annotations

import ast
import copy
import hashlib


NOTEBOOK_SHA256 = "95f08bb82388e9206f45c86daf926bb3d7a7fff9889c89b5fa9596f8e13c3bd4"
SUPPORT_SHA256 = "c44e771ba5980b820f93091e03a303c25dfe8f3232e501f54dc9565731c234b9"
_PATCH_PAIRS = (
    ("_old", "_new"),
    ("_guard_old", "_guard_new"),
    ("_bi_old", "_bi_new"),
    ("_coordinate_manifest_old", "_coordinate_manifest_new"),
    ("_et_old", "_et_new"),
    ("_secondary_tta_old", "_secondary_tta_new"),
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def cell_text(notebook: dict, index: int) -> str:
    return "".join(notebook["cells"][index]["source"])


def _literal_assignments(source: str) -> dict:
    values = {}
    for node in ast.parse(source).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (TypeError, ValueError):
            continue
        if target.id in values:
            raise ValueError(f"Duplicate literal assignment: {target.id}")
        values[target.id] = value
    return values


def materialize_public_predictor(notebook: dict, support_bytes: bytes) -> str:
    """Replay only the known source replacements; never eval/exec public code."""
    if sha256(support_bytes) != SUPPORT_SHA256:
        raise ValueError("Support predictor SHA256 mismatch")
    values = _literal_assignments(cell_text(notebook, 4))
    source = support_bytes.decode("utf-8")

    def replace(old: str, new: str) -> None:
        nonlocal source
        if not isinstance(old, str) or not isinstance(new, str) or source.count(old) != 1:
            raise ValueError("Public source patch anchor is not unique")
        source = source.replace(old, new, 1)

    replace(values["_old"], values["_new"])
    if len(values["_ensemble_replacements"]) != 6:
        raise ValueError("Public ensemble patch table changed")
    for old, new in values["_ensemble_replacements"]:
        replace(old, new)
    for old, new in _PATCH_PAIRS[1:]:
        replace(values[old], values[new])
    compile(source, "public-predictor-source-only", "exec")
    return source


def _is_rot90(node: ast.AST) -> bool:
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "rot90" and isinstance(node.func.value, ast.Name)
            and node.func.value.id in ("torch", "torch_mod"))


def correct_antidiagonal(source: str, *, predictor: bool) -> tuple[str, list[dict]]:
    """Edit just rotation arguments in the final public TTA view and its inverse.

    Other rotations, models, logits, weights, thresholds, topology and public
    post-processing remain byte-for-byte unchanged. Unexpected counts fail.
    """
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line.encode("utf-8")))
    data = source.encode("utf-8")
    edits = []
    forward_names = {"imgs_at", "secondary_imgs_at"} if predictor else {"at"}
    inverse_names = {"det_at", "_u_at", "secondary_det_at", "_secondary_u_at"}
    for node in ast.walk(tree):
        forward = False
        call = None
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in forward_names for t in node.targets
        ):
            outer = node.value
            if (isinstance(outer, ast.Call) and isinstance(outer.func, ast.Attribute)
                    and outer.func.attr == "transpose" and _is_rot90(outer.func.value)):
                call = outer.func.value
                forward = True
        elif _is_rot90(node) and node.args:
            first = node.args[0]
            if (isinstance(first, ast.Call) and isinstance(first.func, ast.Attribute)
                    and first.func.attr == "transpose"):
                payload = first.func.value
                if predictor:
                    if isinstance(payload, ast.Subscript):
                        payload = payload.value
                    if isinstance(payload, ast.Name) and payload.id in inverse_names:
                        call = node
                elif (isinstance(payload, ast.Call) and isinstance(payload.func, ast.Name)
                      and payload.func.id == "model" and len(payload.args) == 1
                      and isinstance(payload.args[0], ast.Name) and payload.args[0].id == "at"):
                    call = node
        if call is None:
            continue
        if len(call.args) != 2 or ast.literal_eval(call.args[1]) != (1 if forward else -1):
            raise ValueError("Anti-diagonal rotation contract changed or already corrected")
        dims = [kw for kw in call.keywords if kw.arg == "dims"]
        if len(dims) != 1 or ast.literal_eval(dims[0].value) != (-2, -1):
            raise ValueError("Expected XY rotation")
        arg = call.args[1]
        start = offsets[arg.lineno - 1] + arg.col_offset
        end = offsets[arg.end_lineno - 1] + arg.end_col_offset
        edits.append(dict(start=start, end=end, value="2" if forward else "-2",
                          direction="forward" if forward else "inverse", line=arg.lineno))
    counts = {kind: sum(e["direction"] == kind for e in edits) for kind in ("forward", "inverse")}
    expected = {"forward": 2, "inverse": 4} if predictor else {"forward": 1, "inverse": 1}
    if counts != expected or len({e["start"] for e in edits}) != len(edits):
        raise ValueError(f"Unexpected anti-diagonal sites: {counts}, expected {expected}")
    for edit in sorted(edits, key=lambda e: e["start"], reverse=True):
        data = data[:edit["start"]] + edit["value"].encode() + data[edit["end"]:]
    result = data.decode("utf-8")
    compile(result, "corrected-public-d4-source-only", "exec")
    return result, sorted(edits, key=lambda e: e["start"])


def geometry_check() -> dict:
    """Independent NumPy group/inverse/equivariance proof, with no data/labels."""
    import numpy as np

    def transform(x, view, fixed):
        if view == 0:
            return x
        if view < 4:
            return np.flip(x, {1: (-1,), 2: (-2,), 3: (-2, -1)}[view])
        if view < 6:
            return np.rot90(x, {4: 1, 5: 3}[view], axes=(-2, -1))
        if view == 6:
            return x.swapaxes(-1, -2)
        return np.rot90(x, 2 if fixed else 1, axes=(-2, -1)).swapaxes(-1, -2)

    def inverse(x, view, fixed):
        if view < 4 or view == 6:
            return transform(x, view, fixed)
        if view < 6:
            return np.rot90(x, -{4: 1, 5: 3}[view], axes=(-2, -1))
        return np.rot90(x.swapaxes(-1, -2), -2 if fixed else -1, axes=(-2, -1))

    def model(x):
        # Deliberately asymmetric local predictor; the group average, not the
        # individual predictor, must commute with a spatial symmetry.
        return 2 * np.roll(x, 1, -1) - np.roll(x, 1, -2) + x * x / 100

    def average(x, fixed):
        return sum(inverse(model(transform(x, v, fixed)), v, fixed) for v in range(8)) / 8

    records = []
    for shape in ((3, 4, 4), (2, 3, 5), (1, 2, 3, 6, 6)):
        grid = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
        unique = {}
        errors = {}
        for fixed in (False, True):
            key = "corrected" if fixed else "legacy"
            views = [transform(grid, v, fixed) for v in range(8)]
            unique[key] = len({(a.shape, a.tobytes()) for a in views})
            if any(not np.array_equal(inverse(a, v, fixed), grid) for v, a in enumerate(views)):
                raise ValueError("A view did not invert exactly")
            result = average(grid, fixed)
            errors[key] = max(float(np.max(np.abs(
                inverse(average(transform(grid, v, True), fixed), v, True) - result
            ))) for v in range(8))
        if unique != {"legacy": 7, "corrected": 8} or errors["corrected"] > 1e-10 or errors["legacy"] <= 1e-10:
            raise ValueError("D4 independent geometry proof failed")
        records.append(dict(shape=list(shape), unique_views=unique, equivariance_max_error=errors))
    return dict(passed=True, records=records, model_calls_per_arm=8,
                labels_opened=False, gpu_seconds=0, quality_gain_established=False)


def corrected_notebook(notebook: dict, legacy: str, corrected: str, dc_source: str) -> dict:
    """Source-only research artifact, not an approved or auto-submitted notebook."""
    output = copy.deepcopy(notebook)
    source = cell_text(output, 4)
    anchor = 'print("secondary edge-feature TTA patch installed and enabled", flush=True)'
    if source.count(anchor) != 1:
        raise ValueError("Correction must run after all public feature patches, before inference")
    install = (
        "\n\n# Project-authored complete-D4 geometry correction; no extra model passes.\n"
        "import hashlib as _d4_hashlib\n"
        "_d4_before = _ps.read_bytes()\n"
        f"if _d4_hashlib.sha256(_d4_before).hexdigest() != {sha256(legacy.encode())!r}:\n"
        "    raise RuntimeError('Public D4 predictor source drift before correction')\n"
        f"_d4_corrected_source = {corrected!r}\n"
        "compile(_d4_corrected_source, str(_ps), 'exec')\n"
        "_ps.write_text(_d4_corrected_source, encoding='utf-8')\n"
        f"if _d4_hashlib.sha256(_ps.read_bytes()).hexdigest() != {sha256(corrected.encode())!r}:\n"
        "    raise RuntimeError('D4 predictor correction did not persist')\n"
        "print('PROJECT_D4_CORRECTION: eight unique views; same eight passes', flush=True)\n"
    )
    output["cells"][4]["source"] = source.replace(anchor, anchor + install, 1).splitlines(keepends=True)
    output["cells"][5]["source"] = dc_source.splitlines(keepends=True)
    for cell in output["cells"]:
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
            compile("".join(cell["source"]), "source-only-candidate-cell", "exec")
    return output
