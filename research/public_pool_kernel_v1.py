"""Detection non-maximum-suppression radius for the public predictor.

WHY

The support-pack predictor extracts detections as local maxima of the heatmap
using a 3D max-pool whose kernel is sized in microns:

    pool_kernel_um: float = 3.0   # max-pool kernel size in um for peak extraction

Two peaks more than 3 um apart therefore both survive. Measured against the
ground truth that is far below the real spacing between cells. Taking cell counts
from `estimated_number_of_nodes` and the imaged volume implied by the geff axis
scales (z 1.625, y/x 0.40625), mean centre-to-centre spacing runs about 11 um in
the densest movies we see and about 26 um in the sparsest. A 3 um suppression
radius is smaller than a single nucleus, so a heatmap with any internal structure
can emit several peaks on one cell.

Those duplicates land inside the scorer's 7 um matching radius AND inside the
5.5-7.25 um motion-relink gate, which is exactly the impostor-candidate geometry
behind our dominant error: false-positive edges, correlating -0.929 with adjusted
edge Jaccard and scaling about fivefold with density. An independent competitor
notebook (noisyislands/biohub-linker-association-mlp) reports the same thing from
its own forensics -- "the production detector creates duplicates (~3 um offset)"
-- and trains against synthetic duplicates to compensate.

WHAT THIS IS NOT

This does not claim duplication is the whole story. Our predicted node count runs
at 0.926 of the estimated truth, i.e. we UNDER-predict overall, which pure
duplication would not produce. The likely picture is both at once: a high
det_threshold (0.965) missing real cells while duplicates survive on bright ones.
That is why the interesting experiment is the pair -- a lower det_threshold to
recover real cells together with a larger suppression radius to drop duplicates --
since the two push node count in opposite directions.

MECHANISM

The constant is not reachable from the environment and has no CLI flag, so it is
changed in the predictor source itself, using the same materialise-edit-embed
path as the D4 correction: hash the shipped source, rewrite exactly one literal,
write it back inside the kernel and re-hash. Drift on either side fails closed.
"""
from __future__ import annotations

import copy

from pathlib import Path
import importlib.util

_SPEC = importlib.util.spec_from_file_location(
    "public_d4_correction", Path(__file__).resolve().parent / "public_d4_correction.py"
)
_d4 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_d4)

sha256 = _d4.sha256

# What the support pack ships.
POOL_KERNEL_DEFAULT = 3.0

# Upper bound is set by the real geometry, not by taste: mean cell spacing is
# about 11 um in the densest movies, so a suppression radius at or above that
# would start merging genuinely distinct cells. 9.0 keeps a margin.
POOL_KERNEL_RANGE = (3.0, 9.0)

_LINE = "    pool_kernel_um: float = 3.0"
_ANCHOR = 'print("secondary edge-feature TTA patch installed and enabled", flush=True)'


def patch_source(source: str, pool_kernel_um: float) -> str:
    """Rewrite exactly one literal. Fails closed on drift or on a no-op."""
    if not POOL_KERNEL_RANGE[0] < pool_kernel_um <= POOL_KERNEL_RANGE[1]:
        raise ValueError(
            f"pool_kernel_um={pool_kernel_um} outside the audited range "
            f"{POOL_KERNEL_RANGE}; above it the radius approaches real cell spacing"
        )
    if source.count(_LINE) != 1:
        raise ValueError("pool_kernel_um default is not present exactly once")
    patched = source.replace(
        _LINE, f"    pool_kernel_um: float = {float(pool_kernel_um)!r}", 1
    )
    compile(patched, "pool-kernel-patched", "exec")

    # Exactly one line may differ, and only in that literal.
    before, after = source.splitlines(), patched.splitlines()
    if len(before) != len(after):
        raise ValueError("pool kernel patch changed the line count")
    diff = [(a, b) for a, b in zip(before, after) if a != b]
    # The shipped line carries a trailing comment, so match on the assignment
    # prefix rather than the whole line; the comment must survive untouched.
    if len(diff) != 1 or not diff[0][0].startswith(_LINE):
        raise ValueError(f"pool kernel patch touched {len(diff)} lines, expected 1")
    old_line, new_line = diff[0]
    if old_line.split("#", 1)[1:] != new_line.split("#", 1)[1:]:
        raise ValueError("pool kernel patch altered the trailing comment")
    return patched


def install(notebook: dict, support_bytes: bytes, pool_kernel_um: float) -> tuple[dict, dict]:
    """Embed the patched predictor with hash guards on both sides."""
    legacy = _d4.materialize_public_predictor(notebook, support_bytes)
    patched = patch_source(legacy, pool_kernel_um)

    output = copy.deepcopy(notebook)
    source = _d4.cell_text(output, 4)
    if source.count(_ANCHOR) != 1:
        raise ValueError("Pool-kernel patch must run after the public feature patches")
    if "PROJECT_D4_CORRECTION" in source:
        # Both rewrite the same file; combining them would make the second
        # overwrite the first's hash guard.
        raise ValueError("Pool-kernel patch cannot be combined with the D4 correction")

    block = (
        "\n\n# Project-authored detection suppression radius; no extra model passes.\n"
        "import hashlib as _pk_hashlib\n"
        "_pk_before = _ps.read_bytes()\n"
        f"if _pk_hashlib.sha256(_pk_before).hexdigest() != {sha256(legacy.encode())!r}:\n"
        "    raise RuntimeError('Public predictor source drift before pool-kernel patch')\n"
        f"_pk_patched_source = {patched!r}\n"
        "compile(_pk_patched_source, str(_ps), 'exec')\n"
        "_ps.write_text(_pk_patched_source, encoding='utf-8')\n"
        f"if _pk_hashlib.sha256(_ps.read_bytes()).hexdigest() != {sha256(patched.encode())!r}:\n"
        "    raise RuntimeError('Pool-kernel patch did not persist')\n"
        f"print('PROJECT_POOL_KERNEL: detection suppression radius "
        f"{POOL_KERNEL_DEFAULT} -> {float(pool_kernel_um)} um', flush=True)\n"
    )
    output["cells"][4]["source"] = source.replace(
        _ANCHOR, _ANCHOR + block, 1
    ).splitlines(keepends=True)

    report = {
        "pool_kernel_um": float(pool_kernel_um),
        "shipped_default": POOL_KERNEL_DEFAULT,
        "audited_range": list(POOL_KERNEL_RANGE),
        "legacy_predictor_sha256": sha256(legacy.encode()),
        "patched_predictor_sha256": sha256(patched.encode()),
        "lines_changed": 1,
        "targets": "duplicate detections inside the 7um match radius and the relink gate",
    }
    return output, report
