#!/usr/bin/env python
"""Build the complete-D4 Kaggle candidate from a pinned public base.

CPU only. Writes a kernel directory and an immutable build manifest. It does not
push, launch, score or submit anything; the guarded runbook still owns that.

    python scripts/build-d4-complete-v1.py --base harmonic --run-mode production
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "public_d4_complete_v1", ROOT / "research" / "public_d4_complete_v1.py"
)
_builder = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_builder)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="harmonic", choices=sorted(_builder.APPROVED_BASES))
    parser.add_argument("--run-mode", default="production", choices=_builder.RUN_MODES)
    parser.add_argument("--validator-n-per-type", type=int, default=12)
    parser.add_argument("--tag", default=None, help="slug suffix")
    parser.add_argument("--with-density", action="store_true",
                        help="density-conditional motion-relink gates")
    parser.add_argument("--density-continuous", action="store_true",
                        help="ship tight_um on the line through the published band steps")
    parser.add_argument("--with-edge-confidence", action="store_true",
                        help="install the motion-relink minimum learned-probability floor")
    parser.add_argument("--edge-confidence-floor", type=float, default=0.0,
                        help="arm the floor; any value in (0, 0.48] is the same switch")
    parser.add_argument("--pool-kernel-um", type=float, default=None,
                        help="detection suppression radius in um (default 3.0, audited 3-9)")
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                        help="audited inference override, repeatable "
                             "(BIOHUB_DET_THRESHOLD, BIOHUB_DUAL_SEED_EDGE_THRESHOLD)")
    parser.add_argument("--no-d4", action="store_true",
                        help="build the stock base without the D4 correction (reference arm)")
    parser.add_argument("--with-recovery", action="store_true",
                        help="also install the frozen strong pruned-track recovery")
    parser.add_argument("--slug", default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    env_overrides = {}
    for item in args.env:
        if "=" not in item:
            print(f"BUILD FAILED: --env expects KEY=VALUE, got {item!r}", file=sys.stderr)
            return 1
        k, v = item.split("=", 1)
        env_overrides[k.strip()] = v.strip()

    # Kaggle derives the kernel slug from the TITLE on first push. Keep slug,
    # title and metadata id mutually consistent so the ref cannot drift.
    suffix = "-recovery" if args.with_recovery else ""
    stem = "stock" if args.no_d4 else "d4"
    if args.with_density: stem += "-density"
    if args.density_continuous: stem += "cont"
    if args.with_edge_confidence: stem += "-edgeconf"
    if args.pool_kernel_um: stem += "-pk"
    tag = f"-{args.tag}" if args.tag else ""
    slug = args.slug or f"biohub-{stem}-complete{suffix}-{args.base}-{args.run_mode}{tag}-v1"
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "kaggle" / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    notebook, manifest = _builder.build_candidate(
        base=args.base,
        run_mode=args.run_mode,
        validator_n_per_type=args.validator_n_per_type,
        with_recovery=args.with_recovery,
        with_d4=not args.no_d4,
        with_density=args.with_density,
        density_continuous=args.density_continuous,
        with_edge_confidence=args.with_edge_confidence,
        edge_confidence_floor=args.edge_confidence_floor,
        env_overrides=env_overrides,
        pool_kernel_um=args.pool_kernel_um,
    )

    checks = _builder.verify_built_notebook(notebook, manifest)
    failed = sorted(name for name, ok in checks.items() if not ok)
    if failed:
        print(f"BUILD VERIFICATION FAILED: {failed}", file=sys.stderr)
        return 1

    notebook_path = out_dir / f"{slug}.ipynb"
    # Write bytes, never write_text: Windows newline translation would make the
    # file on disk differ from the payload this manifest pins.
    # Pure ASCII, always. `kaggle kernels push` from a Windows host re-reads the
    # .ipynb with the system code page; on the first launch that corrupted the
    # embedded predictor (U+00B5 -> U+00C2 U+00B5) and the on-kernel SHA guard
    # rejected it. An ASCII-only file survives any decoder unchanged. Note this
    # must cover notebook metadata too, not just cells: one base carries an em
    # dash in its title.
    payload = (json.dumps(notebook, indent=1, ensure_ascii=True) + "\n").encode("ascii")
    if not payload.isascii():
        print("BUILD FAILED: notebook payload is not pure ASCII", file=sys.stderr)
        return 1
    notebook_path.write_bytes(payload)
    if not notebook_path.read_bytes().isascii():
        print("BUILD FAILED: written notebook is not pure ASCII", file=sys.stderr)
        return 1

    # Title must slugify back to `slug` exactly.
    title = " ".join(part.upper() if part in {"d4", "v1"} else part.capitalize()
                     for part in slug.split("-"))
    assert title.lower().replace(" ", "-") == slug, (title, slug)
    # Kaggle rejects SaveKernel with 400 Bad Request when the title exceeds 50
    # characters. Fail at build time with a clear message instead of burning a
    # single-use launch authorization on an opaque push failure.
    if len(title) > 50:
        print(f"BUILD FAILED: title is {len(title)} chars, Kaggle's limit is 50: {title!r}",
              file=sys.stderr)
        print("Pass a shorter --slug.", file=sys.stderr)
        return 1
    metadata = _builder.kernel_metadata(slug, title, base=args.base)
    metadata["code_file"] = notebook_path.name
    (out_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    manifest.update(
        {
            "slug": slug,
            "built_at": datetime.now(timezone.utc).isoformat(),
            "notebook_path": notebook_path.relative_to(ROOT).as_posix(),
            "notebook_sha256": _builder.sha256(payload),
            "verification": checks,
        }
    )
    manifest_path = out_dir / "build-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"base            {args.base}  ({manifest['base_kaggle_ref']})")
    print(f"base sha256     {manifest['base_notebook_sha256']}")
    print(f"run mode        {args.run_mode}")
    print(f"argument edits  {manifest['total_argument_edits']} "
          f"({len(manifest['predictor_edits'])} predictor + {len(manifest['deepcenter_edits'])} DeepCenter)")
    if manifest.get("pruned_track_recovery"):
        r = manifest["pruned_track_recovery"]
        print(f"recovery        installed, source {r['recovery_sha256'][:16]}..., "
              f"+{r['added_lines']} lines, reimplemented={r['logic_reimplemented']}, "
              f"retuned={r['thresholds_retuned']}")
    print(f"d4 correction   applied={manifest['d4_correction_applied']}")
    print(f"geometry proof  legacy=7 unique views, corrected=8 unique views, "
          f"inverses exact, passed={manifest['geometry_proof']['passed']}")
    print(f"verification    {len(checks)}/{len(checks)} checks passed")
    print(f"notebook        {notebook_path}")
    print(f"manifest        {manifest_path}")
    print(f"notebook sha256 {manifest['notebook_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
