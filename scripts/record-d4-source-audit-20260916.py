#!/usr/bin/env python
"""Record the 2026-09-16 source and discussion review in the audit registry.

This writes an attestation, so it only records review that actually happened and
it refuses to write a hash that was not independently confirmed against the
bytes the competition watch pulled this session.

Reviewed this session:
  * flexonafft/biohub-harmonic-fusion            raw ipynb e378e723...fb44a
  * sjlee101/biohub-lf-dctta020-sectta1-sister16 raw ipynb 998b7bc9...9cdb

Both were materialized through the pinned public patch chain, diffed against
each other and against the 2026-09-10 lf-dctta reference, and their eight-pass
D4 augmentation group was confirmed to contain only seven unique views. Twenty
notebooks were statically screened by the same watch.

Discussions read in full this session: 741242, 741651, 741446. The recent topic
list was enumerated; older topics carried over from the previous attestation.
"""
from __future__ import annotations

import glob
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "policies" / "notebook_audits.json"
CACHE = ROOT / ".biohub" / "cache" / "notebooks"

REVIEWED = {
    "flexonafft/biohub-harmonic-fusion": {
        "raw_ipynb_sha256": "e378e723ff30c3bebbe21b68553b3c666c4592141288f32529c1322b790fb44a",
        "source_sha256": "c0c6e449c4d13fa5d02326d630623eef7038cac8132e1eaad7f9105e17fe39d3",
        "evidence": (
            "2026-09-16 source review of raw notebook e378e723...fb44a. Materialized the "
            "public predictor through the pinned patch chain and confirmed it is "
            "configuration-identical to the 2026-09-10 lf-dctta reference on all three "
            "constants. Confirmed the eight-pass D4 group builds its anti-diagonal view as "
            "rot90(x,1).transpose(), which equals the horizontal flip already in the list, so "
            "the group carries seven unique views. No known metric-hack signature. It remains "
            "a leaderboard-reported public-lineage comparator that declares "
            "leaderboard_feedback_used_for_configuration=True; its advertised score is not "
            "project validation or selection evidence, and its in-notebook proxy sweep is not "
            "our selection procedure."
        ),
    },
    "sjlee101/biohub-lf-dctta020-sectta1-sister16": {
        "raw_ipynb_sha256": "998b7bc99c7aabf89a1a9b82719406f310d55a22e535316e89f40f973d2e9cdb",
        "source_sha256": "63317832392a6d618ad73e12e43cbfbe00aabfb65b838b4b6ace76a11df62bc6",
        "evidence": (
            "2026-09-16 source review of raw notebook 998b7bc9...9cdb. Same materialized "
            "predictor and same three attached CC0 model datasets as biohub-harmonic-fusion; "
            "differs only in SAFE_DIV_SISTER_MAX_UM 16.0, DEEPCENTER_SAFE_DIV_THRESHOLD 0.20 "
            "and SECONDARY_EDGE_FEATURE_TTA_WEIGHT 1.0. Carries the same seven-unique-view D4 "
            "defect. No known metric-hack signature. Public-lineage comparator that declares "
            "leaderboard_feedback_used_for_configuration=True; not an independent model family "
            "and not selection evidence."
        ),
    },
}

NEW_DISCUSSIONS = ["741242", "741651", "741446"]


def _confirm_pulled_bytes() -> None:
    for ref, spec in REVIEWED.items():
        owner, slug = ref.split("/", 1)
        matches = glob.glob(str(CACHE / owner / slug / "*.ipynb"))
        if len(matches) != 1:
            raise SystemExit(f"Expected exactly one pulled notebook for {ref}, found {matches}")
        digest = hashlib.sha256(Path(matches[0]).read_bytes()).hexdigest()
        if digest != spec["raw_ipynb_sha256"]:
            raise SystemExit(
                f"Refusing to attest {ref}: the watch pulled {digest}, but the review "
                f"covered {spec['raw_ipynb_sha256']}."
            )


def main() -> int:
    _confirm_pulled_bytes()
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    before = {row["ref"]: dict(row) for row in registry["notebooks"]}
    by_ref = {row["ref"]: row for row in registry["notebooks"]}

    for ref, spec in REVIEWED.items():
        row = by_ref.get(ref)
        if row is None:
            row = {"ref": ref}
            registry["notebooks"].append(row)
            by_ref[ref] = row
        row["classification"] = "reported_post_patch"
        row["disposition"] = "comparator_not_independent_candidate"
        row["evidence"] = spec["evidence"]
        row["reviewed_sha256"] = spec["source_sha256"]
        row["score_reproduced"] = False

    refs = list(registry.get("discussion_refs", []))
    for topic in NEW_DISCUSSIONS:
        if topic not in refs:
            refs.append(topic)
    registry["discussion_refs"] = refs

    previous = registry["audited_at"]
    registry["audited_at"] = now
    registry["discussions_audited_at"] = now

    REGISTRY.write_text(
        json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )

    print(f"audited_at             {previous}  ->  {now}")
    print(f"discussions_audited_at {previous}  ->  {now}")
    print(f"discussion_refs        +{NEW_DISCUSSIONS}")
    for ref in REVIEWED:
        old = before.get(ref, {}).get("reviewed_sha256")
        print(f"{ref}")
        print(f"  reviewed_sha256      {old or '(new entry)'}")
        print(f"                   ->  {by_ref[ref]['reviewed_sha256']}")
    print(f"notebooks in registry  {len(registry['notebooks'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
