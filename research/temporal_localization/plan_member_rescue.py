"""Choose the next fixed temporal-localizer rescue seed without score ranking."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


RUN_ID = "synthetic256-real-replay-temporal-node-localizer-v2"
EXPECTED_PARAMETER_COUNT = 71_249_805
MINIMUM_MEMBERS = 3
MAXIMUM_MEMBERS = 4
ORIGINAL_SEEDS = (41_021, 41_029, 41_039, 41_047)
RESCUE_SEEDS = (41_057, 41_063, 41_071, 41_081, 41_087, 41_099)
SEED_PATTERN = re.compile(r"member_[0-9]+_seed_([0-9]+)$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def accepted_member(terminal_path: Path) -> dict[str, Any] | None:
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    checkpoint = terminal_path.parent / "localization_model.pt"
    if terminal.get("status") != "completed":
        return None
    if not (
        terminal.get("schema_version") == 1
        and terminal.get("run_id") == RUN_ID
        and terminal.get("parameter_count") == EXPECTED_PARAMETER_COUNT
        and terminal.get("selection_gate_passed") is True
        and terminal.get("audit_gate_passed") is True
        and terminal.get("division_critical_selection_gate_passed") is True
        and terminal.get("division_critical_audit_gate_passed") is True
        and terminal.get("real_selection_gate_passed") is True
        and terminal.get("real_audit_gate_passed") is True
        and terminal.get("real_division_critical_selection_gate_passed") is True
        and terminal.get("real_division_critical_audit_gate_passed") is True
        and terminal.get("serialized_checkpoint_selection_gate_passed") is True
        and terminal.get("checkpoint_frozen_before_audit") is True
        and terminal.get("audit_opened") is True
        and checkpoint.is_file()
        and terminal.get("model_sha256") == sha256_file(checkpoint)
        and terminal.get("real_replay_probability") == 0.25
        and terminal.get("competition_train_data_read") is True
        and terminal.get("competition_test_data_read") is False
        and terminal.get("public_code_copied") is False
        and terminal.get("public_predictions_copied") is False
        and terminal.get("public_leaderboard_used_for_selection") is False
        and terminal.get("submission_created") is False
    ):
        raise ValueError(f"completed rescue member evidence changed: {terminal_path}")
    return {
        "seed": int(terminal["seed"]),
        "model_sha256": str(terminal["model_sha256"]),
        "terminal_path": terminal_path.as_posix(),
    }


def attempted_seeds(results_root: Path) -> set[int]:
    seeds: set[int] = set()
    for member_root in results_root.glob("gpu_*/member_*_seed_*"):
        match = SEED_PATTERN.fullmatch(member_root.name)
        if match:
            seeds.add(int(match.group(1)))
    return seeds


def next_gpu_index(results_root: Path) -> int:
    indices = []
    for path in results_root.glob("gpu_*"):
        try:
            indices.append(int(path.name.removeprefix("gpu_")))
        except ValueError:
            continue
    return max(indices, default=-1) + 1


def plan_rescue(results_root: Path) -> dict[str, Any]:
    results_root = results_root.resolve()
    if not results_root.is_dir():
        raise FileNotFoundError(results_root)
    terminals = sorted(results_root.glob("gpu_*/member_*/worker_terminal.json"))
    accepted = [
        member
        for terminal in terminals
        if (member := accepted_member(terminal)) is not None
    ]
    accepted_seeds = [row["seed"] for row in accepted]
    accepted_hashes = [row["model_sha256"] for row in accepted]
    if len(accepted_seeds) != len(set(accepted_seeds)):
        raise ValueError("accepted temporal-localizer seeds are not unique")
    if len(accepted_hashes) != len(set(accepted_hashes)):
        raise ValueError("byte-identical temporal-localizer checkpoints cannot both vote")
    if len(accepted) > MAXIMUM_MEMBERS:
        raise ValueError(f"too many accepted temporal-localizer members: {len(accepted)}")

    attempted = attempted_seeds(results_root)
    unknown_rescue = sorted((attempted - set(ORIGINAL_SEEDS)) - set(RESCUE_SEEDS))
    if unknown_rescue:
        raise ValueError(f"uncommitted rescue seeds are present: {unknown_rescue}")
    remaining = [seed for seed in RESCUE_SEEDS if seed not in attempted]
    if len(accepted) >= MINIMUM_MEMBERS:
        status = "sufficient_members"
        next_seed = None
        output_index = None
    elif remaining:
        status = "train_next_fixed_seed"
        next_seed = remaining[0]
        output_index = next_gpu_index(results_root)
    else:
        status = "fixed_rescue_pool_exhausted"
        next_seed = None
        output_index = None

    return {
        "schema_version": 1,
        "run_id": "temporal-localizer-member-rescue-policy-v1",
        "status": status,
        "accepted_member_count": len(accepted),
        "accepted_members": accepted,
        "minimum_members": MINIMUM_MEMBERS,
        "maximum_members": MAXIMUM_MEMBERS,
        "original_seeds": list(ORIGINAL_SEEDS),
        "fixed_rescue_seed_order": list(RESCUE_SEEDS),
        "attempted_rescue_seeds": sorted(attempted & set(RESCUE_SEEDS)),
        "next_seed": next_seed,
        "next_gpu_index": output_index,
        "member_or_policy_ranking_performed": False,
        "public_leaderboard_used_for_selection": False,
        "authorized_for_submission": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = plan_rescue(args.results_root)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".partial")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(args.output)


if __name__ == "__main__":
    main()
