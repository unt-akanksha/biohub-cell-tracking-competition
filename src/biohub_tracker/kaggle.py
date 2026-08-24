from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence


_SECRET_PATTERNS = (
    (re.compile(r"(?i)(token|api[_ -]?key|password)\s*[:=]\s*\S+"), r"\1=[REDACTED]"),
    (re.compile(r"(?i)authorization:\s*\S+"), "authorization: [REDACTED]"),
)
_KERNEL_REF = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def redact_diagnostic(message: str) -> str:
    redacted = message
    for pattern, replacement in _SECRET_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted[:1000]


class KaggleCommandError(RuntimeError):
    """A read-only Kaggle command failed without exposing authentication details."""


def extract_json_payload(text: str) -> Any:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        return value
    raise ValueError("Kaggle output did not contain a JSON object or array")


RunFunction = Callable[..., subprocess.CompletedProcess[str]]


@dataclass
class KaggleRunner:
    executable: str = "kaggle"
    run_process: RunFunction = subprocess.run

    def run(self, args: Sequence[str]) -> subprocess.CompletedProcess[str]:
        command = [self.executable, *[str(item) for item in args]]
        completed = self.run_process(
            command,
            shell=False,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            detail = redact_diagnostic(completed.stderr or completed.stdout)
            raise KaggleCommandError(f"Kaggle read failed (exit {completed.returncode}): {detail}")
        return completed

    def run_json(self, args: Sequence[str]) -> Any:
        completed = self.run(args)
        try:
            return extract_json_payload(completed.stdout)
        except ValueError as exc:
            detail = redact_diagnostic(completed.stderr)
            raise KaggleCommandError(f"Kaggle returned invalid JSON: {detail}") from exc

    def pull_kernel_source(self, ref: str, cache_root: Path) -> Path:
        if not _KERNEL_REF.fullmatch(ref):
            raise ValueError(f"invalid Kaggle kernel reference: {ref!r}")
        owner, slug = ref.split("/", 1)
        root = cache_root.resolve()
        target = (root / owner / slug).resolve()
        if root != target and root not in target.parents:
            raise ValueError("kernel cache path escaped its configured root")
        target.mkdir(parents=True, exist_ok=True)
        self.run(["kernels", "pull", ref, "--metadata", "-p", str(target)])
        return target


@dataclass
class FixtureRunner:
    fixture_dir: Path

    def run_json(self, args: Sequence[str]) -> Any:
        name = fixture_name(args)
        path = self.fixture_dir / f"{name}.json"
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)


def fixture_name(args: Sequence[str]) -> str:
    values = list(args)
    if values and values[0] == "quota":
        return "quota"
    if "submissions" in values:
        return "submissions"
    if "leaderboard" in values:
        return "leaderboard"
    if "topics" in values:
        return "topics"
    if values[:2] == ["kernels", "list"]:
        return "kernels"
    raise KeyError(f"no fixture mapping for Kaggle arguments: {values!r}")
