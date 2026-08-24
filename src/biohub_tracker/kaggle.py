from __future__ import annotations

import csv
import io
import json
import re
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence


_SECRET_PATTERNS = (
    (re.compile(r"(?i)(token|api[_ -]?key|password)\s*[:=]\s*\S+"), r"\1=[REDACTED]"),
    (re.compile(r"(?i)authorization:\s*\S+"), "authorization: [REDACTED]"),
)
_KERNEL_REF = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_COMPETITION_SLUG = re.compile(r"^[A-Za-z0-9_.-]+$")


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

    def kernel_status(self, ref: str) -> str:
        if not _KERNEL_REF.fullmatch(ref):
            raise ValueError(f"invalid Kaggle kernel reference: {ref!r}")
        completed = self.run(["kernels", "status", ref])
        match = re.search(r'has status\s+"(?:KernelWorkerStatus\.)?([A-Za-z_]+)"', completed.stdout)
        if not match:
            raise KaggleCommandError("Kaggle kernel status output was ambiguous")
        return match.group(1).upper()

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

    def download_leaderboard(self, competition_slug: str) -> list[dict[str, str]]:
        if not _COMPETITION_SLUG.fullmatch(competition_slug):
            raise ValueError(f"invalid competition slug: {competition_slug!r}")
        with tempfile.TemporaryDirectory(prefix="biohub-leaderboard-") as temporary_dir:
            directory = Path(temporary_dir)
            self.run(
                [
                    "competitions",
                    "leaderboard",
                    competition_slug,
                    "--download",
                    "--path",
                    str(directory),
                    "--quiet",
                ]
            )
            archives = sorted(directory.glob("*.zip"))
            csv_files = sorted(directory.glob("*.csv"))
            if archives:
                with zipfile.ZipFile(archives[0]) as archive:
                    members = sorted(
                        name for name in archive.namelist() if name.lower().endswith(".csv")
                    )
                    if not members:
                        raise KaggleCommandError("downloaded leaderboard archive contained no CSV")
                    with archive.open(members[0]) as handle:
                        text = io.TextIOWrapper(handle, encoding="utf-8-sig", newline="")
                        return list(csv.DictReader(text))
            if csv_files:
                with csv_files[0].open("r", encoding="utf-8-sig", newline="") as handle:
                    return list(csv.DictReader(handle))
            raise KaggleCommandError("Kaggle leaderboard download produced no CSV or ZIP")


@dataclass
class FixtureRunner:
    fixture_dir: Path

    def run_json(self, args: Sequence[str]) -> Any:
        name = fixture_name(args)
        path = self.fixture_dir / f"{name}.json"
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def kernel_status(self, ref: str) -> str:
        if not _KERNEL_REF.fullmatch(ref):
            raise ValueError(f"invalid Kaggle kernel reference: {ref!r}")
        path = self.fixture_dir / "kernel_status.json"
        with path.open("r", encoding="utf-8") as handle:
            values = json.load(handle)
        if not isinstance(values, dict) or ref not in values:
            raise KaggleCommandError(f"fixture has no kernel status for {ref}")
        return str(values[ref]).removeprefix("KernelWorkerStatus.").upper()


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
    if "pages" in values:
        return "pages"
    if values[:2] == ["kernels", "list"]:
        return "kernels"
    raise KeyError(f"no fixture mapping for Kaggle arguments: {values!r}")
