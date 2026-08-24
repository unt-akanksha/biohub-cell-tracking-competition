from __future__ import annotations

import importlib
import importlib.metadata
import json
import re
import secrets
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from .io import canonical_json_bytes, sha256_bytes, sha256_file


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


class ScorerVerificationError(ValueError):
    def __init__(self, reason_code: str, detail: str) -> None:
        self.reason_code = reason_code
        super().__init__(f"{reason_code}: {detail}")


def _required_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ScorerVerificationError("lock_schema_invalid", f"{name} must be an object")
    return value


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ScorerVerificationError("lock_schema_invalid", f"{name} must be non-empty text")
    return value


def _required_sha(value: Any, name: str, *, commit: bool = False) -> str:
    text = _required_text(value, name)
    pattern = _COMMIT if commit else _SHA256
    if not pattern.fullmatch(text):
        raise ScorerVerificationError("lock_schema_invalid", f"{name} has an invalid digest")
    return text


@dataclass(frozen=True)
class ScorerLock:
    schema_version: int
    organizer_repository: str
    organizer_commit: str
    patch_commit: str
    tracksdata_repository: str
    tracksdata_commit: str
    critical_files: tuple[tuple[str, str], ...]
    environment_lock_path: str
    environment_lock_sha256: str
    packages: tuple[tuple[str, str], ...]
    fixture_expected_path: str
    fixture_expected_sha256: str
    fixture_result_sha256: str
    api_symbols: tuple[str, ...]
    raw: dict[str, Any]

    @classmethod
    def from_dict(cls, value: Any) -> ScorerLock:
        root = _required_mapping(value, "lock")
        if root.get("schema_version") != 1:
            raise ScorerVerificationError("lock_schema_invalid", "unsupported schema_version")
        organizer = _required_mapping(root.get("organizer"), "organizer")
        tracksdata = _required_mapping(root.get("tracksdata"), "tracksdata")
        environment = _required_mapping(root.get("environment"), "environment")
        fixtures = _required_mapping(root.get("fixtures"), "fixtures")
        api = _required_mapping(root.get("api"), "api")
        license_value = _required_mapping(root.get("license"), "license")
        constants = _required_mapping(root.get("constants"), "constants")

        if license_value.get("spdx") != "BSD-3-Clause":
            raise ScorerVerificationError("lock_schema_invalid", "license must be BSD-3-Clause")
        if constants != {
            "adjustment_alpha": "0.1",
            "max_distance_um": "7",
            "scale_zyx_um": ["1.625", "0.40625", "0.40625"],
            "score_division_weight": "0.1",
        }:
            raise ScorerVerificationError("lock_schema_invalid", "official constants changed")

        files = _required_mapping(organizer.get("critical_files"), "critical_files")
        normalized_files: list[tuple[str, str]] = []
        for relative, digest in sorted(files.items()):
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts:
                raise ScorerVerificationError("lock_schema_invalid", "unsafe critical file path")
            normalized_files.append((relative, _required_sha(digest, f"critical_files.{relative}")))

        packages = _required_mapping(environment.get("packages"), "environment.packages")
        normalized_packages = tuple(
            sorted(
                (_required_text(name, "package name"), _required_text(version, f"package {name}"))
                for name, version in packages.items()
            )
        )
        symbols = api.get("symbols")
        required_symbols = ("evaluate", "per_sample_metrics", "summarise", "node_recall")
        if (tuple(symbols) if isinstance(symbols, list) else ()) != required_symbols:
            raise ScorerVerificationError("lock_schema_invalid", "organizer API symbol set changed")

        return cls(
            schema_version=1,
            organizer_repository=_required_text(organizer.get("repository"), "organizer.repository"),
            organizer_commit=_required_sha(organizer.get("commit"), "organizer.commit", commit=True),
            patch_commit=_required_sha(organizer.get("patch_commit"), "organizer.patch_commit", commit=True),
            tracksdata_repository=_required_text(tracksdata.get("repository"), "tracksdata.repository"),
            tracksdata_commit=_required_sha(tracksdata.get("commit"), "tracksdata.commit", commit=True),
            critical_files=tuple(normalized_files),
            environment_lock_path=_required_text(environment.get("lock_path"), "environment.lock_path"),
            environment_lock_sha256=_required_sha(environment.get("lock_sha256"), "environment.lock_sha256"),
            packages=normalized_packages,
            fixture_expected_path=_required_text(fixtures.get("expected_path"), "fixtures.expected_path"),
            fixture_expected_sha256=_required_sha(fixtures.get("expected_sha256"), "fixtures.expected_sha256"),
            fixture_result_sha256=_required_sha(
                fixtures.get("perfect_linear_result_sha256"), "fixtures.perfect_linear_result_sha256"
            ),
            api_symbols=required_symbols,
            raw=root,
        )

    @property
    def semantic_sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.raw))


@dataclass(frozen=True)
class VerifiedScorer:
    lock: ScorerLock
    lock_sha256: str
    checkout: Path
    tracksdata_checkout: Path
    module_path: Path
    tracksdata_module_path: Path
    evaluate: Callable[..., Any]
    per_sample_metrics: Callable[..., Any]
    summarise: Callable[..., Any]
    node_recall: Callable[..., Any]
    tracksdata: ModuleType


def _load_lock(path: Path) -> ScorerLock:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScorerVerificationError("lock_unreadable", str(exc)) from exc
    return ScorerLock.from_dict(value)


def _run_git(checkout: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(checkout), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ScorerVerificationError("checkout_invalid", completed.stderr.strip() or "git failed")
    return completed.stdout.strip()


def _normalized_repository(value: str) -> str:
    return value.rstrip("/").removesuffix(".git").lower()


def _verify_checkout(checkout: Path, repository: str, commit: str, *, patch_commit: str | None = None) -> None:
    if not checkout.is_dir() or not (checkout / ".git").exists():
        raise ScorerVerificationError("checkout_missing", str(checkout))
    if not secrets.compare_digest(_run_git(checkout, "rev-parse", "HEAD"), commit):
        raise ScorerVerificationError("checkout_commit_mismatch", str(checkout))
    remote = _run_git(checkout, "remote", "get-url", "origin")
    if not secrets.compare_digest(_normalized_repository(remote), _normalized_repository(repository)):
        raise ScorerVerificationError("checkout_remote_mismatch", str(checkout))
    if patch_commit is not None:
        completed = subprocess.run(
            ["git", "-C", str(checkout), "merge-base", "--is-ancestor", patch_commit, commit],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise ScorerVerificationError("patch_commit_missing", patch_commit)


def _contained_file(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError):
        return False
    return True


def _verify_clean_checkout(checkout: Path) -> None:
    if _run_git(checkout, "status", "--porcelain", "--untracked-files=all"):
        raise ScorerVerificationError("checkout_dirty", str(checkout))


def _locked_requirements(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "==" not in line:
            raise ScorerVerificationError("environment_lock_invalid", line)
        name, version = line.split("==", 1)
        result[name] = version
    return result


def _verify_packages(expected: dict[str, str]) -> None:
    for name, version in sorted(expected.items()):
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise ScorerVerificationError("dependency_missing", name) from exc
        if not secrets.compare_digest(actual, version):
            raise ScorerVerificationError(
                "dependency_version_mismatch", f"{name}: expected {version}, got {actual}"
            )


def _module_file(module: ModuleType, root: Path) -> Path:
    filename = getattr(module, "__file__", None)
    if not filename or not _contained_file(Path(filename), root):
        raise ScorerVerificationError("import_path_escape", str(filename))
    return Path(filename).resolve()


def verify_scorer_lock(
    lock_path: str | Path,
    checkout: str | Path,
    *,
    tracksdata_checkout: str | Path | None = None,
) -> VerifiedScorer:
    lock_file = Path(lock_path).resolve(strict=True)
    lock = _load_lock(lock_file)
    organizer_root = Path(checkout).resolve()
    td_root = (
        Path(tracksdata_checkout).resolve()
        if tracksdata_checkout is not None
        else organizer_root.parent / "tracksdata"
    )

    _verify_checkout(
        organizer_root,
        lock.organizer_repository,
        lock.organizer_commit,
        patch_commit=lock.patch_commit,
    )
    _verify_checkout(td_root, lock.tracksdata_repository, lock.tracksdata_commit)
    for relative, expected in lock.critical_files:
        target = organizer_root / relative
        if not target.is_file():
            raise ScorerVerificationError("source_file_missing", relative)
        actual = sha256_file(target)
        if not secrets.compare_digest(actual, expected):
            raise ScorerVerificationError("source_hash_mismatch", relative)
    _verify_clean_checkout(organizer_root)
    _verify_clean_checkout(td_root)

    workspace_root = lock_file.parent.parent
    environment_lock = workspace_root / lock.environment_lock_path
    if not environment_lock.is_file():
        raise ScorerVerificationError("environment_lock_missing", str(environment_lock))
    if not secrets.compare_digest(sha256_file(environment_lock), lock.environment_lock_sha256):
        raise ScorerVerificationError("environment_lock_hash_mismatch", str(environment_lock))
    expected_file = workspace_root / lock.fixture_expected_path
    if not expected_file.is_file():
        raise ScorerVerificationError("fixture_expected_missing", str(expected_file))
    if not secrets.compare_digest(sha256_file(expected_file), lock.fixture_expected_sha256):
        raise ScorerVerificationError("fixture_expected_hash_mismatch", str(expected_file))

    locked_packages = _locked_requirements(environment_lock)
    locked_packages.update(dict(lock.packages))
    _verify_packages(locked_packages)

    metrics_module = importlib.import_module("tracking_cellmot.metrics")
    tracksdata_module = importlib.import_module("tracksdata")
    module_path = _module_file(metrics_module, organizer_root)
    tracksdata_module_path = _module_file(tracksdata_module, td_root)
    callables: dict[str, Callable[..., Any]] = {}
    for symbol in lock.api_symbols:
        candidate = getattr(metrics_module, symbol, None)
        if not callable(candidate):
            raise ScorerVerificationError("api_symbol_missing", symbol)
        callables[symbol] = candidate

    return VerifiedScorer(
        lock=lock,
        lock_sha256=lock.semantic_sha256,
        checkout=organizer_root,
        tracksdata_checkout=td_root,
        module_path=module_path,
        tracksdata_module_path=tracksdata_module_path,
        evaluate=callables["evaluate"],
        per_sample_metrics=callables["per_sample_metrics"],
        summarise=callables["summarise"],
        node_recall=callables["node_recall"],
        tracksdata=tracksdata_module,
    )


def corroborate_live(lock: ScorerLock, *, timeout_seconds: int = 20) -> dict[str, str]:
    results: dict[str, str] = {}
    for name, repository, commit in (
        ("organizer", lock.organizer_repository, lock.organizer_commit),
        ("tracksdata", lock.tracksdata_repository, lock.tracksdata_commit),
    ):
        base = repository.removesuffix(".git").replace("https://github.com/", "https://api.github.com/repos/")
        request = urllib.request.Request(
            f"{base}/commits/{commit}",
            headers={"Accept": "application/vnd.github+json", "User-Agent": "biohub-scorer-verifier"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                value = json.load(response)
        except (OSError, json.JSONDecodeError) as exc:
            raise ScorerVerificationError("live_provenance_unavailable", f"{name}: {exc}") from exc
        remote_sha = value.get("sha") if isinstance(value, dict) else None
        if not isinstance(remote_sha, str) or not secrets.compare_digest(remote_sha, commit):
            raise ScorerVerificationError("live_commit_mismatch", name)
        results[name] = "verified"
    return results
