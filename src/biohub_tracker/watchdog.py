from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from .io import atomic_replace_json, atomic_write_json, utc_now


Callback = Callable[[], Any]


class BudgetExpired(RuntimeError):
    def __init__(self, terminal: dict[str, Any]):
        super().__init__("declared runtime budget reached its safety margin")
        self.terminal = terminal


def _utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class BudgetWatchdog:
    run_id: str
    declared_budget_seconds: float
    output_dir: Path
    safety_margin_seconds: float = 300.0
    monotonic: Callable[[], float] = time.monotonic
    wall_clock: Callable[[], datetime] = utc_now
    _start_monotonic: float = field(init=False)
    _start_wall: datetime = field(init=False)
    _checkpoints: list[tuple[str, Callback]] = field(default_factory=list, init=False)
    _flushes: list[tuple[str, Callback]] = field(default_factory=list, init=False)
    _terminal: dict[str, Any] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.output_dir = self.output_dir.resolve()
        if not self.run_id.strip():
            raise ValueError("watchdog run_id is required")
        if self.declared_budget_seconds <= 0:
            raise ValueError("declared watchdog budget must be positive")
        if self.safety_margin_seconds < 0 or self.safety_margin_seconds >= self.declared_budget_seconds:
            raise ValueError("watchdog safety margin must be nonnegative and below the budget")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._start_monotonic = self.monotonic()
        self._start_wall = self.wall_clock().astimezone(timezone.utc)

    @property
    def heartbeat_path(self) -> Path:
        return self.output_dir / "watchdog-heartbeat.json"

    @property
    def terminal_path(self) -> Path:
        return self.output_dir / "watchdog-terminal.json"

    @property
    def deadline(self) -> datetime:
        return self._start_wall + timedelta(seconds=self.declared_budget_seconds)

    def register_checkpoint(self, name: str, callback: Callback) -> None:
        self._register(self._checkpoints, name, callback)

    def register_flush(self, name: str, callback: Callback) -> None:
        self._register(self._flushes, name, callback)

    @staticmethod
    def _register(target: list[tuple[str, Callback]], name: str, callback: Callback) -> None:
        normalized = str(name).strip()
        if not normalized or not callable(callback):
            raise ValueError("watchdog callback requires a name and callable")
        if any(existing == normalized for existing, _ in target):
            raise ValueError(f"duplicate watchdog callback: {normalized}")
        target.append((normalized, callback))

    def _timing(self) -> tuple[float, float]:
        elapsed = max(0.0, self.monotonic() - self._start_monotonic)
        remaining = max(0.0, self.declared_budget_seconds - elapsed)
        return elapsed, remaining

    def _base_payload(self, status: str) -> dict[str, Any]:
        elapsed, remaining = self._timing()
        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "status": status,
            "started_at": _utc_text(self._start_wall),
            "declared_deadline": _utc_text(self.deadline),
            "declared_budget_seconds": self.declared_budget_seconds,
            "safety_margin_seconds": self.safety_margin_seconds,
            "elapsed_seconds": elapsed,
            "remaining_seconds": remaining,
            "checkpoint_results": [],
            "flush_results": [],
            "callback_errors": [],
        }

    def pulse(self) -> dict[str, Any]:
        if self._terminal is not None:
            return self._terminal
        heartbeat = self._base_payload("running")
        atomic_replace_json(self.heartbeat_path, heartbeat)
        if heartbeat["remaining_seconds"] <= self.safety_margin_seconds:
            terminal = self.shutdown("budget_exhausted")
            raise BudgetExpired(terminal)
        return heartbeat

    def shutdown(self, status: str = "stopped") -> dict[str, Any]:
        if self._terminal is not None:
            return self._terminal
        if self.terminal_path.exists():
            import json

            with self.terminal_path.open("r", encoding="utf-8") as handle:
                existing = json.load(handle)
            if existing.get("run_id") != self.run_id:
                raise RuntimeError("existing watchdog terminal belongs to another run")
            self._terminal = existing
            return existing
        terminal = self._base_payload(str(status))
        errors: list[dict[str, str]] = []
        for field_name, callbacks in (
            ("checkpoint_results", self._checkpoints),
            ("flush_results", self._flushes),
        ):
            for name, callback in callbacks:
                try:
                    result = callback()
                    terminal[field_name].append({"name": name, "ok": True, "result": repr(result)[:1000]})
                except Exception as exc:  # callbacks are isolated so every cleanup gets a chance
                    message = f"{type(exc).__name__}: {exc}"[:1000]
                    terminal[field_name].append({"name": name, "ok": False, "error": message})
                    errors.append({"name": name, "error": message})
        terminal["callback_errors"] = errors
        terminal["terminal_at"] = _utc_text(self.wall_clock())
        atomic_write_json(self.terminal_path, terminal)
        self._terminal = terminal
        return terminal


def register_checkpoint(watchdog: BudgetWatchdog, name: str, callback: Callback) -> None:
    watchdog.register_checkpoint(name, callback)


def register_flush(watchdog: BudgetWatchdog, name: str, callback: Callback) -> None:
    watchdog.register_flush(name, callback)
