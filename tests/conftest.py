from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def competition_config() -> dict:
    with Path("config/competition.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture
def fixture_dir() -> Path:
    return Path("tests/fixtures/kaggle")

