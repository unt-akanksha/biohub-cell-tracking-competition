from __future__ import annotations

import math
from pathlib import Path
import runpy

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
VERIFY = runpy.run_path(
    str(ROOT / "scripts/verify-948tta2-unanimous-salvage-v4.py")
)


def test_aggregate_matches_notebook_formula() -> None:
    rows = pd.DataFrame(
        {
            "weight": [2.0, 3.0],
            "adjusted_edge_jaccard": [0.8, 0.9],
            "div_tp": [1, 1],
            "div_fp": [0, 1],
            "div_fn": [1, 0],
        }
    )
    result = VERIFY["aggregate"](rows)
    assert math.isclose(result["adjusted_edge_jaccard"], 0.86)
    assert math.isclose(result["division_jaccard"], 0.5)
    assert math.isclose(result["proxy_score"], 0.91)
