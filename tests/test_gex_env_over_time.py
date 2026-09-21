"""Tests for GEX-env-over-time helpers pulled from the dashboard module.

The Streamlit app runs page code at import time, so we AST-extract the
pure helpers instead of importing ``oi_dashboard_app``.
"""
from __future__ import annotations

import ast
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

_DASH = Path(__file__).resolve().parents[1] / "scripts" / "oi_dashboard_app.py"
_KEEP = {
    "_classify_gex_env",
    "_gex_last_per_local_day",
    "_fmt_gex_dollars",
    "_fmt_gex_delta",
}

_NS: dict = {"date": date}
_tree = ast.parse(_DASH.read_text())
for _node in _tree.body:
    if (
        isinstance(_node, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "_GEX_NEUTRAL_ABS"
            for t in _node.targets
        )
    ):
        exec(compile(ast.Module([_node], type_ignores=[]), str(_DASH), "exec"), _NS)
    elif isinstance(_node, ast.FunctionDef) and _node.name in _KEEP:
        exec(compile(ast.Module([_node], type_ignores=[]), str(_DASH), "exec"), _NS)

_classify_gex_env = _NS["_classify_gex_env"]
_gex_last_per_local_day = _NS["_gex_last_per_local_day"]
_fmt_gex_dollars = _NS["_fmt_gex_dollars"]
_fmt_gex_delta = _NS["_fmt_gex_delta"]

ET = ZoneInfo("America/New_York")


def _snap(local: datetime, total: float) -> dict:
    return {
        "captured_at_local": local,
        "captured_at_utc": local.astimezone(timezone.utc),
        "total_gex": total,
        "env": _classify_gex_env(total),
    }


class ClassifyEnvTests(unittest.TestCase):
    def test_neutral_band(self) -> None:
        self.assertEqual(_classify_gex_env(0), "NEUTRAL")
        self.assertEqual(_classify_gex_env(999_999), "NEUTRAL")
        self.assertEqual(_classify_gex_env(-999_999), "NEUTRAL")

    def test_edges(self) -> None:
        self.assertEqual(_classify_gex_env(1_000_000), "POSITIVE")
        self.assertEqual(_classify_gex_env(-1_000_000), "NEGATIVE")
        self.assertEqual(_classify_gex_env(12_000_000), "POSITIVE")
        self.assertEqual(_classify_gex_env(-3_500_000), "NEGATIVE")


class FormatTests(unittest.TestCase):
    def test_compact_dollars(self) -> None:
        self.assertEqual(_fmt_gex_dollars(12_400_000), "$12.4M")
        self.assertEqual(_fmt_gex_dollars(-400_000), "-$400K")

    def test_signed_delta(self) -> None:
        self.assertEqual(_fmt_gex_delta(2_000_000), "+$2.0M")
        self.assertEqual(_fmt_gex_delta(-2_000_000), "-$2.0M")
        self.assertEqual(_fmt_gex_delta(0), "$0")


class LastPerLocalDayTests(unittest.TestCase):
    def test_keeps_latest_print_per_day(self) -> None:
        d0 = datetime(2026, 9, 17, 10, 0, tzinfo=ET)
        d1 = datetime(2026, 9, 18, 9, 30, tzinfo=ET)
        hist = [
            _snap(d0, -2_000_000),
            _snap(d0 + timedelta(hours=6), -8_000_000),
            _snap(d1, 1_500_000),
            _snap(d1 + timedelta(hours=4), 4_000_000),
        ]
        daily = _gex_last_per_local_day(hist)
        self.assertEqual(len(daily), 2)
        self.assertEqual(daily[0]["total_gex"], -8_000_000)
        self.assertEqual(daily[1]["total_gex"], 4_000_000)
        self.assertEqual(daily[0]["env"], "NEGATIVE")
        self.assertEqual(daily[1]["env"], "POSITIVE")

    def test_single_day(self) -> None:
        d0 = datetime(2026, 9, 21, 11, 0, tzinfo=ET)
        daily = _gex_last_per_local_day([_snap(d0, 100)])
        self.assertEqual(len(daily), 1)
        self.assertEqual(daily[0]["env"], "NEUTRAL")


if __name__ == "__main__":
    unittest.main()
