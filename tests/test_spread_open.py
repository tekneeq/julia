"""Unit tests for !lia open spread parsing and pricing."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src" / "julia" / "spread_open.py"
_SPEC = importlib.util.spec_from_file_location("spread_open", _SRC)
assert _SPEC and _SPEC.loader
so = importlib.util.module_from_spec(_SPEC)
sys.modules["spread_open"] = so
_SPEC.loader.exec_module(so)


class ParseTests(unittest.TestCase):
    def test_bear_call(self) -> None:
        req = so.parse_open_spread_args(
            ["SPY", "0dte", "776/778", "call", "1"]
        )
        self.assertEqual(req.symbol, "SPY")
        self.assertEqual(req.exp_token, "0dte")
        self.assertEqual(req.short_strike, 776.0)
        self.assertEqual(req.long_strike, 778.0)
        self.assertEqual(req.option_type, "call")
        self.assertEqual(req.qty, 1)
        self.assertIsNone(req.credit)
        self.assertEqual(req.width, 2.0)
        self.assertEqual(req.kind, "bear call")

    def test_bull_put_with_credit(self) -> None:
        req = so.parse_open_spread_args(
            ["spy", "2026-09-25", "770/768", "put", "2", "0.35"]
        )
        self.assertEqual(req.symbol, "SPY")
        self.assertEqual(req.short_strike, 770.0)
        self.assertEqual(req.long_strike, 768.0)
        self.assertEqual(req.qty, 2)
        self.assertEqual(req.credit, 0.35)
        self.assertEqual(req.kind, "bull put")

    def test_strike_separators(self) -> None:
        for token in ("770/768", "770-768", "770:768", "770 / 768"):
            req = so.parse_open_spread_args(["SPY", "0dte", token, "p", "1"])
            self.assertEqual((req.short_strike, req.long_strike), (770.0, 768.0))

    def test_type_aliases(self) -> None:
        self.assertEqual(
            so.parse_open_spread_args(["SPY", "0dte", "776/778", "c", "1"]).option_type,
            "call",
        )
        self.assertEqual(
            so.parse_open_spread_args(["SPY", "0dte", "770/768", "puts", "1"]).option_type,
            "put",
        )

    def test_rejects_wrong_geometry(self) -> None:
        with self.assertRaises(ValueError):
            so.parse_open_spread_args(["SPY", "0dte", "778/776", "call", "1"])
        with self.assertRaises(ValueError):
            so.parse_open_spread_args(["SPY", "0dte", "768/770", "put", "1"])
        with self.assertRaises(ValueError):
            so.parse_open_spread_args(["SPY", "0dte", "770/770", "put", "1"])

    def test_rejects_bad_inputs(self) -> None:
        with self.assertRaises(ValueError):
            so.parse_open_spread_args(["SPY", "0dte", "776/778", "call"])
        with self.assertRaises(ValueError):
            so.parse_open_spread_args(["SPY", "tomorrow", "776/778", "call", "1"])
        with self.assertRaises(ValueError):
            so.parse_open_spread_args(["SPY", "0dte", "776/778", "call", "0"])
        with self.assertRaises(ValueError):
            so.parse_open_spread_args(["SPY", "0dte", "776/778", "call", "1", "-0.2"])
        with self.assertRaises(ValueError):
            so.parse_open_spread_args(["S&P", "0dte", "776/778", "call", "1"])


class DefaultCreditTests(unittest.TestCase):
    def test_natural_credit(self) -> None:
        c = so.default_credit(
            {"bid": 1.20, "ask": 1.30, "mark": 1.25},
            {"bid": 0.80, "ask": 0.90, "mark": 0.85},
        )
        self.assertEqual(c, 0.30)

    def test_mark_fallback(self) -> None:
        c = so.default_credit({"bid": None, "mark": 1.25}, {"ask": None, "mark": 0.85})
        self.assertEqual(c, 0.40)

    def test_floor_and_missing(self) -> None:
        self.assertEqual(
            so.default_credit({"bid": 0.50}, {"ask": 0.60}), 0.01,
        )
        self.assertIsNone(so.default_credit({"bid": None}, {"ask": 0.60}))


class SummaryTests(unittest.TestCase):
    def test_bull_put_summary(self) -> None:
        req = so.parse_open_spread_args(["SPY", "0dte", "770/768", "put", "2"])
        s = so.spread_summary(req, 0.35)
        self.assertEqual(s["width"], 2.0)
        self.assertEqual(s["max_gain"], 70.0)
        self.assertEqual(s["max_loss"], 330.0)
        self.assertEqual(s["breakeven"], 769.65)

    def test_bear_call_breakeven(self) -> None:
        req = so.parse_open_spread_args(["SPY", "0dte", "776/778", "call", "1"])
        s = so.spread_summary(req, 0.50)
        self.assertEqual(s["breakeven"], 776.50)
        self.assertEqual(s["max_gain"], 50.0)
        self.assertEqual(s["max_loss"], 150.0)


if __name__ == "__main__":
    unittest.main()
