"""Tests for the strike × time option-volume grid."""
from __future__ import annotations

import ast
import unittest
from datetime import datetime
from pathlib import Path

_DASH = Path(__file__).resolve().parents[1] / "scripts" / "oi_dashboard_app.py"
_KEEP_ASSIGN = {
    "_FLOW_STRIKES_FROM_OPEN",
    "_FLOW_STRIKES_AROUND_SPOT",
    "_FLOW_BUCKET_MIN",
}
_KEEP_FN = {
    "_flow_fmt_contracts",
    "_flow_bucket_floor",
    "_flow_strike_window",
    "_flow_volume_deltas",
    "_flow_cell_text",
    "_flow_cell_balance",
    "_flow_strike_axis_label",
    "_flow_oi_increases",
    "_flow_side_maps",
}

_NS: dict = {"datetime": datetime}
_tree = ast.parse(_DASH.read_text())
for _node in _tree.body:
    names = []
    if isinstance(_node, ast.Assign):
        names = [t.id for t in _node.targets if isinstance(t, ast.Name)]
    if any(n in _KEEP_ASSIGN for n in names):
        exec(compile(ast.Module([_node], type_ignores=[]), str(_DASH), "exec"), _NS)
    elif isinstance(_node, ast.FunctionDef) and _node.name in _KEEP_FN:
        exec(compile(ast.Module([_node], type_ignores=[]), str(_DASH), "exec"), _NS)

_flow_fmt_contracts = _NS["_flow_fmt_contracts"]
_flow_bucket_floor = _NS["_flow_bucket_floor"]
_flow_strike_window = _NS["_flow_strike_window"]
_flow_volume_deltas = _NS["_flow_volume_deltas"]
_flow_cell_text = _NS["_flow_cell_text"]
_flow_cell_balance = _NS["_flow_cell_balance"]
_flow_strike_axis_label = _NS["_flow_strike_axis_label"]
_flow_oi_increases = _NS["_flow_oi_increases"]
_flow_side_maps = _NS["_flow_side_maps"]


def _snap(hour, minute, call, put):
    return {
        "ts": datetime(2026, 9, 25, hour, minute),
        "call": call,
        "put": put,
    }


class FormatTests(unittest.TestCase):
    def test_compact(self) -> None:
        self.assertEqual(_flow_fmt_contracts(860), "860")
        self.assertEqual(_flow_fmt_contracts(1200), "1.2k")
        self.assertEqual(_flow_fmt_contracts(12000), "12k")

    def test_cell_text(self) -> None:
        self.assertEqual(_flow_cell_text(0, 0), "")
        self.assertEqual(_flow_cell_text(1200, 300), "1.2k/300")

    def test_balance_is_share_not_size(self) -> None:
        self.assertEqual(_flow_cell_balance(0, 0), 0.0)
        self.assertAlmostEqual(_flow_cell_balance(300, 100), 0.5)
        self.assertAlmostEqual(_flow_cell_balance(100, 300), -0.5)
        # A small one-sided print is as green as a large one.
        self.assertAlmostEqual(_flow_cell_balance(10, 0), 1.0)
        self.assertAlmostEqual(_flow_cell_balance(10000, 0), 1.0)

    def test_strike_label_marks_new_oi(self) -> None:
        self.assertEqual(_flow_strike_axis_label(770), "$770")
        self.assertEqual(_flow_strike_axis_label(770.5), "$770.50")
        self.assertEqual(_flow_strike_axis_label(770, oi_up=True), "$770 ↑")


class WindowTests(unittest.TestCase):
    def test_ten_around_open_plus_spot(self) -> None:
        strikes = [float(k) for k in range(740, 801)]
        window = _flow_strike_window(strikes, 770.2, 790.0, n=10, spot_pad=3)
        self.assertIn(760.0, window)
        self.assertIn(780.0, window)
        self.assertNotIn(759.0, window)
        self.assertIn(787.0, window)
        self.assertIn(793.0, window)
        self.assertNotIn(794.0, window)


class DeltaTests(unittest.TestCase):
    def test_first_bucket_is_cumulative_then_increases(self) -> None:
        snaps = [
            _snap(9, 35, {770.0: 100, 771.0: 40}, {770.0: 20, 771.0: 10}),
            _snap(10, 5, {770.0: 250, 771.0: 40}, {770.0: 80, 771.0: 10}),
            _snap(10, 20, {770.0: 300, 771.0: 90}, {770.0: 80, 771.0: 50}),
        ]
        grid = _flow_volume_deltas(snaps, [770.0, 771.0])
        self.assertEqual(grid["labels"], ["09:30", "10:00"])
        # 770 call: 100 in the first bucket, then 300-100 = 200
        row770 = grid["strikes"].index(770.0)
        self.assertEqual(grid["call"][row770], [100, 200])
        self.assertEqual(grid["put"][row770], [20, 60])
        # 10:00 and 10:20 collapse; latest cumulative wins
        row771 = grid["strikes"].index(771.0)
        self.assertEqual(grid["call"][row771], [40, 50])
        self.assertEqual(grid["put"][row771], [10, 40])

    def test_volume_never_goes_negative(self) -> None:
        snaps = [
            _snap(9, 40, {770.0: 100}, {770.0: 50}),
            _snap(10, 10, {770.0: 80}, {770.0: 50}),
        ]
        grid = _flow_volume_deltas(snaps, [770.0])
        self.assertEqual(grid["call"][0][1], 0)

    def test_missing_strike_keeps_prior_cumulative(self) -> None:
        snaps = [
            _snap(9, 40, {770.0: 100}, {770.0: 10}),
            _snap(10, 10, {}, {770.0: 25}),
        ]
        grid = _flow_volume_deltas(snaps, [770.0])
        self.assertEqual(grid["call"][0], [100, 0])
        self.assertEqual(grid["put"][0], [10, 15])


class OiTests(unittest.TestCase):
    def test_increase_requires_prior_print(self) -> None:
        up = _flow_oi_increases({770.0: 1000}, {770.0: 1500, 771.0: 400})
        self.assertEqual(up, {770.0: 500})

    def test_flat_or_down_ignored(self) -> None:
        up = _flow_oi_increases({770.0: 1000, 771.0: 200}, {770.0: 1000, 771.0: 50})
        self.assertEqual(up, {})


class SideMapTests(unittest.TestCase):
    def test_splits_call_put(self) -> None:
        rows = [
            {"option_type": "call", "strike_price": 770, "volume": 10, "open_interest": 3},
            {"option_type": "put", "strike_price": 770, "volume": 4, "open_interest": 8},
            {"option_type": "call", "strike_price": 771, "volume": None, "open_interest": None},
        ]
        call, put, call_oi, put_oi = _flow_side_maps(rows)
        self.assertEqual(call[770.0], 10)
        self.assertEqual(put[770.0], 4)
        self.assertEqual(call_oi[770.0], 3)
        self.assertEqual(put_oi[770.0], 8)
        self.assertEqual(call[771.0], 0)


if __name__ == "__main__":
    unittest.main()
