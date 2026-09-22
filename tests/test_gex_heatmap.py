"""Tests for the on-chart GEX heatmap grid."""
from __future__ import annotations

import ast
import unittest
from datetime import date
from pathlib import Path

_DASH = Path(__file__).resolve().parents[1] / "scripts" / "oi_dashboard_app.py"
_KEEP_ASSIGN = {
    "_GEX_HM_MAX_STRIKES",
}
_KEEP_FN = {
    "_fmt_gex_cell",
    "_gex_heat_color",
    "_gex_heatmap_matrix",
}

_NS: dict = {}
_tree = ast.parse(_DASH.read_text())
for _node in _tree.body:
    if (
        isinstance(_node, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id in _KEEP_ASSIGN
            for t in _node.targets
        )
    ):
        exec(compile(ast.Module([_node], type_ignores=[]), str(_DASH), "exec"), _NS)
    elif isinstance(_node, ast.FunctionDef) and _node.name in _KEEP_FN:
        exec(compile(ast.Module([_node], type_ignores=[]), str(_DASH), "exec"), _NS)

_fmt_gex_cell = _NS["_fmt_gex_cell"]
_gex_heat_color = _NS["_gex_heat_color"]
_gex_heatmap_matrix = _NS["_gex_heatmap_matrix"]


class CellFormatTests(unittest.TestCase):
    def test_compact(self) -> None:
        self.assertEqual(_fmt_gex_cell(1_200_000), "1.2M")
        self.assertEqual(_fmt_gex_cell(-400_000), "-400K")
        self.assertEqual(_fmt_gex_cell(0), "0")


class HeatColorTests(unittest.TestCase):
    def test_sign_buckets(self) -> None:
        zmax = 10_000_000
        plus = _gex_heat_color(8_000_000, zmax)
        minus = _gex_heat_color(-8_000_000, zmax)
        flat = _gex_heat_color(0, zmax)
        self.assertIn("253, 216, 53", plus)  # yellow large +
        self.assertIn("106, 27, 154", minus)  # purple large −
        self.assertIn("30, 34, 45", flat)


class MatrixTests(unittest.TestCase):
    def test_clips_to_visible_band_and_fills_holes(self) -> None:
        d0 = date(2026, 9, 23)
        d1 = date(2026, 9, 25)
        cols = [
            (d0, {660.0: -2e6, 661.0: 1e6, 680.0: 5e6}),
            (d1, {660.0: -1e6, 662.0: 3e6}),
        ]
        grid = _gex_heatmap_matrix(cols, y_lo=659.0, y_hi=663.0)
        self.assertIsNotNone(grid)
        self.assertEqual(grid["exps"], [d0, d1])
        self.assertEqual(grid["strikes"], [660.0, 661.0, 662.0])
        self.assertNotIn(680.0, grid["strikes"])
        # 661 missing on d1 → None
        i661 = grid["strikes"].index(661.0)
        self.assertEqual(grid["z"][i661][0], 1e6)
        self.assertIsNone(grid["z"][i661][1])
        self.assertEqual(grid["text"][i661][0], "1.0M")
        self.assertEqual(grid["text"][i661][1], "")

    def test_caps_strike_count(self) -> None:
        d0 = date(2026, 9, 23)
        by_k = {float(600 + i): float(i * 1_000) for i in range(80)}
        grid = _gex_heatmap_matrix(
            [(d0, by_k)], y_lo=600.0, y_hi=700.0, center=640.0, max_strikes=12,
        )
        self.assertIsNotNone(grid)
        self.assertLessEqual(len(grid["strikes"]), 13)  # cap + center
        self.assertIn(640.0, grid["strikes"])

    def test_empty(self) -> None:
        self.assertIsNone(_gex_heatmap_matrix([], y_lo=1, y_hi=2))
        self.assertIsNone(_gex_heatmap_matrix(
            [(date(2026, 9, 23), {100.0: 1.0})], y_lo=200, y_hi=210,
        ))


if __name__ == "__main__":
    unittest.main()
