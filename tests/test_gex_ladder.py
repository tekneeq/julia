"""Tests for the day-only GEX level ladder on the session chart."""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

_DASH = Path(__file__).resolve().parents[1] / "scripts" / "oi_dashboard_app.py"
_KEEP_ASSIGN = {
    "_GEX_LADDER_MAX_LEVELS",
    "_GEX_LADDER_POS",
    "_GEX_LADDER_NEG",
}
_KEEP_FN = {
    "_fmt_gex_cell",
    "_gex_ladder_levels",
    "_gex_ladder_fill",
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
_gex_ladder_levels = _NS["_gex_ladder_levels"]
_gex_ladder_fill = _NS["_gex_ladder_fill"]


class CellFormatTests(unittest.TestCase):
    def test_compact(self) -> None:
        self.assertEqual(_fmt_gex_cell(1_200_000), "1.2M")
        self.assertEqual(_fmt_gex_cell(-400_000), "-400K")
        self.assertEqual(_fmt_gex_cell(0), "0")


class LadderLevelsTests(unittest.TestCase):
    def test_clips_to_visible_band(self) -> None:
        by_k = {770.0: 190_000.0, 772.0: -1_200_000.0, 800.0: 9e6}
        ladder = _gex_ladder_levels(by_k, y_lo=769.0, y_hi=776.0)
        self.assertIsNotNone(ladder)
        strikes = [k for k, _ in ladder["levels"]]
        self.assertEqual(strikes, [770.0, 772.0])
        self.assertEqual(ladder["gmax"], 1_200_000.0)

    def test_caps_by_magnitude_keeps_price_order(self) -> None:
        by_k = {float(700 + i): float((i % 7 + 1) * 100_000) for i in range(60)}
        ladder = _gex_ladder_levels(by_k, y_lo=700, y_hi=760, max_levels=10)
        self.assertIsNotNone(ladder)
        strikes = [k for k, _ in ladder["levels"]]
        self.assertEqual(len(strikes), 10)
        self.assertEqual(strikes, sorted(strikes))
        # Only the biggest levels survive the cap (values run 100K–700K).
        self.assertTrue(all(g >= 600_000.0 for _, g in ladder["levels"]))

    def test_empty_or_flat(self) -> None:
        self.assertIsNone(_gex_ladder_levels({}, y_lo=1, y_hi=2))
        self.assertIsNone(_gex_ladder_levels({100.0: 1.0}, y_lo=200, y_hi=210))
        self.assertIsNone(_gex_ladder_levels({100.0: 0.0}, y_lo=99, y_hi=101))


class LadderFillTests(unittest.TestCase):
    def test_sign_colors(self) -> None:
        gmax = 1_000_000
        plus = _gex_ladder_fill(900_000, gmax)
        minus = _gex_ladder_fill(-900_000, gmax)
        self.assertIn("38, 198, 218", plus)
        self.assertIn("236, 64, 122", minus)

    def test_alpha_scales_with_magnitude(self) -> None:
        gmax = 1_000_000
        strong = float(_gex_ladder_fill(1_000_000, gmax).split(",")[-1].rstrip(")"))
        weak = float(_gex_ladder_fill(50_000, gmax).split(",")[-1].rstrip(")"))
        self.assertGreater(strong, weak)


if __name__ == "__main__":
    unittest.main()
