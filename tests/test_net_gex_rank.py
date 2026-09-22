"""Tests for net GEX day ranking stats."""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

_DASH = Path(__file__).resolve().parents[1] / "scripts" / "oi_dashboard_app.py"

_NS: dict = {}
_tree = ast.parse(_DASH.read_text())
for _node in _tree.body:
    if isinstance(_node, ast.FunctionDef) and _node.name == "_net_gex_rank_stats":
        exec(compile(ast.Module([_node], type_ignores=[]), str(_DASH), "exec"), _NS)

_net_gex_rank_stats = _NS["_net_gex_rank_stats"]


class RankStatsTests(unittest.TestCase):
    def test_biggest_day(self) -> None:
        r = _net_gex_rank_stats(-9e6, [1e6, -2e6, 3e6, -4e6])
        self.assertEqual(r["rank"], 1)
        self.assertEqual(r["n"], 4)
        self.assertEqual(r["pctile"], 100.0)
        self.assertEqual(r["max_abs"], 4e6)

    def test_smallest_day(self) -> None:
        r = _net_gex_rank_stats(0.5e6, [1e6, -2e6, 3e6, -4e6])
        self.assertEqual(r["rank"], 5)
        self.assertEqual(r["pctile"], 0.0)

    def test_middle_uses_abs(self) -> None:
        # |today| = 2.5M sits between 2M and 3M.
        r = _net_gex_rank_stats(-2.5e6, [1e6, -2e6, 3e6, -4e6])
        self.assertEqual(r["rank"], 3)
        self.assertEqual(r["pctile"], 50.0)
        self.assertEqual(r["avg_abs"], (1e6 + 2e6 + 3e6 + 4e6) / 4)
        self.assertEqual(r["median_abs"], 2.5e6)

    def test_odd_median(self) -> None:
        r = _net_gex_rank_stats(1e6, [1e6, 2e6, 3e6])
        self.assertEqual(r["median_abs"], 2e6)

    def test_empty_history(self) -> None:
        self.assertIsNone(_net_gex_rank_stats(1e6, []))


if __name__ == "__main__":
    unittest.main()
