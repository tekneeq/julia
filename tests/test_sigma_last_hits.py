"""Tests for the "last time the close beat ±1σ / ±2σ / ±3σ" lookup."""
from __future__ import annotations

import ast
import unittest
from datetime import date, timedelta
from pathlib import Path

_DASH = Path(__file__).resolve().parents[1] / "scripts" / "oi_dashboard_app.py"
_KEEP_ASSIGN = {"IMPLIED_MOVE_LEVELS", "_SIGMA_LAST_HIT_SESSIONS"}
_KEEP_FN = {"_business_days_between", "_last_sigma_hits", "_fmt_sessions_ago"}

_NS: dict = {"date": date, "timedelta": timedelta}
_tree = ast.parse(_DASH.read_text())
for _node in _tree.body:
    names = []
    if isinstance(_node, ast.Assign):
        names = [t.id for t in _node.targets if isinstance(t, ast.Name)]
    elif isinstance(_node, ast.AnnAssign) and isinstance(_node.target, ast.Name):
        names = [_node.target.id]
    if any(n in _KEEP_ASSIGN for n in names):
        exec(compile(ast.Module([_node], type_ignores=[]), str(_DASH), "exec"), _NS)
    elif isinstance(_node, ast.FunctionDef) and _node.name in _KEEP_FN:
        exec(compile(ast.Module([_node], type_ignores=[]), str(_DASH), "exec"), _NS)

_last_sigma_hits = _NS["_last_sigma_hits"]
_fmt_sessions_ago = _NS["_fmt_sessions_ago"]
LEVELS = _NS["IMPLIED_MOVE_LEVELS"]
C1, C2, C3 = (c for c, _, _ in LEVELS)

TODAY = date(2026, 10, 22)  # Thursday


def _entry(day: date, actual: float | None, one_sigma: float | None = 0.80) -> dict:
    implied = None
    if one_sigma is not None:
        implied = {C1: one_sigma, C2: one_sigma * 2, C3: one_sigma * 3}
    return {"day": day, "actual_pct": actual, "implied": implied}


def _weekdays_back(end: date, n: int) -> list[date]:
    days: list[date] = []
    d = end
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d -= timedelta(days=1)
    return list(reversed(days))


class LastSigmaHitTests(unittest.TestCase):
    def test_finds_most_recent_per_band(self) -> None:
        days = _weekdays_back(TODAY, 10)
        moves = [0.1, -2.6, 0.3, 1.0, -0.2, 0.4, -1.7, 0.2, 0.9, 0.5]
        # 1σ band is ±0.80, 2σ ±1.60, 3σ ±2.40.
        hist = [_entry(d, m) for d, m in zip(days, moves)]
        hits = _last_sigma_hits(hist, TODAY)
        # Last ≥1σ close is +0.9 two sessions before today.
        self.assertEqual(hits[C1]["day"], days[-2])
        self.assertEqual(hits[C1]["sessions_ago"], 1)
        self.assertEqual(hits[C1]["actual_pct"], 0.9)
        # Last ≥2σ close is −1.7.
        self.assertEqual(hits[C2]["day"], days[-4])
        self.assertEqual(hits[C2]["sessions_ago"], 3)
        # Last ≥3σ close is −2.6.
        self.assertEqual(hits[C3]["day"], days[1])
        self.assertEqual(hits[C3]["sessions_ago"], 8)
        self.assertEqual(hits[C3]["calendar_days"], (TODAY - days[1]).days)
        self.assertFalse(hits[C3]["is_live"])

    def test_none_when_never_hit(self) -> None:
        days = _weekdays_back(TODAY, 5)
        hist = [_entry(d, 0.2) for d in days]
        hits = _last_sigma_hits(hist, TODAY)
        self.assertIsNone(hits[C2])
        self.assertIsNone(hits[C3])
        self.assertIsNone(hits[C1])

    def test_today_live_counts_and_days_since_is_zero(self) -> None:
        days = _weekdays_back(TODAY, 3)
        hist = [_entry(days[0], 0.1), _entry(days[1], 0.2), _entry(TODAY, -2.5)]
        hits = _last_sigma_hits(hist, TODAY)
        for conf in (C1, C2, C3):
            self.assertEqual(hits[conf]["day"], TODAY)
            self.assertTrue(hits[conf]["is_live"])
            self.assertEqual(hits[conf]["sessions_ago"], 0)
            self.assertEqual(hits[conf]["calendar_days"], 0)
        self.assertEqual(_fmt_sessions_ago(hits[C3]), "today (live)")

    def test_skips_unscored_sessions(self) -> None:
        days = _weekdays_back(TODAY, 4)
        hist = [
            _entry(days[0], -3.0),           # scored, beats every band
            _entry(days[1], 5.0, None),      # no implied band — ignored
            _entry(days[2], None),           # no close — ignored
            _entry(days[3], 0.1),
        ]
        hits = _last_sigma_hits(hist, TODAY)
        self.assertEqual(hits[C3]["day"], days[0])
        self.assertEqual(hits[C3]["scanned"], 2)

    def test_exact_band_counts_as_hit(self) -> None:
        hist = [_entry(TODAY - timedelta(days=1), 1.6)]
        hits = _last_sigma_hits(hist, TODAY)
        self.assertIsNotNone(hits[C2])
        self.assertIsNone(hits[C3])

    def test_sessions_ago_text(self) -> None:
        hit = {"is_live": False, "sessions_ago": 1, "calendar_days": 3}
        self.assertEqual(_fmt_sessions_ago(hit), "1 session ago (3d)")
        hit = {"is_live": False, "sessions_ago": 8, "calendar_days": 12}
        self.assertEqual(_fmt_sessions_ago(hit), "8 sessions ago (12d)")


if __name__ == "__main__":
    unittest.main()
