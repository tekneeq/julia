"""Unit tests for Discord price-watch parsing and the SQLite store."""
from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

_SRC = Path(__file__).resolve().parents[1] / "src" / "julia" / "price_watch.py"
_SPEC = importlib.util.spec_from_file_location("price_watch", _SRC)
assert _SPEC and _SPEC.loader
pw = importlib.util.module_from_spec(_SPEC)
sys.modules["price_watch"] = pw
_SPEC.loader.exec_module(pw)

ET = ZoneInfo("America/New_York")


def _et(y, mo, d, h, mi=0, s=0) -> datetime:
    return datetime(y, mo, d, h, mi, s, tzinfo=ET)


class ParseSpecTests(unittest.TestCase):
    def test_1m(self) -> None:
        spec = pw.parse_watch_spec("1m")
        self.assertEqual(spec.kind, pw.KIND_INTERVAL)
        self.assertEqual(spec.interval_seconds, 60)
        self.assertEqual(spec.text, "every 1m")

    def test_5_minutes(self) -> None:
        spec = pw.parse_watch_spec("5 minutes")
        self.assertEqual(spec.interval_seconds, 300)

    def test_1h(self) -> None:
        self.assertEqual(pw.parse_watch_spec("1h").interval_seconds, 3600)

    def test_interval_too_short(self) -> None:
        with self.assertRaises(ValueError):
            pw.parse_watch_spec("5s")

    def test_plus_1_pct(self) -> None:
        spec = pw.parse_watch_spec("1%")
        self.assertEqual(spec.kind, pw.KIND_PCT)
        self.assertEqual(spec.pct_level, 1.0)
        self.assertTrue(pw.pct_condition_met(1.0, 1.0))
        self.assertFalse(pw.pct_condition_met(1.0, 0.99))

    def test_minus_1_pct(self) -> None:
        spec = pw.parse_watch_spec("-1%")
        self.assertEqual(spec.pct_level, -1.0)
        self.assertTrue(pw.pct_condition_met(-1.0, -1.2))
        self.assertFalse(pw.pct_condition_met(-1.0, -0.5))

    def test_plus_signed_pct(self) -> None:
        self.assertEqual(pw.parse_watch_spec("+1.5%").pct_level, 1.5)

    def test_price(self) -> None:
        spec = pw.parse_watch_spec("759")
        self.assertEqual(spec.kind, pw.KIND_PRICE)
        self.assertEqual(spec.price_level, 759.0)

    def test_price_hits_resolve(self) -> None:
        spec = pw.parse_watch_spec("759")
        below = pw.resolve_price_op(spec, 750.0)
        self.assertEqual(below.price_op, ">=")
        self.assertTrue(pw.price_condition_met(">=", 759.0, 759.0))
        above = pw.resolve_price_op(spec, 760.0)
        self.assertEqual(above.price_op, "<=")

    def test_args(self) -> None:
        req = pw.parse_watch_args(["spy", "1m"])
        self.assertEqual(req.symbol, "SPY")
        self.assertEqual(req.spec.kind, pw.KIND_INTERVAL)

    def test_bad_trigger(self) -> None:
        with self.assertRaises(ValueError):
            pw.parse_watch_spec("soon")

    def test_fractal_spec(self) -> None:
        spec = pw.parse_watch_spec("fractal")
        self.assertEqual(spec.kind, pw.KIND_FRACTAL)
        self.assertEqual(spec.interval_seconds, 300)
        self.assertEqual(pw.parse_watch_spec("5m fractal").kind, pw.KIND_FRACTAL)
        self.assertEqual(pw.parse_watch_spec("fractal 5m").kind, pw.KIND_FRACTAL)
        req = pw.parse_watch_args(["spy", "fractal"])
        self.assertEqual(req.symbol, "SPY")
        self.assertEqual(req.spec.kind, pw.KIND_FRACTAL)

    def test_fractal_other_timeframe_rejected(self) -> None:
        with self.assertRaises(ValueError):
            pw.parse_watch_spec("1m fractal")


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = str(Path(self.tmp.name) / "watch.db")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _quote(self, price: float, daily_pct: float = 0.2) -> dict:
        ref = 650.0
        return {
            "symbol": "SPY",
            "price": price,
            "ref": ref,
            "daily_chg": price - ref,
            "daily_pct": daily_pct,
        }

    def test_interval_due_and_since_last(self) -> None:
        spec = pw.parse_watch_spec("1m")
        start = _et(2026, 9, 18, 10, 0)
        job = pw.insert_watch(
            symbol="SPY",
            spec=spec,
            last_price=650.0,
            last_printed_at=start,
            print_count=1,
            db_path=self.db,
        )
        none = pw.claim_due_watches(
            {"SPY": self._quote(650.10)},
            start + timedelta(seconds=30),
            db_path=self.db,
        )
        self.assertEqual(none, [])
        due = pw.claim_due_watches(
            {"SPY": self._quote(651.00)},
            start + timedelta(seconds=60),
            db_path=self.db,
        )
        self.assertEqual(len(due), 1)
        updated, quote = due[0]
        self.assertEqual(updated["status"], pw.STATUS_PENDING)
        self.assertEqual(updated["print_count"], 2)
        self.assertEqual(quote["prev_print_price"], 650.0)
        msg = pw.format_quote_message(updated, quote)
        self.assertIn("since last", msg)
        self.assertIn("+1.00", msg)

    def test_pct_fires_once(self) -> None:
        spec = pw.parse_watch_spec("1%")
        pw.insert_watch(symbol="SPY", spec=spec, db_path=self.db)
        none = pw.claim_due_watches(
            {"SPY": self._quote(655.0, daily_pct=0.8)},
            db_path=self.db,
        )
        self.assertEqual(none, [])
        due = pw.claim_due_watches(
            {"SPY": self._quote(657.0, daily_pct=1.05)},
            db_path=self.db,
        )
        self.assertEqual(len(due), 1)
        self.assertEqual(due[0][0]["status"], pw.STATUS_FIRED)
        again = pw.claim_due_watches(
            {"SPY": self._quote(658.0, daily_pct=1.2)},
            db_path=self.db,
        )
        self.assertEqual(again, [])

    def test_price_and_cancel(self) -> None:
        spec = pw.resolve_price_op(pw.parse_watch_spec("759"), 750.0)
        job = pw.insert_watch(symbol="SPY", spec=spec, db_path=self.db)
        cancelled = pw.cancel_watch(int(job["id"]), db_path=self.db)
        assert cancelled is not None
        self.assertEqual(cancelled["status"], pw.STATUS_CANCELLED)
        self.assertEqual(
            pw.claim_due_watches(
                {"SPY": self._quote(759.0)},
                db_path=self.db,
            ),
            [],
        )

    def test_fractal_watch_prints_once(self) -> None:
        start = _et(2026, 10, 2, 9, 30)
        bars = _bars(start, [
            (5, 1), (6, 1), (7, 1), (10, 1), (8, 1), (7, 1), (6, 1),
        ])
        now = start + timedelta(minutes=35)
        spec = pw.parse_watch_spec("fractal")
        pw.insert_watch(
            symbol="SPY",
            spec=spec,
            last_price=9.0,
            last_printed_at=start,
            db_path=self.db,
        )
        due = pw.claim_due_watches(
            {"SPY": self._quote(9.5)},
            now,
            candles={"SPY": bars},
            db_path=self.db,
        )
        self.assertEqual(len(due), 1)
        job, quote = due[0]
        self.assertEqual(job["status"], pw.STATUS_PENDING)
        self.assertEqual(job["print_count"], 1)
        msg = pw.format_quote_message(job, quote)
        self.assertIn("fractal **top**", msg)
        self.assertIn("$10.00", msg)
        self.assertIn("09:45", msg)
        again = pw.claim_due_watches(
            {"SPY": self._quote(9.5)},
            now + timedelta(seconds=20),
            candles={"SPY": bars},
            db_path=self.db,
        )
        self.assertEqual(again, [])


def _bars(start: datetime, pairs: list[tuple[float, float]]) -> list[dict]:
    return [
        {
            "ts": start + timedelta(minutes=5 * i),
            "high": high,
            "low": low,
        }
        for i, (high, low) in enumerate(pairs)
    ]


class FractalTests(unittest.TestCase):
    def test_two_candle_top_and_not_one_candle_noise(self) -> None:
        start = _et(2026, 10, 2, 9, 30)
        # Index 3 is a real top: high 10 beats two candles each side.
        bars = _bars(start, [
            (5, 1), (6, 1), (7, 1), (10, 1), (8, 1), (7, 1), (6, 1),
        ])
        # All seven bars are closed.
        now = start + timedelta(minutes=35)
        found = pw.confirmed_fractals(bars, now=now)
        tops = [f for f in found if f["kind"] == "top"]
        self.assertEqual(len(tops), 1)
        self.assertEqual(tops[0]["price"], 10)
        self.assertEqual(tops[0]["ts"], start + timedelta(minutes=15))
        # Confirmed when the second candle after it closes (10:00 bar
        # starts 9:55, closes 10:00).
        self.assertEqual(tops[0]["confirmed_at"], start + timedelta(minutes=30))

        # 5 is above the candles next to it, but the candle two back is
        # higher, so a two-candle fractal does not count it.
        noise = _bars(start, [
            (6, 1), (3, 1), (5, 1), (4, 1), (4, 1), (4, 1), (4, 1),
        ])
        self.assertEqual(pw.confirmed_fractals(noise, now=now), [])

    def test_bottom_and_outside_bar(self) -> None:
        start = _et(2026, 10, 2, 9, 30)
        now = start + timedelta(minutes=35)
        bars = _bars(start, [
            (9, 5), (9, 4), (9, 3), (10, 1), (9, 3), (9, 4), (9, 5),
        ])
        found = pw.confirmed_fractals(bars, now=now)
        kinds = {(f["kind"], f["price"]) for f in found}
        self.assertIn(("top", 10), kinds)
        self.assertIn(("bottom", 1), kinds)

    def test_forming_bar_does_not_confirm(self) -> None:
        start = _et(2026, 10, 2, 9, 30)
        bars = _bars(start, [
            (5, 1), (6, 1), (7, 1), (10, 1), (8, 1), (7, 1), (6, 1),
        ])
        # The confirming candle (the second one after the top) is still open.
        too_soon = start + timedelta(minutes=30) - timedelta(seconds=1)
        self.assertEqual(pw.confirmed_fractals(bars, now=too_soon), [])
        closed = pw.confirmed_fractals(bars, now=start + timedelta(minutes=30))
        self.assertEqual([f["kind"] for f in closed], ["top"])

    def test_watermark_skips_already_confirmed(self) -> None:
        start = _et(2026, 10, 2, 9, 30)
        bars = _bars(start, [
            (5, 1), (6, 1), (7, 1), (10, 1), (8, 1), (7, 1), (6, 1),
        ])
        now = start + timedelta(minutes=35)
        confirm = start + timedelta(minutes=30)
        self.assertEqual(pw.fractals_since(bars, confirm, now=now), [])
        fresh = pw.fractals_since(bars, confirm - timedelta(seconds=1), now=now)
        self.assertEqual(len(fresh), 1)


if __name__ == "__main__":
    unittest.main()
