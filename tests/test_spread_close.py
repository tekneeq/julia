"""Unit tests for SPY spread-close parsing and the SQLite job store."""
from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

_SRC = Path(__file__).resolve().parents[1] / "src" / "julia" / "spread_close.py"
_SPEC = importlib.util.spec_from_file_location("spread_close", _SRC)
assert _SPEC and _SPEC.loader
sc = importlib.util.module_from_spec(_SPEC)
sys.modules["spread_close"] = sc
_SPEC.loader.exec_module(sc)

ET = ZoneInfo("America/New_York")


def _et(y, mo, d, h, mi=0) -> datetime:
    return datetime(y, mo, d, h, mi, tzinfo=ET)


class ParseTriggerTests(unittest.TestCase):
    def test_now(self) -> None:
        t = sc.parse_trigger("now", now=_et(2026, 9, 18, 10, 0))
        self.assertEqual(t.kind, "now")

    def test_at_2pm_today(self) -> None:
        t = sc.parse_trigger("at 2pm", now=_et(2026, 9, 18, 10, 0))
        self.assertEqual(t.kind, "at")
        self.assertEqual(t.trigger_at, _et(2026, 9, 18, 14, 0))
        self.assertIn("2pm", t.text)

    def test_at_1400_rolls_tomorrow(self) -> None:
        t = sc.parse_trigger("at 14:00", now=_et(2026, 9, 18, 15, 0))
        self.assertEqual(t.trigger_at, _et(2026, 9, 19, 14, 0))
        self.assertIn("2026-09-19", t.text)

    def test_at_2_30pm(self) -> None:
        t = sc.parse_trigger("at 2:30 pm", now=_et(2026, 9, 18, 10, 0))
        self.assertEqual(t.trigger_at, _et(2026, 9, 18, 14, 30))

    def test_in_1h(self) -> None:
        t = sc.parse_trigger("in 1h", now=_et(2026, 9, 18, 10, 0))
        self.assertEqual(t.kind, "in")
        self.assertEqual(t.trigger_at, _et(2026, 9, 18, 11, 0))

    def test_in_1_hour_from_now(self) -> None:
        t = sc.parse_trigger("in 1 hour from now", now=_et(2026, 9, 18, 10, 0))
        self.assertEqual(t.trigger_at, _et(2026, 9, 18, 11, 0))

    def test_in_30m(self) -> None:
        t = sc.parse_trigger("in 30m", now=_et(2026, 9, 18, 10, 0))
        self.assertEqual(t.trigger_at, _et(2026, 9, 18, 10, 30))

    def test_if_spy_gte(self) -> None:
        t = sc.parse_trigger("if spy >= 650", now=_et(2026, 9, 18, 10, 0))
        self.assertEqual(t.kind, "price")
        self.assertEqual(t.price_op, ">=")
        self.assertEqual(t.price_level, 650.0)
        self.assertFalse(t.hits_unresolved)

    def test_when_spy_hits(self) -> None:
        t = sc.parse_trigger("when spy hits 650", now=_et(2026, 9, 18, 10, 0))
        self.assertTrue(t.hits_unresolved)
        self.assertEqual(t.price_level, 650.0)

    def test_hits_resolves_from_below(self) -> None:
        t = sc.parse_trigger("hits 650")
        resolved = sc.resolve_hits_trigger(t, 648.0)
        self.assertEqual(resolved.price_op, ">=")
        self.assertTrue(sc.price_condition_met(">=", 650.0, 650.0))
        self.assertFalse(sc.price_condition_met(">=", 650.0, 649.9))

    def test_hits_resolves_from_above(self) -> None:
        t = sc.parse_trigger("when spy hits 650")
        resolved = sc.resolve_hits_trigger(t, 652.0)
        self.assertEqual(resolved.price_op, "<=")
        self.assertTrue(sc.price_condition_met("<=", 650.0, 650.0))

    def test_bad_trigger(self) -> None:
        with self.assertRaises(ValueError):
            sc.parse_trigger("whenever")


class ParseCloseArgsTests(unittest.TestCase):
    def test_no_qty_at(self) -> None:
        req = sc.parse_close_spread_args(
            ["a1b2c3", "d4e5f6", "at", "2pm"],
            now=_et(2026, 9, 18, 10, 0),
        )
        self.assertEqual(req.id1, "a1b2c3")
        self.assertIsNone(req.qty)
        self.assertEqual(req.trigger.kind, "at")

    def test_qty_then_in(self) -> None:
        req = sc.parse_close_spread_args(
            ["aa11bb", "cc22dd", "2", "in", "1h"],
            now=_et(2026, 9, 18, 10, 0),
        )
        self.assertEqual(req.qty, 2)
        self.assertEqual(req.trigger.kind, "in")

    def test_if_spy(self) -> None:
        req = sc.parse_close_spread_args(
            ["aa", "bb", "if", "spy", ">=", "650.5"],
        )
        self.assertEqual(req.trigger.price_level, 650.5)
        self.assertEqual(req.trigger.price_op, ">=")


class ValidateLegsTests(unittest.TestCase):
    def _pos(
        self,
        sid: str,
        *,
        side: str,
        strike: float,
        qty: float = 2,
        symbol: str = "SPY",
        exp: str = "2026-09-18",
        otype: str = "put",
    ) -> dict:
        return {
            "_short_id": sid,
            "_symbol": symbol,
            "_expiration": exp,
            "_option_type": otype,
            "_side": side,
            "_strike": strike,
            "_qty": qty,
        }

    def test_credit_put_spread(self) -> None:
        long = self._pos("long01", side="long", strike=640)
        short = self._pos("shrt01", side="short", strike=650)
        legs = sc.validate_spread_legs(long, short, None)
        self.assertEqual(legs.long_id, "long01")
        self.assertEqual(legs.short_id, "shrt01")
        self.assertEqual(legs.qty, 2)

    def test_ids_either_order(self) -> None:
        long = self._pos("long01", side="long", strike=640)
        short = self._pos("shrt01", side="short", strike=650)
        legs = sc.validate_spread_legs(short, long, 1)
        self.assertEqual(legs.qty, 1)
        self.assertEqual(legs.short_strike, 650)

    def test_rejects_non_spy(self) -> None:
        with self.assertRaises(ValueError):
            sc.validate_spread_legs(
                self._pos("a", side="long", strike=100, symbol="QQQ"),
                self._pos("b", side="short", strike=110, symbol="QQQ"),
                None,
            )

    def test_rejects_two_longs(self) -> None:
        with self.assertRaises(ValueError):
            sc.validate_spread_legs(
                self._pos("a", side="long", strike=640),
                self._pos("b", side="long", strike=650),
                None,
            )

    def test_rejects_qty_above_min(self) -> None:
        with self.assertRaises(ValueError):
            sc.validate_spread_legs(
                self._pos("a", side="long", strike=640, qty=1),
                self._pos("b", side="short", strike=650, qty=3),
                2,
            )


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = str(Path(self.tmp.name) / "spread.db")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _legs(self) -> sc.SpreadLegs:
        return sc.SpreadLegs(
            symbol="SPY",
            expiration="2026-09-18",
            option_type="put",
            long_id="long01",
            long_strike=640.0,
            short_id="shrt01",
            short_strike=650.0,
            qty=1,
            long_label="long01 long 640 PUT",
            short_label="shrt01 short 650 PUT",
        )

    def test_insert_and_status(self) -> None:
        trig = sc.parse_trigger("at 2pm", now=_et(2026, 9, 18, 10, 0))
        job = sc.insert_job(legs=self._legs(), trigger=trig, db_path=self.db)
        self.assertEqual(job["status"], sc.STATUS_PENDING)
        lines = sc.format_status(sc.list_pending(db_path=self.db))
        self.assertIn("#1", lines)
        self.assertIn("shrt01", lines)

    def test_claim_time_due(self) -> None:
        trig = sc.parse_trigger("at 2pm", now=_et(2026, 9, 18, 10, 0))
        sc.insert_job(legs=self._legs(), trigger=trig, db_path=self.db)
        none = sc.claim_due_jobs(_et(2026, 9, 18, 13, 59), db_path=self.db)
        self.assertEqual(none, [])
        due = sc.claim_due_jobs(_et(2026, 9, 18, 14, 0), db_path=self.db)
        self.assertEqual(len(due), 1)
        self.assertEqual(due[0]["status"], sc.STATUS_FIRING)
        again = sc.claim_due_jobs(_et(2026, 9, 18, 14, 1), db_path=self.db)
        self.assertEqual(again, [])

    def test_claim_price_due(self) -> None:
        trig = sc.parse_trigger("if spy >= 650")
        sc.insert_job(legs=self._legs(), trigger=trig, db_path=self.db)
        self.assertEqual(sc.claim_due_jobs(spy_price=649.9, db_path=self.db), [])
        due = sc.claim_due_jobs(spy_price=650.0, db_path=self.db)
        self.assertEqual(len(due), 1)

    def test_cancel(self) -> None:
        trig = sc.parse_trigger("in 1h", now=_et(2026, 9, 18, 10, 0))
        job = sc.insert_job(legs=self._legs(), trigger=trig, db_path=self.db)
        cancelled = sc.cancel_job(int(job["id"]), db_path=self.db)
        assert cancelled is not None
        self.assertEqual(cancelled["status"], sc.STATUS_CANCELLED)
        self.assertEqual(sc.claim_due_jobs(_et(2026, 9, 18, 12, 0), db_path=self.db), [])


if __name__ == "__main__":
    unittest.main()
