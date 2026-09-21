"""Unit tests for GEX flip-print math."""
from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from datetime import date
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src" / "julia" / "gex_flip.py"
_SPEC = importlib.util.spec_from_file_location("gex_flip", _SRC)
assert _SPEC and _SPEC.loader
gf = importlib.util.module_from_spec(_SPEC)
sys.modules["gex_flip"] = gf
_SPEC.loader.exec_module(gf)


def _contract(k: float, sigma: float, oi: int, typ: str) -> dict:
    return {
        "K": k,
        "sigma": sigma,
        "oi": oi,
        "sign": -1.0 if typ == "call" else 1.0,
    }


class YearsToExpiryTests(unittest.TestCase):
    def test_same_day_floors_to_one_365(self) -> None:
        self.assertAlmostEqual(
            gf.years_to_expiry("2026-09-21", date(2026, 9, 21)),
            1.0 / 365.0,
        )

    def test_five_days(self) -> None:
        self.assertAlmostEqual(
            gf.years_to_expiry("2026-09-26", date(2026, 9, 21)),
            5.0 / 365.0,
        )


class ContractsFromRowsTests(unittest.TestCase):
    def test_skips_zero_oi_and_bad_iv(self) -> None:
        rows = [
            {"option_type": "call", "strike_price": 100, "implied_vol": 0.2, "open_interest": 10},
            {"option_type": "put", "strike_price": 90, "implied_vol": 0.2, "open_interest": 0},
            {"option_type": "put", "strike_price": 95, "implied_vol": None, "open_interest": 5},
            {"option_type": "put", "strike_price": 95, "implied_vol": 0.18, "open_interest": 7},
        ]
        cs = gf.contracts_from_strike_rows(rows)
        self.assertEqual(len(cs), 2)
        self.assertEqual(cs[0]["sign"], -1.0)
        self.assertEqual(cs[1]["sign"], 1.0)
        self.assertEqual(cs[1]["oi"], 7)


class FormulaTests(unittest.TestCase):
    def test_matches_option_pricer_identity(self) -> None:
        """gex = sign * n(d1) * OI / (sigma * sqrt(T)) at q=r=0."""
        S, K, T, sigma, oi = 100.0, 100.0, 1.0 / 365.0, 0.20, 1000
        d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
        pdf = math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
        expected = -pdf * oi / (sigma * math.sqrt(T))  # call
        got = gf.net_gex_at_spots(
            [_contract(K, sigma, oi, "call")],
            np.array([S]),
            T=T,
            r=0.0,
        )
        self.assertAlmostEqual(float(got[0]), expected, places=4)


class FlipFindTests(unittest.TestCase):
    def test_put_below_call_above_crosses_zero(self) -> None:
        # Equal OI / IV; ATM gamma follows spot. Below 660 puts win (GEX+);
        # above 660 calls win (GEX−).
        T = 1.0 / 365.0
        contracts = [
            _contract(650.0, 0.18, 5000, "put"),
            _contract(670.0, 0.18, 5000, "call"),
        ]
        flips = gf.find_gex_flips(contracts, spot=660.0, T=T, r=0.0)
        self.assertIsNotNone(flips["zero"])
        self.assertAlmostEqual(flips["zero"], 660.0, delta=1.50)

        low = float(gf.net_gex_at_spots(
            contracts, np.array([640.0]), T=T, r=0.0,
        )[0])
        high = float(gf.net_gex_at_spots(
            contracts, np.array([680.0]), T=T, r=0.0,
        )[0])
        self.assertGreater(low, 0.0)
        self.assertLess(high, 0.0)

    def test_crossings_interpolate(self) -> None:
        spots = np.array([10.0, 20.0, 30.0])
        values = np.array([-4.0, 2.0, 8.0])
        hits = gf.crossings_of(spots, values, 0.0)
        self.assertEqual(len(hits), 1)
        # -4 at 10, +2 at 20 → zero at 10 + 4/6 * 10 = 16.666...
        self.assertAlmostEqual(hits[0], 10.0 + (4.0 / 6.0) * 10.0, places=6)

    def test_no_contracts(self) -> None:
        flips = gf.find_gex_flips([], spot=660.0, T=1 / 365, r=0.02)
        self.assertIsNone(flips["zero"])
        self.assertIsNone(flips["net_at_spot"])


if __name__ == "__main__":
    unittest.main()
