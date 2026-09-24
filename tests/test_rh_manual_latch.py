"""Tests for the manual-login latch that stops repeated MFA attempts."""
from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src" / "julia" / "rh_auth.py"
_SPEC = importlib.util.spec_from_file_location("rh_auth", _SRC)
assert _SPEC and _SPEC.loader
ra = importlib.util.module_from_spec(_SPEC)
sys.modules["rh_auth"] = ra
_SPEC.loader.exec_module(ra)


class ManualLatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        # Point every state file at the sandbox.
        ra.TOKEN_DIR = tmp
        ra.LOCK_PATH = tmp / "rh-login.lock"
        ra.COOLDOWN_PATH = tmp / "rh-login.cooldown"
        ra.MANUAL_PATH = tmp / "rh-login.manual"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_unset_by_default(self) -> None:
        self.assertIsNone(ra.manual_login_required())
        self.assertIsNone(ra.manual_login_since())

    def test_set_read_clear(self) -> None:
        ra.set_manual_login_required("429 Too Many Requests")
        self.assertEqual(ra.manual_login_required(), "429 Too Many Requests")
        self.assertIsNotNone(ra.manual_login_since())
        ra.clear_manual_login_required()
        self.assertIsNone(ra.manual_login_required())

    def test_empty_reason_gets_default(self) -> None:
        ra.set_manual_login_required("   ")
        self.assertEqual(ra.manual_login_required(), "previous login failed")

    def test_clear_is_idempotent(self) -> None:
        ra.clear_manual_login_required()
        ra.clear_manual_login_required()
        self.assertIsNone(ra.manual_login_required())

    def test_login_blocked_while_latched(self) -> None:
        """login_robinhood returns False without starting any challenge."""
        ra.set_manual_login_required("test latch")
        # is_logged_in fails fast here (no robin_stocks session), so the
        # latch check is the next gate — reaching an import of
        # robin_stocks would raise in this env, proving the gate works.
        self.assertFalse(ra.login_robinhood("user", "pass"))

    def test_cooldown_still_works_independently(self) -> None:
        ra.set_login_cooldown(60)
        self.assertGreater(ra.cooldown_remaining(), 0)
        self.assertIsNone(ra.manual_login_required())
        ra.clear_login_cooldown()
        self.assertEqual(ra.cooldown_remaining(), 0.0)


if __name__ == "__main__":
    unittest.main()
