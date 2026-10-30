"""Price poller must not stall silently on a hung socket or dead token."""
from __future__ import annotations

import importlib.util
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_RA_SRC = _ROOT / "src" / "julia" / "rh_auth.py"
_PP_SRC = _ROOT / "scripts" / "price_poller.py"


class _FakeSession:
    """Stands in for robin_stocks' module-level ``requests.Session``."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict]] = []
        self.headers: dict = {}

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return {"method": method, "url": url, **kwargs}

    # Mirror requests.Session: get/post route through request().
    def get(self, url, **kwargs):
        return self.request("GET", url, **kwargs)

    def post(self, url, data=None, json=None, **kwargs):
        return self.request("POST", url, data=data, json=json, **kwargs)


def _install_fake_robin_stocks(profile=None):
    """Register a minimal robin_stocks package in sys.modules."""
    pkg = types.ModuleType("robin_stocks")
    rh = types.ModuleType("robin_stocks.robinhood")
    helper = types.ModuleType("robin_stocks.robinhood.helper")
    profiles = types.ModuleType("robin_stocks.robinhood.profiles")
    stocks = types.ModuleType("robin_stocks.robinhood.stocks")
    helper.SESSION = _FakeSession()
    profiles.load_account_profile = lambda *a, **k: profile
    stocks.get_latest_price = lambda *a, **k: [None]
    rh.helper = helper
    rh.profiles = profiles
    rh.stocks = stocks
    pkg.robinhood = rh
    sys.modules["robin_stocks"] = pkg
    sys.modules["robin_stocks.robinhood"] = rh
    sys.modules["robin_stocks.robinhood.helper"] = helper
    sys.modules["robin_stocks.robinhood.profiles"] = profiles
    sys.modules["robin_stocks.robinhood.stocks"] = stocks
    return rh


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class RequestTimeoutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rh = _install_fake_robin_stocks(profile={"url": "x"})
        self.ra = _load("rh_auth_t", _RA_SRC)

    def tearDown(self) -> None:
        for k in list(sys.modules):
            if k.startswith("robin_stocks") or k in ("rh_auth_t", "julia.rh_auth"):
                sys.modules.pop(k, None)

    def test_get_without_timeout_gets_default(self) -> None:
        self.assertTrue(self.ra.install_request_timeout(7.5))
        res = self.rh.helper.SESSION.get("https://rh/quotes", params={"s": "SPY"})
        self.assertEqual(res["timeout"], 7.5)
        self.assertEqual(res["params"], {"s": "SPY"})

    def test_explicit_timeout_is_kept(self) -> None:
        self.ra.install_request_timeout(7.5)
        res = self.rh.helper.SESSION.post("https://rh/login", data={"a": 1}, timeout=16)
        self.assertEqual(res["timeout"], 16)

    def test_idempotent_and_no_double_wrap(self) -> None:
        self.ra.install_request_timeout(5)
        self.ra.install_request_timeout(5)
        self.ra.install_request_timeout(9)
        res = self.rh.helper.SESSION.get("https://rh/x")
        self.assertEqual(res["timeout"], 9)
        # Exactly one real call reached the underlying request().
        self.assertEqual(len(self.rh.helper.SESSION.calls), 1)

    def test_missing_library_is_false(self) -> None:
        for k in list(sys.modules):
            if k.startswith("robin_stocks"):
                sys.modules.pop(k)
        sys.modules["robin_stocks"] = None  # type: ignore[assignment]
        try:
            self.assertFalse(self.ra.install_request_timeout())
        finally:
            sys.modules.pop("robin_stocks", None)


class IsLoggedInTests(unittest.TestCase):
    def tearDown(self) -> None:
        for k in list(sys.modules):
            if k.startswith("robin_stocks") or k == "rh_auth_t":
                sys.modules.pop(k, None)

    def test_profile_dict_means_logged_in(self) -> None:
        _install_fake_robin_stocks(profile={"url": "acct"})
        ra = _load("rh_auth_t", _RA_SRC)
        self.assertTrue(ra.is_logged_in())

    def test_none_profile_means_expired_token(self) -> None:
        # robin_stocks returns None on a 401 instead of raising.
        _install_fake_robin_stocks(profile=None)
        ra = _load("rh_auth_t", _RA_SRC)
        self.assertFalse(ra.is_logged_in())


class _Timeout(Exception):
    pass


class PollerFailureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rh = _install_fake_robin_stocks(profile={"url": "x"})
        # price_poller imports julia.daily_moves_store at module load.
        julia = types.ModuleType("julia")
        dms = types.ModuleType("julia.daily_moves_store")
        dms.record_tick = lambda **k: None
        dms.get_ticks = lambda *a, **k: []
        dms.latest_tick = lambda *a, **k: None
        dms.prune_ticks = lambda *a, **k: None
        julia.daily_moves_store = dms
        sys.modules["julia"] = julia
        sys.modules["julia.daily_moves_store"] = dms
        # A requests stand-in so the classifier can isinstance-check.
        requests = types.ModuleType("requests")
        exceptions = types.ModuleType("requests.exceptions")
        exceptions.Timeout = _Timeout
        exceptions.ConnectionError = type("ConnectionError", (Exception,), {})
        requests.exceptions = exceptions
        sys.modules["requests"] = requests
        sys.modules["requests.exceptions"] = exceptions
        self.pp = _load("price_poller_t", _PP_SRC)
        self.pp._login_ok = True
        self.pp._empty_quotes = 0

    def tearDown(self) -> None:
        for k in list(sys.modules):
            if k.startswith(("robin_stocks", "julia", "requests")) or k == "price_poller_t":
                sys.modules.pop(k, None)

    def test_price_resets_empty_streak(self) -> None:
        self.rh.stocks.get_latest_price = lambda *a, **k: ["761.25"]
        self.pp._empty_quotes = 2
        price, why = self.pp._fetch_last_price("SPY")
        self.assertEqual(price, 761.25)
        self.assertIsNone(why)
        self.assertEqual(self.pp._empty_quotes, 0)
        self.assertTrue(self.pp._login_ok)

    def test_empty_quotes_force_session_recheck(self) -> None:
        self.rh.stocks.get_latest_price = lambda *a, **k: [None]
        for _ in range(self.pp.EMPTY_QUOTE_RELOGIN_AFTER - 1):
            price, why = self.pp._fetch_last_price("SPY")
            self.assertIsNone(price)
            self.assertEqual(why, self.pp.FAIL_EMPTY)
            self.assertTrue(self.pp._login_ok)
        self.pp._fetch_last_price("SPY")
        self.assertFalse(self.pp._login_ok)

    def test_timeout_is_network_and_keeps_session(self) -> None:
        def boom(*a, **k):
            raise _Timeout("read timed out")

        self.rh.stocks.get_latest_price = boom
        price, why = self.pp._fetch_last_price("SPY")
        self.assertIsNone(price)
        self.assertEqual(why, self.pp.FAIL_NETWORK)
        self.assertTrue(self.pp._login_ok)

    def test_rate_limit_keeps_session(self) -> None:
        def boom(*a, **k):
            raise Exception("429 Client Error: Too Many Requests")

        self.rh.stocks.get_latest_price = boom
        _price, why = self.pp._fetch_last_price("SPY")
        self.assertEqual(why, self.pp.FAIL_RATE_LIMIT)
        self.assertTrue(self.pp._login_ok)

    def test_other_error_drops_session(self) -> None:
        def boom(*a, **k):
            raise ValueError("weird payload")

        self.rh.stocks.get_latest_price = boom
        _price, why = self.pp._fetch_last_price("SPY")
        self.assertEqual(why, self.pp.FAIL_ERROR)
        self.assertFalse(self.pp._login_ok)


PROMPT_URL = "https://api.robinhood.com/push/abc/get_prompts_status/"


class PromptDeadlineTests(unittest.TestCase):
    """A missed app-push must not spin inside the login lock."""

    def setUp(self) -> None:
        self.rh = _install_fake_robin_stocks(profile=None)
        self.auth = types.ModuleType("robin_stocks.robinhood.authentication")
        self.gets: list[str] = []
        self._bodies: list[dict] = []

        def request_get(*args, **kwargs):
            url = args[0] if args else kwargs.get("url", "")
            self.gets.append(str(url))
            if self._bodies:
                return self._bodies.pop(0)
            return {"challenge_status": "issued"}

        def original_input(prompt=""):
            return "typed"

        self.auth.request_get = request_get
        self.auth.input = original_input
        self.rh.authentication = self.auth
        sys.modules["robin_stocks.robinhood.authentication"] = self.auth

        def rh_login(**kwargs):
            # Same shape as robin_stocks 3.4.0 ``_validate_sherrif_id``:
            # poll until validated, with no deadline of its own.
            # login() catches that error and returns None.
            try:
                for _ in range(30):
                    body = self.auth.request_get(url=PROMPT_URL)
                    if (
                        isinstance(body, dict)
                        and body.get("challenge_status") == "validated"
                    ):
                        return {"access_token": "tok"}
                return {"access_token": "gave-up-uncapped"}
            except Exception:
                return None

        self.rh.login = rh_login
        self.ra = _load("rh_auth_t", _RA_SRC)
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        self.ra.TOKEN_DIR = tmp
        self.ra.LOCK_PATH = tmp / "rh-login.lock"
        self.ra.COOLDOWN_PATH = tmp / "rh-login.cooldown"
        self.ra.MANUAL_PATH = tmp / "rh-login.manual"

    def tearDown(self) -> None:
        self._tmp.cleanup()
        for k in list(sys.modules):
            if k.startswith("robin_stocks") or k == "rh_auth_t":
                sys.modules.pop(k, None)

    def test_uncapped_prompt_loop_stops_and_latches(self) -> None:
        self.assertTrue(self.ra.install_prompt_deadline(0))
        started = time.monotonic()
        self.assertFalse(self.ra.login_robinhood("user", "pass"))
        self.assertLess(time.monotonic() - started, 2.0)
        reason = self.ra.manual_login_required() or ""
        self.assertIn("app-push approval not confirmed", reason)
        # First poll is allowed; the second hits the deadline.
        self.assertEqual(self.gets, [PROMPT_URL, PROMPT_URL])

    def test_validated_prompt_is_not_cut_off(self) -> None:
        self._bodies = [
            {"challenge_status": "issued"},
            {"challenge_status": "validated"},
        ]
        self.ra.install_prompt_deadline(0)
        self.assertEqual(
            self.auth.request_get(url=PROMPT_URL)["challenge_status"], "issued"
        )
        self.assertEqual(
            self.auth.request_get(url=PROMPT_URL)["challenge_status"],
            "validated",
        )

    def test_terminal_prompt_fails_immediately(self) -> None:
        self._bodies = [{"challenge_status": "expired"}]
        self.ra.install_prompt_deadline(90)
        with self.assertRaises(TimeoutError):
            self.auth.request_get(url=PROMPT_URL)
        self.assertEqual(len(self.gets), 1)
        self.assertIn("expired", self.ra.take_prompt_failure() or "")

    def test_other_gets_are_not_capped(self) -> None:
        self.ra.install_prompt_deadline(0)
        quote = "https://api.robinhood.com/quotes/SPY/"
        self.auth.request_get(quote)
        self.auth.request_get(quote)
        self.assertEqual(self.gets, [quote, quote])

    def test_stdin_code_does_not_block(self) -> None:
        self.ra.install_prompt_deadline(90)
        with self.assertRaises(TimeoutError):
            self.auth.input(
                "Enter the sms verification code sent to your device: "
            )
        self.assertEqual(self.auth.input("Robinhood username: "), "typed")
        self.assertIn("stdin", self.ra.take_prompt_failure() or "")

    def test_reinstall_without_seconds_keeps_the_cap(self) -> None:
        self.ra.install_prompt_deadline(0)
        self.assertTrue(self.ra.install_prompt_deadline())
        self.assertEqual(self.auth._julia_prompt_wait_sec, 0)
        self.auth.request_get(url=PROMPT_URL)
        with self.assertRaises(TimeoutError):
            self.auth.request_get(url=PROMPT_URL)

    def test_missing_library_is_false(self) -> None:
        for k in list(sys.modules):
            if k.startswith("robin_stocks"):
                sys.modules.pop(k)
        sys.modules["robin_stocks"] = None  # type: ignore[assignment]
        try:
            self.assertFalse(self.ra.install_prompt_deadline())
        finally:
            sys.modules.pop("robin_stocks", None)


if __name__ == "__main__":
    unittest.main()
