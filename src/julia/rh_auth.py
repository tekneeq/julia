"""Serialized Robinhood login shared by poller, dashboard, scheduler, Discord.

``robin_stocks`` starts a *new* device-approval challenge every time
``rh.login()`` runs without a loaded session. If the price poller, OI
batch, Streamlit, and Discord bot all call login at once, Robinhood
rate-limits ``/push/.../get_prompts_status/`` (HTTP 429). The library
then treats a ``None`` body as a dict and crashes with
``'NoneType' object is not subscriptable``.

This module:
  * takes an exclusive file lock so only **one** process can MFA
  * lets waiters load the pickle the winner just wrote
  * cools down after 429 / failed MFA so we don't immediately stack
    more prompt-status polls
  * after ANY failed login, **latches to manual**: no process starts
    another MFA challenge until a human clears the latch (dashboard
    sidebar → Restart selected + RH login). Cached sessions keep
    working; only *new* challenges are blocked.
  * caps the library's app-push poll. ``_validate_sherrif_id`` waits
    on ``get_prompts_status`` in ``while True`` with no deadline, and
    that call sits inside ``login_lock``. A missed phone approval
    otherwise freezes the price poller for hours while its pid stays
    alive and no new ticks are written. SMS/email ``input()`` is the
    same kind of hang in a headless process, so that path fails fast
    too.
"""
from __future__ import annotations

import builtins
import fcntl
import os
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

TOKEN_DIR = Path.home() / ".tokens"
LOCK_PATH = TOKEN_DIR / "rh-login.lock"
COOLDOWN_PATH = TOKEN_DIR / "rh-login.cooldown"
# Set after a failed login; while present, automatic MFA is disabled.
MANUAL_PATH = TOKEN_DIR / "rh-login.manual"
# After a 429 / failed challenge, wait before anyone starts another.
DEFAULT_COOLDOWN_SEC = 180
SESSION_EXPIRES_SEC = 86400 * 7
# robin_stocks issues GETs with no timeout, so one dead socket can park
# a poller forever with nothing in the log. Every request gets this cap.
REQUEST_TIMEOUT_SEC = 15.0
# How long to wait for the Robinhood app push before giving up and
# latching auto-login off. Long enough to approve on a phone; short
# enough that a missed push cannot stall ticks for the rest of the day.
PROMPT_WAIT_SEC = 90.0
# Prompt statuses that will never become "validated".
_TERMINAL_PROMPT_STATUSES = frozenset(
    {"expired", "failed", "denied", "rejected"}
)
# Set by the prompt-wait wrapper; read once login() returns.
_prompt_started_at: Optional[float] = None
_prompt_failure: Optional[str] = None


def _now_label() -> str:
    return datetime.now().strftime("%H:%M:%S")


@contextmanager
def login_lock() -> Iterator[None]:
    """Block until this process is the only one allowed to MFA."""
    TOKEN_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOCK_PATH, "a+") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def cooldown_remaining() -> float:
    try:
        until = float(COOLDOWN_PATH.read_text().strip())
    except (OSError, ValueError):
        return 0.0
    return max(0.0, until - time.time())


def set_login_cooldown(seconds: float = DEFAULT_COOLDOWN_SEC) -> None:
    TOKEN_DIR.mkdir(parents=True, exist_ok=True)
    COOLDOWN_PATH.write_text(str(time.time() + float(seconds)))


def clear_login_cooldown() -> None:
    try:
        COOLDOWN_PATH.unlink()
    except OSError:
        pass


def manual_login_required() -> Optional[str]:
    """Reason string when auto-login is latched off, else None."""
    try:
        text = MANUAL_PATH.read_text().strip()
    except OSError:
        return None
    return text or "previous login failed"


def manual_login_since() -> Optional[datetime]:
    """When the manual latch was set (file mtime), else None."""
    try:
        return datetime.fromtimestamp(MANUAL_PATH.stat().st_mtime)
    except OSError:
        return None


def set_manual_login_required(reason: str) -> None:
    """Disable automatic MFA until a human clears the latch."""
    TOKEN_DIR.mkdir(parents=True, exist_ok=True)
    MANUAL_PATH.write_text(reason.strip() or "previous login failed")


def clear_manual_login_required() -> None:
    try:
        MANUAL_PATH.unlink()
    except OSError:
        pass


def install_request_timeout(seconds: float = REQUEST_TIMEOUT_SEC) -> bool:
    """Give every robin_stocks HTTP call a default timeout.

    Wraps the library's shared ``requests.Session.request`` so calls
    that pass no ``timeout`` get ``seconds``; explicit timeouts (its
    POSTs use 16s) are left alone. Idempotent. Returns False when
    robin_stocks is not importable.
    """
    try:
        from robin_stocks.robinhood import helper
    except Exception:  # noqa: BLE001
        return False
    session = getattr(helper, "SESSION", None)
    if session is None:
        return False
    if getattr(session, "_julia_timeout_sec", None) == seconds:
        return True
    original = getattr(session, "_julia_request_original", None) or session.request

    def request_with_timeout(method, url, **kwargs):
        kwargs.setdefault("timeout", seconds)
        return original(method, url, **kwargs)

    session._julia_request_original = original
    session.request = request_with_timeout
    session._julia_timeout_sec = seconds
    return True


def _reset_prompt_clock() -> None:
    """Start a fresh app-push window for the next login attempt."""
    global _prompt_started_at, _prompt_failure
    _prompt_started_at = None
    _prompt_failure = None


def take_prompt_failure() -> Optional[str]:
    """Return and clear the reason the last app-push wait was abandoned."""
    global _prompt_failure
    reason = _prompt_failure
    _prompt_failure = None
    return reason


def _remember_prompt_failure(reason: str) -> None:
    global _prompt_failure
    _prompt_failure = reason


def _prompt_url(args: tuple, kwargs: dict) -> str:
    if args:
        return str(args[0])
    return str(kwargs.get("url", ""))


def _give_up_on_prompt(reason: str) -> None:
    global _prompt_started_at
    _prompt_started_at = None
    _remember_prompt_failure(reason)
    print(f"[{_now_label()}] {reason}. Stopping the approval poll.", flush=True)
    raise TimeoutError(reason)


def _enforce_prompt_deadline(auth) -> None:
    """Raise once ``get_prompts_status`` has been polled past the cap.

    The first poll starts the clock and is allowed through, so a prompt
    that is already approved still succeeds. Later polls fail the login
    instead of spinning until the process is killed.
    """
    global _prompt_started_at
    seconds = float(getattr(auth, "_julia_prompt_wait_sec", PROMPT_WAIT_SEC))
    now = time.monotonic()
    started = _prompt_started_at
    if started is None:
        _prompt_started_at = now
        return
    if now - started >= seconds:
        _give_up_on_prompt(
            f"app-push approval not confirmed within {seconds:.0f}s"
        )


def install_prompt_deadline(seconds: Optional[float] = None) -> bool:
    """Stop robin_stocks waiting forever on an app-push or a stdin code.

    Replaces ``authentication.request_get`` so polls of
    ``get_prompts_status`` raise ``TimeoutError`` after ``seconds``
    (default :data:`PROMPT_WAIT_SEC`). Replaces ``input`` on that module
    so an SMS/email code prompt fails immediately instead of blocking a
    headless poller. Idempotent: calling again with no ``seconds`` keeps
    the cap already installed. Returns False when robin_stocks is not
    importable.

    ``rh.login`` catches the timeout and returns ``None``. Callers use
    :func:`take_prompt_failure` for the latch reason.
    """
    try:
        from robin_stocks.robinhood import authentication as auth
    except Exception:  # noqa: BLE001
        return False
    if not hasattr(auth, "request_get"):
        return False

    if getattr(auth.request_get, "_julia_prompt_bounded", False):
        if seconds is not None:
            auth._julia_prompt_wait_sec = float(seconds)
        return True

    auth._julia_prompt_wait_sec = float(
        PROMPT_WAIT_SEC if seconds is None else seconds
    )
    original_get = auth.request_get

    def request_get_bounded(*args, **kwargs):
        url = _prompt_url(args, kwargs)
        if "get_prompts_status" not in url:
            return original_get(*args, **kwargs)
        body = original_get(*args, **kwargs)
        if isinstance(body, dict):
            status = str(body.get("challenge_status") or "")
            if status == "validated":
                _reset_prompt_clock()
                return body
            if status in _TERMINAL_PROMPT_STATUSES:
                _give_up_on_prompt(f"app-push challenge {status}")
        _enforce_prompt_deadline(auth)
        return body

    request_get_bounded._julia_prompt_bounded = True
    request_get_bounded._julia_original_request_get = original_get
    auth.request_get = request_get_bounded

    if not getattr(auth, "_julia_input_bounded", False):
        original_input = getattr(auth, "input", builtins.input)

        def input_bounded(prompt=""):
            text = str(prompt).lower()
            if "verification code" in text or "enter the" in text:
                _give_up_on_prompt(
                    "Robinhood asked for a verification code on stdin; "
                    "this process will not block on it"
                )
            return original_input(prompt)

        auth.input = input_bounded
        auth._julia_input_bounded = True
    return True


def is_logged_in() -> bool:
    """True if *this process* already has a working Authorization header.

    robin_stocks swallows HTTP errors and returns ``None`` instead of
    raising, so an expired token must be detected from an empty
    profile, not from an exception.
    """
    try:
        import robin_stocks.robinhood as rh

        profile = rh.profiles.load_account_profile()
        return isinstance(profile, dict) and bool(profile)
    except Exception:
        return False


def _looks_like_rate_limit(exc: BaseException) -> bool:
    text = f"{exc!r} {exc}"
    return "429" in text or "Too Many Requests" in text


def login_robinhood(username: str, password: str) -> bool:
    """Load the cached session or run one MFA login.

    Returns True when this process has a working session afterwards.
    Concurrent callers block on ``login_lock``; the first one to finish
    writes ``~/.tokens/robinhood.pickle`` and the rest just load it.
    """
    with login_lock():
        if is_logged_in():
            return True

        reason = manual_login_required()
        if reason is not None:
            print(
                f"[{_now_label()}] RH auto-login is OFF (latched after: "
                f"{reason}). No new device challenge will be started. "
                "Re-enable via the dashboard sidebar → 'Restart selected "
                "+ RH login' (or julia.rh_auth.clear_manual_login_required).",
                flush=True,
            )
            return False

        wait = cooldown_remaining()
        if wait > 0:
            print(
                f"[{_now_label()}] RH login cooling down {wait:.0f}s after a "
                "recent 429 / failed MFA — not starting another device "
                "challenge.",
                flush=True,
            )
            return False

        import robin_stocks.robinhood as rh

        print(
            f"[{_now_label()}] RH login — this process holds the login lock; "
            "approve the Robinhood app push if one appears. Other workers "
            "are waiting (not starting extra MFA polls).",
            flush=True,
        )
        if not install_prompt_deadline():
            print(
                f"[{_now_label()}] could not cap the Robinhood app-push wait",
                flush=True,
            )
        # A previous attempt in this process must not eat the new window.
        _reset_prompt_clock()
        data = None
        try:
            data = rh.login(
                username=username,
                password=password,
                store_session=True,
                expiresIn=SESSION_EXPIRES_SEC,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[{_now_label()}] RH login error: {exc!r}", flush=True)
            extra = 120 if _looks_like_rate_limit(exc) else 0
            set_login_cooldown(DEFAULT_COOLDOWN_SEC + extra)
            set_manual_login_required(
                take_prompt_failure()
                or f"{type(exc).__name__}: {exc}"[:200]
            )
            print(
                f"[{_now_label()}] Auto-login now DISABLED so no more "
                "challenges stack up. Re-enable from the dashboard: "
                "Services → Restart selected + RH login.",
                flush=True,
            )
            return False

        if is_logged_in():
            clear_login_cooldown()
            clear_manual_login_required()
            print(f"[{_now_label()}] RH login OK", flush=True)
            return True

        print(
            f"[{_now_label()}] RH login did not establish a session "
            f"(robin_stocks returned {data!r}). Auto-login now DISABLED "
            "so we don't pile more MFA polls — re-enable from the "
            "dashboard: Services → Restart selected + RH login.",
            flush=True,
        )
        set_login_cooldown()
        set_manual_login_required(
            take_prompt_failure()
            or "login returned no session (missed MFA?)"
        )
        return False


def ensure_robinhood_login() -> bool:
    """Best-effort login from ``RH_USERNAME`` / ``RH_PASSWORD``."""
    install_request_timeout()
    install_prompt_deadline()
    if is_logged_in():
        return True
    username = os.getenv("RH_USERNAME")
    password = os.getenv("RH_PASSWORD")
    if not username or not password:
        return False
    return login_robinhood(username, password)


def load_session_if_present() -> bool:
    """Try the pickle without starting MFA when a cooldown is active.

    Used by callers that only need a cheap session check. If no pickle
    is loaded yet, this still goes through ``login_robinhood`` (lock +
    cooldown) so we never start a second challenge during a 429 window.
    """
    return ensure_robinhood_login()
