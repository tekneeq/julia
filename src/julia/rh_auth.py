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
"""
from __future__ import annotations

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


def is_logged_in() -> bool:
    """True if *this process* already has a working Authorization header."""
    try:
        import robin_stocks.robinhood as rh

        rh.profiles.load_account_profile()
        return True
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
                f"{type(exc).__name__}: {exc}"[:200]
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
        set_manual_login_required("login returned no session (missed MFA?)")
        return False


def ensure_robinhood_login() -> bool:
    """Best-effort login from ``RH_USERNAME`` / ``RH_PASSWORD``."""
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
