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
"""
from __future__ import annotations

import fcntl
import os
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

TOKEN_DIR = Path.home() / ".tokens"
LOCK_PATH = TOKEN_DIR / "rh-login.lock"
COOLDOWN_PATH = TOKEN_DIR / "rh-login.cooldown"
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
            return False

        if is_logged_in():
            clear_login_cooldown()
            print(f"[{_now_label()}] RH login OK", flush=True)
            return True

        print(
            f"[{_now_label()}] RH login did not establish a session "
            f"(robin_stocks returned {data!r}). Cooling down "
            f"{DEFAULT_COOLDOWN_SEC}s so we don't pile more MFA polls.",
            flush=True,
        )
        set_login_cooldown()
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
