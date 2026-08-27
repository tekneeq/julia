#!/usr/bin/env -S uv run python
"""Managed Discord stock-bot process for the julia EC2 host.

Wraps ``julia.discorder`` with the same PID / --status / --stop / --replace
lifecycle as ``price_poller.py`` so ``./restart-discord-bot.sh`` (and
``deploy.sh``) can start it detached inside the dashboard container.

Examples
--------
    ./scripts/discord_bot.py --status
    ./scripts/discord_bot.py --replace
    ./scripts/discord_bot.py --stop

    # From the host (preferred):
    ./restart-discord-bot.sh
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PID_FILE = REPO_ROOT / "logs" / "discord-bot.pid"

sys.path.insert(0, str(REPO_ROOT / "src"))

_PID_MARKER = "discord_bot.py"


# ---------------------------------------------------------------------------
# PID file (same pattern as price_poller.py / oi_scheduler.py)
# ---------------------------------------------------------------------------

def _read_pid(pid_file: Path) -> int | None:
    try:
        return int(pid_file.read_text().strip())
    except (OSError, ValueError):
        return None


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _pid_cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\x00", b" ").decode(errors="replace")


def _is_our_process(pid: int) -> bool:
    return _PID_MARKER in _pid_cmdline(pid)


def _scrub_stale_pid(pid_file: Path) -> None:
    pid = _read_pid(pid_file)
    if pid is None or pid == os.getpid():
        return
    if not _pid_alive(pid) or not _is_our_process(pid):
        pid_file.unlink(missing_ok=True)


def _running_pid(pid_file: Path) -> int | None:
    pid = _read_pid(pid_file)
    if pid is None or pid == os.getpid() or not _pid_alive(pid):
        return None
    if not _is_our_process(pid):
        return None
    return pid


def _write_pid(pid_file: Path) -> None:
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(f"{os.getpid()}\n")


def _clear_pid(pid_file: Path) -> None:
    if _read_pid(pid_file) == os.getpid():
        pid_file.unlink(missing_ok=True)


def _stop_running(pid_file: Path, timeout: float = 15.0) -> bool:
    _scrub_stale_pid(pid_file)
    pid = _running_pid(pid_file)
    if pid is None:
        return False
    print(f"[{datetime.now():%H:%M:%S}] stopping discord bot pid {pid}…")
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as e:
        print(f"  couldn't signal pid {pid}: {e!r}")
        return False
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _pid_alive(pid):
            print(f"  pid {pid} exited.")
            return True
        time.sleep(0.25)
    print(f"  pid {pid} still alive after {timeout:.0f}s — SIGKILL.")
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass
    return True


def _status_report(pid_file: Path) -> int:
    now = datetime.now().astimezone()
    print(f"discord-bot status  ·  {now:%a %Y-%m-%d %H:%M:%S}")
    token = os.getenv("DISCORD_BOT_TOKEN")
    print(f"Token:  {'set' if token else 'MISSING (set DISCORD_BOT_TOKEN in .env)'}")
    channel = os.getenv("DISCORD_CHANNEL_ID")
    print(f"Channel greeting: {channel or '(none)'}")
    pid = _running_pid(pid_file)
    if pid:
        print(f"\nProcess: ● RUNNING (pid {pid}, pid-file {pid_file.name})")
    else:
        stale = _read_pid(pid_file)
        extra = f" — stale pid-file points at {stale}" if stale else ""
        print(f"\nProcess: ○ NOT RUNNING{extra}")
        print("         start it with  ./restart-discord-bot.sh")
    try:
        from julia import tickers_store

        n = tickers_store.ticker_count()
        synced = tickers_store.get_meta("last_synced_at") or "never"
        print(f"\nTicker cache: {n:,} symbols  ·  last synced {synced}")
    except Exception as e:  # noqa: BLE001
        print(f"\nTicker cache: unavailable ({e!r})")
    return 0 if pid else 1


def main(argv: list[str] | None = None) -> int:
    from dotenv import load_dotenv

    load_dotenv()

    p = argparse.ArgumentParser(
        description="Julia Discord stock bot (PID-managed)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--pid-file",
        default=str(DEFAULT_PID_FILE),
        help=f"PID file path (default: {DEFAULT_PID_FILE})",
    )
    p.add_argument(
        "--replace",
        action="store_true",
        help="Stop any running bot before starting",
    )
    p.add_argument(
        "--stop",
        action="store_true",
        help="Stop the running bot and exit",
    )
    p.add_argument(
        "--status",
        action="store_true",
        help="Print running status and exit",
    )
    args = p.parse_args(argv)
    pid_file = Path(args.pid_file)

    if args.status:
        return _status_report(pid_file)

    if args.stop:
        if not _stop_running(pid_file):
            print(f"No discord bot running (pid-file {pid_file}).")
        return 0

    if not os.getenv("DISCORD_BOT_TOKEN"):
        print(
            "DISCORD_BOT_TOKEN is not set — refusing to start. "
            "Add it to the host .env (see README).",
            file=sys.stderr,
        )
        return 2

    _scrub_stale_pid(pid_file)
    if args.replace:
        _stop_running(pid_file)
    elif (existing := _running_pid(pid_file)) is not None:
        print(
            f"A discord bot is already running (pid {existing}). "
            f"Use --replace to restart it, or --stop to shut it down.",
            file=sys.stderr,
        )
        return 1

    _write_pid(pid_file)

    def _on_signal(signum, _frame):  # noqa: ANN001
        print(f"[{datetime.now():%H:%M:%S}] got signal {signum}, shutting down…")
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    try:
        from julia.discorder import run_bot

        print(f"[{datetime.now():%H:%M:%S}] starting discord stock bot…")
        run_bot()
    finally:
        _clear_pid(pid_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
