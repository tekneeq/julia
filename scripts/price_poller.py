#!/usr/bin/env -S uv run python
"""Dedicated live price poller for the "Today's price action" chart.

Polls Robinhood for the latest regular-session price every N seconds
(default 15) during market hours and stores each print in the
``spot_ticks`` table. This is deliberately independent of
``oi_scheduler.py`` — the OI batch is heavyweight (full options chains,
PNG rendering) and fires every 30 minutes, which is far too coarse for
a precise intraday price chart. This poller does exactly one quote
request per ticker per interval and nothing else.

The Streamlit dashboard reads these ticks directly for the live session
chart, and also distills them into per-minute points (source='tick')
that feed the historical-twin matcher.

Runs forever in the foreground. Ctrl-C to stop. Handles:
  * market-hours gating (09:30–16:00 ET, Mon–Fri; sleeps until the
    next open otherwise)
  * interval alignment (fires at :00 / :15 / :30 / :45 of each minute
    for interval=15, so tick spacing stays even)
  * Robinhood login (cached ~/.tokens or RH_USERNAME / RH_PASSWORD),
    with re-login and backoff on transient failures
  * clean shutdown on SIGINT / SIGTERM, pid-file management

Examples
--------
    # Start with the defaults (SPY every 15s)
    ./scripts/price_poller.py

    # Finer cadence, more tickers
    ./scripts/price_poller.py --tickers SPY,QQQ --interval-sec 10

    # One poll right now (auth smoke test), then exit
    ./scripts/price_poller.py --once

    # Is it up? How many ticks landed today?
    ./scripts/price_poller.py --status

    # Replace a running poller / stop it
    ./scripts/price_poller.py --replace
    ./scripts/price_poller.py --stop

Background
----------
    nohup ./scripts/price_poller.py > logs/price-poller.out 2>&1 &
    # or, from the host: ./restart-price-poller.sh
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PID_FILE = REPO_ROOT / "logs" / "price-poller.pid"

sys.path.insert(0, str(REPO_ROOT / "src"))
from julia import daily_moves_store  # noqa: E402

# Regular session in local (container) time — Dockerfile pins ET.
MARKET_OPEN = dtime(9, 30)
# Poll slightly past 16:00 so the official closing print is captured.
POLL_END = dtime(16, 1)


# ---------------------------------------------------------------------------
# PID file (same pattern as oi_scheduler.py)
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


def _running_pid(pid_file: Path) -> int | None:
    pid = _read_pid(pid_file)
    if pid is None or pid == os.getpid() or not _pid_alive(pid):
        return None
    return pid


def _write_pid(pid_file: Path) -> None:
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(f"{os.getpid()}\n")


def _clear_pid(pid_file: Path) -> None:
    if _read_pid(pid_file) == os.getpid():
        pid_file.unlink(missing_ok=True)


def _stop_running(pid_file: Path, timeout: float = 15.0) -> bool:
    pid = _running_pid(pid_file)
    if pid is None:
        return False
    print(f"[{datetime.now():%H:%M:%S}] stopping price poller pid {pid}…")
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


# ---------------------------------------------------------------------------
# Robinhood
# ---------------------------------------------------------------------------

_login_ok = False


def _ensure_login() -> bool:
    """Best-effort login, reusing the CLI's cached token or env creds."""
    global _login_ok
    if _login_ok:
        return True
    try:
        from julia.main import is_logged_in, login_robinhood
        if is_logged_in():
            _login_ok = True
            return True
        username = os.getenv("RH_USERNAME")
        password = os.getenv("RH_PASSWORD")
        if username and password:
            login_robinhood(username, password)
            _login_ok = is_logged_in()
            return _login_ok
    except Exception as e:  # noqa: BLE001
        print(f"[{datetime.now():%H:%M:%S}] login failed: {e!r}")
    return False


def _fetch_last_price(ticker: str) -> float | None:
    """Latest regular-session print, or None on any failure."""
    global _login_ok
    try:
        import robin_stocks.robinhood as rh
        vals = rh.stocks.get_latest_price(
            ticker, priceType=None, includeExtendedHours=False
        )
        if vals and vals[0]:
            return float(vals[0])
    except Exception as e:  # noqa: BLE001
        # Force a re-login attempt on the next cycle — an expired token
        # surfaces here as a generic request error.
        _login_ok = False
        print(f"[{datetime.now():%H:%M:%S}] {ticker}: fetch failed: {e!r}")
    return None


# ---------------------------------------------------------------------------
# Session clock
# ---------------------------------------------------------------------------

def _session_open(d: date) -> datetime:
    return datetime.combine(d, MARKET_OPEN)


def _next_session_open(now: datetime) -> datetime:
    d = now.date()
    if now.weekday() < 5 and now.time() < MARKET_OPEN:
        return _session_open(d)
    d += timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return _session_open(d)


def _next_aligned(now: datetime, interval_sec: int) -> datetime:
    """The next wall-clock instant aligned to the interval grid."""
    seconds_today = now.hour * 3600 + now.minute * 60 + now.second
    next_s = (seconds_today // interval_sec + 1) * interval_sec
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return midnight + timedelta(seconds=next_s)


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

def _fmt_age(seconds: float) -> str:
    secs = int(seconds)
    if secs < 60:
        return f"{secs}s ago"
    if secs < 3600:
        return f"{secs // 60}m{secs % 60:02d}s ago"
    return f"{secs // 3600}h{(secs % 3600) // 60:02d}m ago"


def _status_report(tickers: list[str], pid_file: Path) -> int:
    now = datetime.now().astimezone()
    print(f"price-poller status  ·  {now:%a %Y-%m-%d %H:%M:%S}")
    pid = _running_pid(pid_file)
    if pid:
        print(f"\nProcess: ● RUNNING (pid {pid}, pid-file {pid_file.name})")
    else:
        stale = _read_pid(pid_file)
        extra = f" — stale pid-file points at {stale}" if stale else ""
        print(f"\nProcess: ○ NOT RUNNING{extra}")
        print("         start it with  ./restart-price-poller.sh")
    print()
    for ticker in tickers:
        ticks = daily_moves_store.get_ticks(ticker, now.date())
        last = daily_moves_store.latest_tick(ticker)
        if ticks:
            age = _fmt_age((now - ticks[-1]["ts"]).total_seconds())
            print(
                f"  {ticker:<6} today: {len(ticks):>5} ticks  "
                f"{ticks[0]['ts']:%H:%M:%S} → {ticks[-1]['ts']:%H:%M:%S}  "
                f"last ${ticks[-1]['price']:.2f} ({age})"
            )
        elif last:
            print(
                f"  {ticker:<6} today: 0 ticks  "
                f"(last ever: {last['session_date']} ${last['price']:.2f})"
            )
        else:
            print(f"  {ticker:<6} no ticks recorded yet")
    return 0


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

_shutdown = False


def _handle_signal(signum, _frame):
    global _shutdown
    print(f"\n[{datetime.now():%H:%M:%S}] received signal {signum}, shutting down…")
    _shutdown = True


def _sleep_until(target: datetime) -> None:
    while not _shutdown:
        remaining = (target - datetime.now()).total_seconds()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 2.0))


def _poll_once(tickers: list[str]) -> int:
    """Poll each ticker once; returns the number of ticks recorded."""
    if not _ensure_login():
        return 0
    recorded = 0
    for ticker in tickers:
        price = _fetch_last_price(ticker)
        if price is None or price <= 0:
            continue
        ts = datetime.now(timezone.utc)
        daily_moves_store.record_tick(ticker=ticker, ts=ts, price=price)
        recorded += 1
        print(
            f"[{ts.astimezone():%H:%M:%S}] {ticker} ${price:.2f}"
        )
    return recorded


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Poll live prices every few seconds during market hours — "
            "the high-resolution feed for the dashboard's session chart."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--tickers", default="SPY",
        help="Comma-separated tickers to poll (default: SPY).",
    )
    p.add_argument(
        "--interval-sec", type=int, default=15,
        help="Seconds between polls during market hours (default: 15).",
    )
    p.add_argument(
        "--pid-file", default=str(DEFAULT_PID_FILE),
        help=f"Where to track the running poller (default: {DEFAULT_PID_FILE}).",
    )
    p.add_argument(
        "--replace", action="store_true",
        help="Stop an already-running poller before starting (restart).",
    )
    p.add_argument(
        "--stop", action="store_true",
        help="Stop the running poller and exit.",
    )
    p.add_argument(
        "--status", action="store_true",
        help="Show whether the poller is up and today's tick coverage.",
    )
    p.add_argument(
        "--once", action="store_true",
        help="Poll each ticker once right now (ignores market hours) and "
             "exit — handy as an auth smoke test.",
    )
    args = p.parse_args()

    # Detached runs redirect stdout to a file; line-buffer it so
    # `tail -f logs/price-poller.out` shows progress live.
    sys.stdout.reconfigure(line_buffering=True)

    if args.interval_sec < 5:
        p.error("--interval-sec must be >= 5 (be nice to the API).")

    pid_file = Path(args.pid_file)
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    if not tickers:
        p.error("--tickers must name at least one symbol.")

    if args.status:
        return _status_report(tickers, pid_file)

    if args.stop:
        if not _stop_running(pid_file):
            print(f"No poller running (pid-file {pid_file}).")
        return 0

    if args.once:
        n = _poll_once(tickers)
        print(f"recorded {n}/{len(tickers)} tick(s)")
        return 0 if n == len(tickers) else 1

    if args.replace:
        _stop_running(pid_file)
    elif (existing := _running_pid(pid_file)) is not None:
        print(
            f"A price poller is already running (pid {existing}). "
            f"Use --replace to restart it, or --stop to shut it down.",
            file=sys.stderr,
        )
        return 1

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    _write_pid(pid_file)

    print(
        f"[{datetime.now():%Y-%m-%d %H:%M:%S}] price poller starting  "
        f"tickers={','.join(tickers)}  interval={args.interval_sec}s  "
        f"window={MARKET_OPEN:%H:%M}–{POLL_END:%H:%M} ET Mon–Fri"
    )

    consecutive_failures = 0
    pruned_for: date | None = None
    try:
        while not _shutdown:
            now = datetime.now()
            in_session = (
                now.weekday() < 5
                and MARKET_OPEN <= now.time() < POLL_END
            )
            if not in_session:
                # Once per day, right after the session, trim old ticks.
                if pruned_for != now.date():
                    for ticker in tickers:
                        daily_moves_store.prune_ticks(ticker)
                    pruned_for = now.date()
                nxt = _next_session_open(now)
                print(
                    f"[{now:%H:%M:%S}] market closed — sleeping until "
                    f"{nxt:%a %Y-%m-%d %H:%M}"
                )
                _sleep_until(nxt)
                continue

            recorded = _poll_once(tickers)
            if recorded == 0:
                consecutive_failures += 1
                # Auth or network trouble: back off so a broken login
                # doesn't hammer the API 4×/minute all session long.
                if consecutive_failures >= 3:
                    backoff = min(60 * 2 ** (consecutive_failures - 3), 300)
                    print(
                        f"[{datetime.now():%H:%M:%S}] "
                        f"{consecutive_failures} failed polls — backing "
                        f"off {backoff}s"
                    )
                    _sleep_until(datetime.now() + timedelta(seconds=backoff))
                    continue
            else:
                consecutive_failures = 0

            _sleep_until(_next_aligned(datetime.now(), args.interval_sec))
    finally:
        _clear_pid(pid_file)

    print(f"[{datetime.now():%H:%M:%S}] price poller stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
