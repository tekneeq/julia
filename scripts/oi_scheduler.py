#!/usr/bin/env -S uv run python
"""Local scheduler for `oi_batch.py` — a stdlib alternative to cron.

Fires `oi_batch.py --rolling-days N --tickers ...` at fixed slots
during market hours (default: every 30 min, 9:30am–4:00pm ET, Mon–Fri),
plus one settlement run at 4:05pm.

Runs forever in the foreground. Ctrl-C to stop. No third-party deps —
pure stdlib. Handles:
  * clock alignment (fires at :00 / :30 exactly, not drifting)
  * weekend + off-hours skipping
  * date rollover across midnight / weekend / after-close
  * clean shutdown on SIGINT / SIGTERM

Examples
--------
    # Start with the defaults (SPY, 5 business days ahead)
    ./scripts/oi_scheduler.py

    # Different ticker set, longer window
    ./scripts/oi_scheduler.py --tickers SPY,QQQ --days-ahead 7

    # Custom cadence — fire every 15 min instead of 30
    ./scripts/oi_scheduler.py --interval-min 15

    # Preview: show the next 10 scheduled fire times and exit
    ./scripts/oi_scheduler.py --dry-run

    # Keep the Mac awake while running so the loop doesn't sleep with the lid
    caffeinate -s ./scripts/oi_scheduler.py

Background
----------
    # Detach from the terminal, log to file
    nohup ./scripts/oi_scheduler.py > logs/oi-scheduler.out 2>&1 &

    # Stop it later
    pkill -f oi_scheduler.py
"""
from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
from datetime import datetime, time as dtime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BATCH_SCRIPT = REPO_ROOT / "scripts" / "oi_batch.py"

# Market slots (assumed local time on the Mac == ET; set your Mac TZ
# accordingly, or shift these two constants if you're not on ET).
MARKET_OPEN = dtime(9, 30)
MARKET_CLOSE = dtime(16, 0)
POST_CLOSE = dtime(16, 5)  # settlement snapshot with --refresh-cache


# ---------------------------------------------------------------------------
# Scheduling math
# ---------------------------------------------------------------------------

def _next_business_day(d: datetime) -> datetime:
    d = d.replace(hour=0, minute=0, second=0, microsecond=0)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def _slots_for_day(day: datetime, interval_min: int) -> list[datetime]:
    """All fire times on a given business day: aligned :00/:30 during
    market hours + the post-close settlement run.
    """
    slots: list[datetime] = []
    step = timedelta(minutes=interval_min)
    t = day.replace(
        hour=MARKET_OPEN.hour, minute=MARKET_OPEN.minute,
        second=0, microsecond=0,
    )
    close_dt = day.replace(
        hour=MARKET_CLOSE.hour, minute=MARKET_CLOSE.minute,
        second=0, microsecond=0,
    )
    # Round `t` UP to nearest interval boundary from 09:30 (so with
    # interval_min=30 the slots are 09:30, 10:00, 10:30, ...).
    while t <= close_dt:
        slots.append(t)
        t += step
    slots.append(day.replace(
        hour=POST_CLOSE.hour, minute=POST_CLOSE.minute,
        second=0, microsecond=0,
    ))
    return slots


def _next_slot(now: datetime, interval_min: int) -> datetime:
    """The first scheduled fire time strictly after `now`."""
    today = _next_business_day(now)
    for slot in _slots_for_day(today, interval_min):
        if slot > now:
            return slot
    # No slots left today → roll to next business day's 09:30.
    tomorrow = _next_business_day(today + timedelta(days=1))
    return _slots_for_day(tomorrow, interval_min)[0]


def _upcoming_slots(now: datetime, interval_min: int, n: int) -> list[datetime]:
    slots: list[datetime] = []
    cursor = now
    for _ in range(n):
        s = _next_slot(cursor, interval_min)
        slots.append(s)
        cursor = s
    return slots


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def _run_batch(
    tickers: str,
    days_ahead: int,
    workers: int,
    refresh_cache: bool,
    log_dir: Path,
) -> tuple[int, float]:
    log_dir.mkdir(parents=True, exist_ok=True)
    daily_log = log_dir / f"{datetime.now().strftime('%Y-%m-%d')}.log"

    cmd = [
        sys.executable, str(BATCH_SCRIPT),
        "--rolling-days", str(days_ahead),
        "--tickers", tickers,
        "--workers", str(workers),
    ]
    if refresh_cache:
        cmd.append("--refresh-cache")

    banner = f"\n{'=' * 70}\n[{datetime.now():%Y-%m-%d %H:%M:%S}] firing: {' '.join(cmd)}\n{'=' * 70}\n"
    print(banner.rstrip())
    with open(daily_log, "a") as f:
        f.write(banner)
        f.flush()
        started = time.time()
        rc = subprocess.call(cmd, cwd=REPO_ROOT, stdout=f, stderr=subprocess.STDOUT)
        elapsed = time.time() - started
        f.write(f"\n[{datetime.now():%H:%M:%S}] exit={rc} elapsed={elapsed:.1f}s\n")
    return rc, elapsed


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

_shutdown = False


def _handle_signal(signum, _frame):
    global _shutdown
    print(f"\n[{datetime.now():%H:%M:%S}] received signal {signum}, shutting down after current sleep…")
    _shutdown = True


def _sleep_until(target: datetime) -> None:
    """Sleep in short chunks so signals stay responsive."""
    while not _shutdown:
        now = datetime.now()
        remaining = (target - now).total_seconds()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 5.0))


def main() -> int:
    p = argparse.ArgumentParser(
        description="Fire oi_batch.py on a fixed schedule (stdlib-only cron alternative).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--tickers", default="SPY",
        help="Comma-separated tickers passed through to oi_batch.py (default: SPY).",
    )
    p.add_argument(
        "--days-ahead", type=int, default=5,
        help="Business days of expirations to snapshot per fire (default: 5).",
    )
    p.add_argument(
        "--interval-min", type=int, default=30,
        help="Cadence during market hours, in minutes (default: 30).",
    )
    p.add_argument(
        "--workers", type=int, default=4,
        help="Parallel workers for each batch run (default: 4).",
    )
    p.add_argument(
        "--no-refresh-cache", action="store_true",
        help="Skip --refresh-cache on intraday fires (post-close always refreshes).",
    )
    p.add_argument(
        "--log-dir", default=str(REPO_ROOT / "logs" / "oi-scheduler"),
        help="Where to write per-day batch logs.",
    )
    p.add_argument(
        "--fire-once-on-start", action="store_true",
        help="Fire immediately on startup, then wait for the next scheduled slot.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print the next 10 scheduled fire times and exit, no batches run.",
    )
    args = p.parse_args()

    if args.interval_min < 1 or 60 % args.interval_min != 0:
        p.error("--interval-min must divide 60 evenly (e.g. 5, 10, 15, 20, 30, 60).")

    log_dir = Path(args.log_dir)

    if args.dry_run:
        upcoming = _upcoming_slots(datetime.now(), args.interval_min, 10)
        print(f"Next 10 fires (interval={args.interval_min}m, days-ahead={args.days_ahead}):")
        for i, s in enumerate(upcoming, 1):
            settlement = " (settlement, --refresh-cache)" if s.time() == POST_CLOSE else ""
            print(f"  {i:2}. {s:%a %Y-%m-%d %H:%M}{settlement}")
        return 0

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    print(
        f"[{datetime.now():%Y-%m-%d %H:%M:%S}] scheduler starting  "
        f"tickers={args.tickers}  days-ahead={args.days_ahead}  "
        f"interval={args.interval_min}m  workers={args.workers}  "
        f"logs={log_dir}"
    )
    upcoming = _upcoming_slots(datetime.now(), args.interval_min, 3)
    print("  next 3 fires:")
    for s in upcoming:
        print(f"    · {s:%a %H:%M}")

    if args.fire_once_on_start:
        _run_batch(
            args.tickers, args.days_ahead, args.workers,
            refresh_cache=not args.no_refresh_cache,
            log_dir=log_dir,
        )

    while not _shutdown:
        target = _next_slot(datetime.now(), args.interval_min)
        wait_sec = (target - datetime.now()).total_seconds()
        wait_min = wait_sec / 60
        wait_str = (
            f"{wait_sec:.0f}s" if wait_sec < 60
            else f"{wait_min:.1f}m" if wait_min < 60
            else f"{wait_min / 60:.1f}h"
        )
        print(
            f"[{datetime.now():%H:%M:%S}] sleeping "
            f"{wait_str} → next fire at {target:%a %H:%M}"
        )
        _sleep_until(target)
        if _shutdown:
            break

        is_settlement = target.time() == POST_CLOSE
        # Settlement always refreshes; intraday obeys --no-refresh-cache flag.
        refresh_cache = True if is_settlement else (not args.no_refresh_cache)
        try:
            rc, elapsed = _run_batch(
                args.tickers, args.days_ahead, args.workers,
                refresh_cache=refresh_cache,
                log_dir=log_dir,
            )
            tag = "settlement" if is_settlement else "intraday"
            print(f"[{datetime.now():%H:%M:%S}] {tag} batch finished  rc={rc}  ({elapsed:.1f}s)")
        except Exception as e:  # noqa: BLE001
            print(f"[{datetime.now():%H:%M:%S}] batch raised: {e!r}")

    print(f"[{datetime.now():%H:%M:%S}] scheduler stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
