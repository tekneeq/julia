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

    # Don't wait for the next :00/:30 slot — fire a batch right now, then
    # settle into the normal schedule
    ./scripts/oi_scheduler.py --now

    # Fire one batch immediately and exit (no scheduling loop) — handy for
    # backfilling an expiration that's missing data
    ./scripts/oi_scheduler.py --run-once

    # Different ticker set, longer window
    ./scripts/oi_scheduler.py --tickers SPY,QQQ --days-ahead 7

    # Custom cadence — fire every 15 min instead of 30
    ./scripts/oi_scheduler.py --interval-min 15

    # Preview: show the next 10 scheduled fire times and exit
    ./scripts/oi_scheduler.py --dry-run

    # What has actually run? Coverage of the current window + fire history
    ./scripts/oi_scheduler.py --status

    # Keep the Mac awake while running so the loop doesn't sleep with the lid
    caffeinate -s ./scripts/oi_scheduler.py

Background
----------
    # Detach from the terminal, log to file
    nohup ./scripts/oi_scheduler.py > logs/oi-scheduler.out 2>&1 &

    # Replace whatever is already running (SIGTERM the old pid, then start)
    ./scripts/oi_scheduler.py --replace

    # Stop it later
    ./scripts/oi_scheduler.py --stop
"""
from __future__ import annotations

import argparse
import os
import signal
import sqlite3
import subprocess
import sys
import time
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BATCH_SCRIPT = REPO_ROOT / "scripts" / "oi_batch.py"
DB_PATH = REPO_ROOT / ".options_cache" / "predictions.db"
BATCH_LOG_DIR = REPO_ROOT / "logs" / "oi-batch"
DEFAULT_PID_FILE = REPO_ROOT / "logs" / "oi-scheduler.pid"

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


def _next_business_date(d: date) -> date:
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def _add_business_days(d: date, n: int) -> date:
    while n > 0:
        d += timedelta(days=1)
        if d.weekday() < 5:
            n -= 1
    return d


def _rolling_window(day: date, days_ahead: int) -> list[date]:
    """Expirations that ``oi_batch.py --rolling-days N`` targets on ``day``.

    Mirrors oi_batch's own math so the status report can tell you what
    *should* have been captured on any past day, not just today.
    """
    start = _next_business_date(day)
    end = _add_business_days(start, days_ahead - 1)
    out: list[date] = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


# ---------------------------------------------------------------------------
# PID file — lets "restart" actually restart, and --status know if we're up
# ---------------------------------------------------------------------------
# Caveat: pids get recycled. A stale pid file whose number has been reused
# by an unrelated process would read as "running". The window is small and
# the blast radius is a refused start, so we don't try to be cleverer.

_PID_MARKER = "oi_scheduler.py"


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
        return True  # exists, just not ours to signal
    return True


def _pid_cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\x00", b" ").decode(errors="replace")


def _is_our_process(pid: int) -> bool:
    """True only if ``pid`` is actually this scheduler (not a recycled PID).

    ``logs/oi-scheduler.pid`` lives on a host volume, so after
    ``docker rm`` + recreate the old number can belong to Streamlit / uv
    inside the new container. Never SIGTERM those.
    """
    return _PID_MARKER in _pid_cmdline(pid)


def _scrub_stale_pid(pid_file: Path) -> None:
    """Drop a pid file that doesn't point at a live scheduler."""
    pid = _read_pid(pid_file)
    if pid is None:
        return
    if pid == os.getpid():
        return
    if not _pid_alive(pid) or not _is_our_process(pid):
        pid_file.unlink(missing_ok=True)


def _running_pid(pid_file: Path) -> int | None:
    """The pid of a live scheduler other than us, if there is one."""
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
    # Only if it's still ours — a replacement scheduler may have claimed
    # the file while we were winding down.
    if _read_pid(pid_file) == os.getpid():
        pid_file.unlink(missing_ok=True)


def _stop_running(pid_file: Path, timeout: float = 30.0) -> bool:
    """SIGTERM the running scheduler and wait for it to exit.

    The loop only checks its shutdown flag between 5s sleep chunks, and a
    fire already in flight runs to completion, so we escalate to SIGKILL
    after ``timeout``. Returns False if nothing was running.
    """
    _scrub_stale_pid(pid_file)
    pid = _running_pid(pid_file)
    if pid is None:
        return False
    print(f"[{datetime.now():%H:%M:%S}] stopping scheduler pid {pid}…")
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
        time.sleep(0.5)
    print(f"  pid {pid} still alive after {timeout:.0f}s (mid-batch?) — SIGKILL.")
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass
    return True


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
    # oi_batch.py refreshes the Robinhood cache by default; opt out via
    # its --no-refresh-cache flag when we want to reuse the cached chain.
    if not refresh_cache:
        cmd.append("--no-refresh-cache")

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
# Status report — "what actually ran, and what's missing?"
# ---------------------------------------------------------------------------

def _parse_utc(iso_ts: str) -> datetime:
    return datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))


def _fmt_age(seconds: float) -> str:
    secs = int(seconds)
    if secs < 60:
        return f"{secs}s ago"
    if secs < 3600:
        return f"{secs // 60}m ago"
    if secs < 86400:
        return f"{secs // 3600}h{(secs % 3600) // 60:02d}m ago"
    return f"{secs // 86400}d ago"


def _query(sql: str, params: tuple = ()) -> list[tuple]:
    """Read-only query against the snapshot DB. Empty list on any problem
    (no DB yet, table not created yet, locked, ...) — status should never
    be the thing that crashes."""
    if not DB_PATH.exists():
        return []
    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    except sqlite3.Error:
        return []
    try:
        return conn.execute(sql, params).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()


def _captures(
    tickers: list[str], *, expirations: list[date] | None = None,
    since: date | None = None,
) -> list[tuple[str, str, datetime]]:
    """(ticker, expiration_iso, captured_at_local) rows, oldest first."""
    where, params = ["1=1"], []
    if expirations:
        where.append(
            "expiration_date IN (%s)" % ",".join("?" * len(expirations))
        )
        params.extend(e.isoformat() for e in expirations)
    if since is not None:
        midnight_utc = datetime.combine(
            since, dtime(0, 0)
        ).astimezone().astimezone(timezone.utc)
        where.append("captured_at >= ?")
        params.append(midnight_utc.isoformat(timespec="seconds"))
    rows = _query(
        "SELECT ticker, expiration_date, captured_at FROM gex_snapshots "
        f"WHERE {' AND '.join(where)} ORDER BY captured_at",
        tuple(params),
    )
    wanted = set(tickers)
    return [
        (t, exp, _parse_utc(ts).astimezone())
        for t, exp, ts in rows
        if not wanted or t in wanted
    ]


def _cluster_fires(times: list[datetime], gap_min: float = 10.0) -> list[datetime]:
    """Collapse per-job capture times into one timestamp per batch fire.

    Each fire writes one snapshot per (ticker × expiration) a few seconds
    apart, so raw row counts overstate how often the scheduler fired. Any
    gap longer than ``gap_min`` starts a new fire.
    """
    fires: list[datetime] = []
    prev: datetime | None = None
    for t in sorted(times):
        if prev is None or (t - prev).total_seconds() > gap_min * 60:
            fires.append(t)
        prev = t
    return fires


def _batch_job_hint(ticker: str, exp: date) -> str | None:
    """Last meaningful line of the per-job batch log, if one exists.

    When an expiration has no snapshots, this usually says why — e.g.
    Robinhood hasn't listed the chain yet, or auth expired.
    """
    log = BATCH_LOG_DIR / f"{ticker}-{exp.isoformat()}.log"
    try:
        text = log.read_text(errors="replace")
        age = _fmt_age(time.time() - log.stat().st_mtime)
    except OSError:
        return None
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return f"{log.relative_to(REPO_ROOT)} ({age}): (empty)"
    # An error line beats the tool chatter (uv deprecation warnings) that
    # tends to be the literal last line of a failed job's output.
    pick = next(
        (ln for ln in reversed(lines) if ln.startswith(("❌", "Error:"))),
        None,
    ) or next(
        (ln for ln in reversed(lines) if not ln.lower().startswith("warning:")),
        lines[-1],
    )
    return f"{log.relative_to(REPO_ROOT)} ({age}): {pick[:120]}"


def _status_report(
    tickers: list[str], days_ahead: int, pid_file: Path, history_days: int,
) -> int:
    now = datetime.now()
    print(f"oi-scheduler status  ·  {now:%a %Y-%m-%d %H:%M:%S %Z}")
    print(f"  db: {DB_PATH}{'' if DB_PATH.exists() else '   ← does not exist yet'}")

    # --- is it up? ---------------------------------------------------
    _scrub_stale_pid(pid_file)
    pid = _running_pid(pid_file)
    if pid:
        print(f"\nProcess: ● RUNNING (pid {pid}, pid-file {pid_file.name})")
    else:
        print("\nProcess: ○ NOT RUNNING")
        print("         start it with  ./restart-oi-scheduler.sh --now")

    # --- current window coverage -------------------------------------
    window = _rolling_window(now.date(), days_ahead)
    print(
        f"\nCurrent window ({days_ahead} business days): "
        f"{window[0]} .. {window[-1]}"
    )
    rows = _captures(tickers, expirations=window)
    by_key: dict[tuple[str, str], list[datetime]] = {}
    for t, exp, ts in rows:
        by_key.setdefault((t, exp), []).append(ts)

    print(f"  {'ticker':<7} {'expiration':<17} {'snaps':>6} {'today':>6}  last capture")
    gaps: list[tuple[str, date]] = []
    for ticker in tickers:
        for exp in window:
            times = by_key.get((ticker, exp.isoformat()), [])
            today_n = sum(1 for t in times if t.date() == now.date())
            label = f"{exp} ({exp:%a})"
            if times:
                last = max(times)
                last_str = f"{last:%m-%d %H:%M}  ({_fmt_age((now - last.replace(tzinfo=None)).total_seconds())})"
                flag = "" if today_n else "   ← nothing today"
            else:
                last_str, flag = "never", "   ← NO DATA"
                gaps.append((ticker, exp))
            print(
                f"  {ticker:<7} {label:<17} {len(times):>6} {today_n:>6}  "
                f"{last_str}{flag}"
            )
    for ticker, exp in gaps:
        hint = _batch_job_hint(ticker, exp)
        if hint:
            print(f"      ↳ {ticker} {exp}: {hint}")

    # --- fire history -------------------------------------------------
    since = now.date() - timedelta(days=history_days)
    hist = _captures(tickers, since=since)
    print(f"\nDates it ran (last {history_days} days, by capture day):")
    if not hist:
        print("  (no snapshots recorded — the scheduler has never completed a fire)")
    else:
        by_day: dict[date, list[tuple[str, datetime]]] = {}
        for _t, exp, ts in hist:
            by_day.setdefault(ts.date(), []).append((exp, ts))
        print(
            f"  {'day':<15} {'fires':>5}  {'first':>5}  {'last':>5}  "
            f"expirations covered"
        )
        for day in sorted(by_day):
            entries = by_day[day]
            times = [ts for _e, ts in entries]
            fires = _cluster_fires(times)
            covered = {date.fromisoformat(e) for e, _ts in entries}
            expected = set(_rolling_window(day, days_ahead))
            missing = sorted(expected - covered)
            shown = ", ".join(f"{d:%m-%d}" for d in sorted(covered))
            miss = (
                "   ⚠ missing " + ", ".join(f"{d:%m-%d}" for d in missing)
                if missing else ""
            )
            day_str = f"{day:%a %Y-%m-%d}"
            print(
                f"  {day_str:<15} {len(fires):>5}  "
                f"{min(times):%H:%M}  {max(times):%H:%M}  {shown}{miss}"
            )
        expected_today = set(_rolling_window(now.date(), days_ahead))
        covered_today = {
            date.fromisoformat(e) for _t, e, ts in hist if ts.date() == now.date()
        }
        if expected_today - covered_today:
            print(
                "\n  ⚠ Today is missing: "
                + ", ".join(str(d) for d in sorted(expected_today - covered_today))
                + "\n    Backfill now with:  ./restart-oi-scheduler.sh --run-once"
            )
    print(
        f"\nLogs: {BATCH_LOG_DIR.relative_to(REPO_ROOT)}/ (per job)  ·  "
        f"logs/oi-scheduler/YYYY-MM-DD.log (per fire)"
    )
    return 0


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
        "--now", "--fire-once-on-start", dest="fire_once_on_start",
        action="store_true",
        help="Fire immediately on startup, then wait for the next scheduled slot.",
    )
    p.add_argument(
        "--run-once", action="store_true",
        help="Fire one batch right now and exit — no scheduling loop. Use "
             "this to backfill an expiration that's missing data.",
    )
    p.add_argument(
        "--pid-file", default=str(DEFAULT_PID_FILE),
        help=f"Where to track the running scheduler (default: {DEFAULT_PID_FILE}).",
    )
    p.add_argument(
        "--replace", action="store_true",
        help="Stop an already-running scheduler before starting (restart).",
    )
    p.add_argument(
        "--stop", action="store_true",
        help="Stop the running scheduler and exit.",
    )
    p.add_argument(
        "--status", action="store_true",
        help="Show whether the scheduler is up, which expirations in the "
             "current window have data, and the dates/times it actually "
             "fired. Exits without touching the scheduler.",
    )
    p.add_argument(
        "--history-days", type=int, default=10,
        help="How many days of fire history --status prints (default: 10).",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print the next 10 scheduled fire times and exit, no batches run.",
    )
    args = p.parse_args()

    # Detached runs redirect stdout to a file, which makes it block-
    # buffered — the log would sit empty for hours. Line-buffer it so
    # `tail -f logs/oi-scheduler.out` shows progress as it happens.
    sys.stdout.reconfigure(line_buffering=True)

    if args.interval_min < 1 or 60 % args.interval_min != 0:
        p.error("--interval-min must divide 60 evenly (e.g. 5, 10, 15, 20, 30, 60).")

    log_dir = Path(args.log_dir)
    pid_file = Path(args.pid_file)
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    if args.status:
        return _status_report(
            tickers, args.days_ahead, pid_file, args.history_days,
        )

    if args.stop:
        if _stop_running(pid_file):
            return 0
        print(f"No scheduler running (pid-file {pid_file}).")
        return 0

    if args.dry_run:
        upcoming = _upcoming_slots(datetime.now(), args.interval_min, 10)
        print(f"Next 10 fires (interval={args.interval_min}m, days-ahead={args.days_ahead}):")
        for i, s in enumerate(upcoming, 1):
            settlement = " (settlement, --refresh-cache)" if s.time() == POST_CLOSE else ""
            print(f"  {i:2}. {s:%a %Y-%m-%d %H:%M}{settlement}")
        return 0

    if args.run_once:
        print(
            f"[{datetime.now():%Y-%m-%d %H:%M:%S}] one-off run  "
            f"tickers={args.tickers}  days-ahead={args.days_ahead}"
        )
        window = _rolling_window(date.today(), args.days_ahead)
        print(f"  expirations: {', '.join(str(d) for d in window)}")
        rc, elapsed = _run_batch(
            args.tickers, args.days_ahead, args.workers,
            refresh_cache=not args.no_refresh_cache,
            log_dir=log_dir,
        )
        print(
            f"[{datetime.now():%H:%M:%S}] batch "
            f"{'OK' if rc == 0 else f'FAIL rc={rc}'} ({elapsed:.1f}s)"
        )
        return rc

    _scrub_stale_pid(pid_file)
    if args.replace:
        _stop_running(pid_file)
    elif (existing := _running_pid(pid_file)) is not None:
        print(
            f"A scheduler is already running (pid {existing}). "
            f"Use --replace to restart it, or --stop to shut it down.",
            file=sys.stderr,
        )
        return 1

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    # Ignore SIGHUP so a detached ``docker exec`` teardown can't kill us.
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    _write_pid(pid_file)

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

    try:
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
            # Settlement always refreshes; intraday obeys --no-refresh-cache.
            refresh_cache = True if is_settlement else (not args.no_refresh_cache)
            try:
                rc, elapsed = _run_batch(
                    args.tickers, args.days_ahead, args.workers,
                    refresh_cache=refresh_cache,
                    log_dir=log_dir,
                )
                tag = "settlement" if is_settlement else "intraday"
                status = "OK" if rc == 0 else f"FAIL rc={rc}"
                # A real batch takes 10s+ per ticker/date. Anything under a
                # second means every job died before doing work — usually a
                # bad CLI flag or a hard import error. Flag it loudly so it
                # can't hide across many quiet cron cycles.
                if elapsed < 1.0 and rc != 0:
                    print(
                        f"[{datetime.now():%H:%M:%S}] ⚠️  {tag} batch died in "
                        f"{elapsed:.1f}s with rc={rc} — likely a CLI/argparse "
                        f"error. See {log_dir}/{datetime.now():%Y-%m-%d}.log"
                    )
                else:
                    print(
                        f"[{datetime.now():%H:%M:%S}] {tag} batch {status}  "
                        f"({elapsed:.1f}s)"
                    )
            except Exception as e:  # noqa: BLE001
                print(f"[{datetime.now():%H:%M:%S}] batch raised: {e!r}")
    finally:
        _clear_pid(pid_file)

    print(f"[{datetime.now():%H:%M:%S}] scheduler stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
