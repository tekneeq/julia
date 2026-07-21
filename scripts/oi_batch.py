#!/usr/bin/env -S uv run python
"""Batch-run `lia oi-dashboard` across a range of expiration dates.

Skips weekends by default (options don't trade Sat/Sun). Fans jobs out to
a small worker pool so a week of expirations finishes in ~1× the slowest
individual run instead of 5×.

Examples
--------
    # Every business day, Mon 07/27 → Fri 07/31, SPY only, refresh cache
    ./scripts/oi_batch.py --from 2026-07-27 --to 2026-07-31

    # Multiple tickers, same date range, more parallelism
    ./scripts/oi_batch.py --from 2026-07-27 --to 2026-07-31 \\
        --tickers SPY,QQQ,IWM --workers 8

    # Preview the commands without running them
    ./scripts/oi_batch.py --from 2026-07-27 --to 2026-07-31 --dry-run

    # Pass through extra flags to `lia oi-dashboard`
    ./scripts/oi_batch.py --from 2026-07-27 --to 2026-07-31 \\
        --extra "--range 10 --rate 0.03"

Notes
-----
* Federal market holidays (July 4th, Thanksgiving, etc.) are NOT skipped
  by this script — options don't trade those days, so if you happen to
  aim at a holiday date `lia` will report "No options data" and the job
  will exit non-zero. That's benign; the batch keeps going.
* Each job auto-snapshots to the DB (via `oi-dashboard`), so a nightly
  cron of this script builds up your `oi-dashboard-diff` history.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import os
import shlex
import subprocess
import sys
import time
from datetime import date, timedelta


def _parse_date(s: str) -> date:
    try:
        return date.fromisoformat(s)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"Invalid date '{s}' — expected YYYY-MM-DD"
        ) from e


def _daterange(start: date, end: date, *, include_weekends: bool):
    d = start
    while d <= end:
        if include_weekends or d.weekday() < 5:  # Mon=0..Fri=4
            yield d
        d += timedelta(days=1)


def _build_cmd(
    ticker: str, exp: date, refresh_cache: bool, extra: list[str]
) -> list[str]:
    cmd = [
        "uv", "run", "lia", "oi-dashboard",
        "--ticker", ticker,
        "--expiration", exp.isoformat(),
        "--no-open",
    ]
    if refresh_cache:
        cmd.append("--refresh-cache")
    cmd.extend(extra)
    return cmd


def _run_one(
    ticker: str,
    exp: date,
    refresh_cache: bool,
    extra: list[str],
    log_dir: str | None,
) -> tuple[str, date, int, float, str]:
    cmd = _build_cmd(ticker, exp, refresh_cache, extra)
    env = os.environ.copy()
    env.setdefault("UV_NATIVE_TLS", "true")

    started = time.time()
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    elapsed = time.time() - started

    combined = (result.stdout or "") + (result.stderr or "")
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        fn = os.path.join(log_dir, f"{ticker}-{exp.isoformat()}.log")
        with open(fn, "w") as f:
            f.write(f"$ {' '.join(shlex.quote(x) for x in cmd)}\n\n")
            f.write(combined)

    return ticker, exp, result.returncode, elapsed, combined


def _last_meaningful_line(text: str) -> str:
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def main() -> int:
    p = argparse.ArgumentParser(
        description="Batch-run `lia oi-dashboard` across a date range.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--from", dest="start", type=_parse_date, required=True,
        help="Start expiration date, YYYY-MM-DD (inclusive).",
    )
    p.add_argument(
        "--to", dest="end", type=_parse_date, required=True,
        help="End expiration date, YYYY-MM-DD (inclusive).",
    )
    p.add_argument(
        "--tickers", default="SPY",
        help="Comma-separated tickers (default: SPY).",
    )
    p.add_argument(
        "--workers", type=int, default=4,
        help="Max parallel jobs (default: 4). Each job hits the Robinhood "
             "API — don't set this too high.",
    )
    p.add_argument(
        "--no-refresh-cache", action="store_true",
        help="Skip --refresh-cache (use cached options chain if present).",
    )
    p.add_argument(
        "--include-weekends", action="store_true",
        help="Don't skip Sat/Sun (rarely useful — options don't trade "
             "weekends — but here if you need it).",
    )
    p.add_argument(
        "--log-dir", default="logs/oi-batch",
        help="Per-job log directory (default: logs/oi-batch). "
             "Pass '' to disable file logging.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print the commands that would run and exit.",
    )
    p.add_argument(
        "--extra", default="",
        help="Extra flags passed through to `lia oi-dashboard`, "
             "e.g. --extra '--range 10 --rate 0.03'",
    )
    args = p.parse_args()

    if args.start > args.end:
        p.error("--from must be <= --to")

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    dates = list(
        _daterange(args.start, args.end, include_weekends=args.include_weekends)
    )
    extra = shlex.split(args.extra) if args.extra else []
    refresh_cache = not args.no_refresh_cache
    log_dir = args.log_dir or None
    jobs = [(t, d) for t in tickers for d in dates]

    if not jobs:
        print(
            f"No jobs to run in {args.start}..{args.end} "
            f"(all dates fell on weekends?).",
            file=sys.stderr,
        )
        return 1

    print(
        f"Planning {len(jobs)} job(s): {len(tickers)} ticker(s) × "
        f"{len(dates)} date(s)"
    )
    print(f"  Tickers: {', '.join(tickers)}")
    print(f"  Dates:   {', '.join(d.isoformat() for d in dates)}")
    print(f"  Workers: {args.workers}")
    print(f"  Log dir: {log_dir or '(disabled)'}")

    if args.dry_run:
        print("\nDry run — commands that would execute:")
        for t, d in jobs:
            cmd = _build_cmd(t, d, refresh_cache, extra)
            print("  UV_NATIVE_TLS=true " + " ".join(shlex.quote(x) for x in cmd))
        return 0

    print()
    started_all = time.time()
    results: list[tuple[str, date, int]] = []

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.workers
    ) as pool:
        futures = {
            pool.submit(
                _run_one, t, d, refresh_cache, extra, log_dir
            ): (t, d)
            for t, d in jobs
        }
        for fut in concurrent.futures.as_completed(futures):
            ticker, exp, rc, elapsed, out = fut.result()
            status = "OK  " if rc == 0 else f"FAIL({rc})"
            summary = _last_meaningful_line(out)
            print(
                f"[{status}] {ticker} {exp}  "
                f"({elapsed:5.1f}s)  {summary}"
            )
            results.append((ticker, exp, rc))

    elapsed_total = time.time() - started_all
    failed = [r for r in results if r[2] != 0]
    ok = len(results) - len(failed)
    print()
    print(
        f"Done in {elapsed_total:.1f}s. "
        f"{ok}/{len(results)} succeeded"
        + (f", {len(failed)} failed" if failed else "")
        + "."
    )
    if failed:
        print("Failures:")
        for t, d, rc in failed:
            print(f"  {t} {d} (exit {rc})")
        if log_dir:
            print(f"Logs: {log_dir}/")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
