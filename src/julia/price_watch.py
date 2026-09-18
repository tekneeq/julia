"""Discord price watches — interval prints and % / price alerts.

    !lia watch SPY 1m      print last + daily change every minute
    !lia watch SPY 1%      print when today's % change hits +1%
    !lia watch SPY -1%     print when today's % change hits −1%
    !lia watch SPY 759     print when last hits 759
    !lia watch             list watches
    !lia watch close <id>
    !spy watch             same list (SPY shorthand also accepts a trigger)

Jobs persist in SQLite so the Discord bot can restart without losing them.
"""
from __future__ import annotations

import os
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

DEFAULT_DB_PATH = os.path.join(".options_cache", "price_watch.db")

KIND_INTERVAL = "interval"
KIND_PCT = "pct"
KIND_PRICE = "price"

STATUS_PENDING = "pending"
STATUS_FIRED = "fired"
STATUS_CANCELLED = "cancelled"

MIN_INTERVAL_SECONDS = 15

_SCHEMA = """
CREATE TABLE IF NOT EXISTS price_watches (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at       TEXT NOT NULL,
    created_by       TEXT,
    channel_id       TEXT,
    symbol           TEXT NOT NULL,
    kind             TEXT NOT NULL,
    trigger_text     TEXT NOT NULL,
    interval_seconds INTEGER,
    pct_level        REAL,
    price_level      REAL,
    price_op         TEXT,
    last_price       REAL,
    last_printed_at  TEXT,
    print_count      INTEGER NOT NULL DEFAULT 0,
    status           TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_price_watches_status
    ON price_watches(status, kind, symbol);
"""


@contextmanager
def _connect(db_path: str = DEFAULT_DB_PATH):
    parent = os.path.dirname(db_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA journal_mode = WAL")
    try:
        conn.executescript(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def _now_et() -> datetime:
    return datetime.now(ET)


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ET)
    return dt.astimezone(ET).isoformat(timespec="seconds")


def _parse_iso(raw: Optional[str]) -> Optional[datetime]:
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ET)
    return dt.astimezone(ET)


def _row_to_job(row: sqlite3.Row) -> dict[str, Any]:
    return {k: row[k] for k in row.keys()}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WatchSpec:
    kind: str
    text: str
    interval_seconds: Optional[int] = None
    pct_level: Optional[float] = None
    price_level: Optional[float] = None
    price_op: Optional[str] = None


@dataclass(frozen=True)
class WatchRequest:
    symbol: str
    spec: WatchSpec


def _fmt_interval(seconds: int) -> str:
    if seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    if seconds % 60 == 0:
        return f"{seconds // 60}m"
    return f"{seconds}s"


def parse_watch_spec(text: str) -> WatchSpec:
    """Parse ``1m`` / ``1%`` / ``-1%`` / ``759``."""
    raw = " ".join(text.split()).strip().lower()
    if not raw:
        raise ValueError(_watch_usage())

    interval = re.fullmatch(
        r"(\d+(?:\.\d+)?)\s*"
        r"(s|sec|secs|second|seconds|m|min|mins|minute|minutes|"
        r"h|hr|hrs|hour|hours)",
        raw,
    )
    if interval:
        amount = float(interval.group(1))
        if amount <= 0:
            raise ValueError("Interval must be > 0.")
        unit = interval.group(2)
        if unit.startswith("h"):
            seconds = int(round(amount * 3600))
        elif unit.startswith("m"):
            seconds = int(round(amount * 60))
        else:
            seconds = int(round(amount))
        if seconds < MIN_INTERVAL_SECONDS:
            raise ValueError(
                f"Interval must be at least {MIN_INTERVAL_SECONDS}s "
                f"(got {_fmt_interval(seconds)})."
            )
        label = _fmt_interval(seconds)
        return WatchSpec(KIND_INTERVAL, f"every {label}", seconds)

    pct = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)\s*%", raw)
    if pct:
        level = float(pct.group(1))
        if level == 0:
            raise ValueError("Percent trigger cannot be 0.")
        sign = "+" if level > 0 else ""
        return WatchSpec(
            KIND_PCT,
            f"when day {sign}{level:g}%",
            pct_level=level,
        )

    price = re.fullmatch(r"(\d+(?:\.\d+)?)", raw)
    if price:
        level = float(price.group(1))
        if level <= 0:
            raise ValueError("Price level must be > 0.")
        return WatchSpec(
            KIND_PRICE,
            f"when last hits {level:g}",
            price_level=level,
        )

    raise ValueError(f"Could not parse watch `{text}`.\n{_watch_usage()}")


def parse_watch_args(args: list[str]) -> WatchRequest:
    """Parse ``TICKER 1m|1%|-1%|759``."""
    if len(args) < 2:
        raise ValueError(
            "Usage: `!lia watch TICKER 1m|1%|-1%|759`\n"
            "List: `!lia watch` / `!spy watch` · "
            "Stop: `!lia watch close <id>` / `!spy watch close <id>`"
        )
    symbol = args[0].strip().upper().lstrip("$")
    if not symbol.isalnum():
        raise ValueError(f"Bad ticker `{args[0]}`.")
    spec = parse_watch_spec(" ".join(args[1:]))
    return WatchRequest(symbol=symbol, spec=spec)


def _watch_usage() -> str:
    return (
        "Watch trigger must be one of:\n"
        "  `1m` / `5m` / `1h`     print every interval\n"
        "  `1%` / `-1%`           print when today's % hits that level\n"
        "  `759`                  print when last hits that price"
    )


def resolve_price_op(spec: WatchSpec, last: float) -> WatchSpec:
    """Turn ``hits N`` into ``>=`` or ``<=`` from the current print."""
    if spec.kind != KIND_PRICE or spec.price_level is None:
        return spec
    level = spec.price_level
    op = "<=" if last >= level else ">="
    return WatchSpec(
        KIND_PRICE,
        f"when last hits {level:g} → {op} {level:g} (spot {last:g})",
        price_level=level,
        price_op=op,
    )


def price_condition_met(op: Optional[str], level: Optional[float], last: float) -> bool:
    if not op or level is None:
        return False
    if op == ">=":
        return last >= level
    if op == "<=":
        return last <= level
    if op == ">":
        return last > level
    if op == "<":
        return last < level
    return False


def pct_condition_met(level: Optional[float], daily_pct: Optional[float]) -> bool:
    if level is None or daily_pct is None:
        return False
    if level >= 0:
        return daily_pct >= level
    return daily_pct <= level


def interval_due(job: dict[str, Any], now: datetime) -> bool:
    seconds = int(job.get("interval_seconds") or 0)
    if seconds <= 0:
        return False
    last = _parse_iso(job.get("last_printed_at"))
    if last is None:
        return True
    return (now - last) >= timedelta(seconds=seconds)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def insert_watch(
    *,
    symbol: str,
    spec: WatchSpec,
    created_by: Optional[str] = None,
    channel_id: Optional[str] = None,
    last_price: Optional[float] = None,
    last_printed_at: Optional[datetime] = None,
    print_count: int = 0,
    status: str = STATUS_PENDING,
    db_path: str = DEFAULT_DB_PATH,
) -> dict[str, Any]:
    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO price_watches (
                created_at, created_by, channel_id, symbol, kind,
                trigger_text, interval_seconds, pct_level, price_level,
                price_op, last_price, last_printed_at, print_count, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _iso(_now_et()),
                created_by,
                channel_id,
                symbol.upper(),
                spec.kind,
                spec.text,
                spec.interval_seconds,
                spec.pct_level,
                spec.price_level,
                spec.price_op,
                last_price,
                _iso(last_printed_at) if last_printed_at else None,
                int(print_count),
                status,
            ),
        )
        watch_id = int(cur.lastrowid)
    job = get_watch(watch_id, db_path=db_path)
    assert job is not None
    return job


def get_watch(watch_id: int, *, db_path: str = DEFAULT_DB_PATH) -> Optional[dict[str, Any]]:
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM price_watches WHERE id = ?",
            (watch_id,),
        ).fetchone()
    return _row_to_job(row) if row else None


def list_watches(
    *,
    statuses: Optional[tuple[str, ...]] = None,
    limit: int = 40,
    db_path: str = DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    sql = "SELECT * FROM price_watches"
    params: list[Any] = []
    if statuses:
        placeholders = ",".join("?" * len(statuses))
        sql += f" WHERE status IN ({placeholders})"
        params.extend(statuses)
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(int(limit))
    with _connect(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [_row_to_job(r) for r in rows]


def list_pending(*, db_path: str = DEFAULT_DB_PATH) -> list[dict[str, Any]]:
    return list_watches(statuses=(STATUS_PENDING,), limit=100, db_path=db_path)


def cancel_watch(watch_id: int, *, db_path: str = DEFAULT_DB_PATH) -> Optional[dict[str, Any]]:
    job = get_watch(watch_id, db_path=db_path)
    if job is None:
        return None
    if job["status"] != STATUS_PENDING:
        return job
    with _connect(db_path) as conn:
        conn.execute(
            """
            UPDATE price_watches
            SET status = ?
            WHERE id = ? AND status = ?
            """,
            (STATUS_CANCELLED, watch_id, STATUS_PENDING),
        )
    return get_watch(watch_id, db_path=db_path)


def mark_printed(
    watch_id: int,
    *,
    price: float,
    now: Optional[datetime] = None,
    expected_last_printed_at: Optional[str] = None,
    status: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH,
) -> Optional[dict[str, Any]]:
    """Record a print. Returns None if another tick already claimed it."""
    now = now or _now_et()
    with _connect(db_path) as conn:
        if expected_last_printed_at is None:
            cur = conn.execute(
                """
                UPDATE price_watches
                SET last_price = ?, last_printed_at = ?,
                    print_count = print_count + 1,
                    status = COALESCE(?, status)
                WHERE id = ? AND last_printed_at IS NULL AND status = ?
                """,
                (price, _iso(now), status, watch_id, STATUS_PENDING),
            )
        else:
            cur = conn.execute(
                """
                UPDATE price_watches
                SET last_price = ?, last_printed_at = ?,
                    print_count = print_count + 1,
                    status = COALESCE(?, status)
                WHERE id = ? AND last_printed_at = ? AND status = ?
                """,
                (
                    price,
                    _iso(now),
                    status,
                    watch_id,
                    expected_last_printed_at,
                    STATUS_PENDING,
                ),
            )
        if cur.rowcount == 0:
            return None
    return get_watch(watch_id, db_path=db_path)


def claim_due_watches(
    quotes: dict[str, dict[str, Any]],
    now: Optional[datetime] = None,
    *,
    db_path: str = DEFAULT_DB_PATH,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Claim pending watches that should print. Interval stays pending."""
    now = now or _now_et()
    if now.tzinfo is None:
        now = now.replace(tzinfo=ET)
    else:
        now = now.astimezone(ET)
    claimed: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for job in list_pending(db_path=db_path):
        quote = quotes.get(str(job.get("symbol") or "").upper())
        if not quote or quote.get("price") is None:
            continue
        last = float(quote["price"])
        kind = job.get("kind")
        new_status: Optional[str] = None
        due = False
        if kind == KIND_INTERVAL:
            due = interval_due(job, now)
        elif kind == KIND_PCT:
            due = pct_condition_met(job.get("pct_level"), quote.get("daily_pct"))
            if due:
                new_status = STATUS_FIRED
        elif kind == KIND_PRICE:
            due = price_condition_met(job.get("price_op"), job.get("price_level"), last)
            if due:
                new_status = STATUS_FIRED
        if not due:
            continue
        quote_view = dict(quote)
        quote_view["prev_print_price"] = job.get("last_price")
        updated = mark_printed(
            int(job["id"]),
            price=last,
            now=now,
            expected_last_printed_at=job.get("last_printed_at"),
            status=new_status,
            db_path=db_path,
        )
        if updated:
            claimed.append((updated, quote_view))
    return claimed


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _signed_money(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "—"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:,.{digits}f}"


def _signed_pct(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "—"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.{digits}f}%"


signed_money = _signed_money
signed_pct = _signed_pct


def format_quote_message(job: dict[str, Any], quote: dict[str, Any]) -> str:
    symbol = quote.get("symbol") or job.get("symbol") or "?"
    price = quote.get("price")
    daily_chg = quote.get("daily_chg")
    daily_pct = quote.get("daily_pct")
    # Prefer the pre-update last so "since last" is not zero after claim.
    prev = quote.get("prev_print_price")
    vs_last = None
    vs_last_pct = None
    if prev not in (None, "") and price is not None and float(prev) != 0:
        vs_last = float(price) - float(prev)
        vs_last_pct = vs_last / float(prev) * 100.0

    price_s = f"${float(price):,.2f}" if price is not None else "—"
    kind = job.get("kind")
    jid = job.get("id")
    trig = job.get("trigger_text") or "?"
    status = job.get("status") or "?"

    if kind == KIND_PCT:
        headline = f"**{symbol}** hit `{_signed_pct(job.get('pct_level'))}` on the day"
    elif kind == KIND_PRICE:
        level = job.get("price_level")
        headline = f"**{symbol}** hit `${float(level):g}`" if level is not None else f"**{symbol}**"
    else:
        headline = f"**{symbol}**"

    lines = [
        f"{headline} · `{price_s}`",
        f"Day `{_signed_money(daily_chg)}` (`{_signed_pct(daily_pct)}`)"
        + (
            f" · since last `{_signed_money(vs_last)}` (`{_signed_pct(vs_last_pct)}`)"
            if vs_last is not None
            else ""
        ),
        f"`#{jid}` `{status}` {trig}",
    ]
    return "\n".join(lines)


def format_watch_line(job: dict[str, Any]) -> str:
    jid = job.get("id")
    status = job.get("status") or "?"
    symbol = job.get("symbol") or "?"
    trig = job.get("trigger_text") or "?"
    n = job.get("print_count") or 0
    last = job.get("last_price")
    last_s = f"${float(last):,.2f}" if last not in (None, "") else "—"
    return f"`#{jid}` `{status}` **{symbol}** {trig} · last `{last_s}` · prints `{n}`"


def format_watch_status(jobs: list[dict[str, Any]]) -> str:
    if not jobs:
        return (
            "No price watches. Start one with `!lia watch SPY 1m`, "
            "`!lia watch SPY 1%`, or `!lia watch SPY 759`.\n"
            "List: `!spy watch` · stop: `!spy watch close <id>`"
        )
    lines = [f"**Price watches** · {len(jobs)}"]
    for job in jobs:
        lines.append(format_watch_line(job))
    lines.append("_Stop with_ `!lia watch close <id>` _or_ `!spy watch close <id>`")
    return "\n".join(lines)
