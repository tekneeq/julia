"""Pending SPY credit-spread closes for the Discord ``!lia close`` commands.

A job closes both legs of a two-contract spread: buy-to-close the short
(sold) strike and sell-to-close the long (bought) strike. Triggers are a
clock time (``at 2pm``), a relative delay (``in 1h``), a SPY print
(``if spy >= 650`` / ``when spy hits 650``), or ``now``.

Jobs persist in SQLite so the Discord bot can restart without losing
watches. SPY-only for now.
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

DEFAULT_DB_PATH = os.path.join(".options_cache", "spread_close.db")
SPREAD_SYMBOL = "SPY"

STATUS_PENDING = "pending"
STATUS_FIRING = "firing"
STATUS_FIRED = "fired"
STATUS_PARTIAL = "partial"
STATUS_CANCELLED = "cancelled"
STATUS_ERROR = "error"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS spread_close_jobs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    TEXT NOT NULL,
    created_by    TEXT,
    channel_id    TEXT,
    symbol        TEXT NOT NULL,
    expiration    TEXT,
    option_type   TEXT,
    long_id       TEXT NOT NULL,
    long_strike   REAL,
    short_id      TEXT NOT NULL,
    short_strike  REAL,
    qty           INTEGER NOT NULL,
    trigger_kind  TEXT NOT NULL,
    trigger_text  TEXT NOT NULL,
    trigger_at    TEXT,
    price_op      TEXT,
    price_level   REAL,
    status        TEXT NOT NULL,
    last_error    TEXT,
    fired_at      TEXT,
    result_text   TEXT
);

CREATE INDEX IF NOT EXISTS idx_spread_close_status
    ON spread_close_jobs(status, trigger_kind);
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
# Trigger parsing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Trigger:
    kind: str  # now | at | in | price
    text: str
    trigger_at: Optional[datetime] = None
    price_op: Optional[str] = None
    price_level: Optional[float] = None
    hits_unresolved: bool = False


@dataclass(frozen=True)
class CloseRequest:
    id1: str
    id2: str
    qty: Optional[int]
    trigger: Trigger


def _fmt_clock(dt: datetime) -> str:
    hour = dt.hour % 12 or 12
    ampm = "am" if dt.hour < 12 else "pm"
    if dt.minute == 0:
        return f"{hour}{ampm}"
    return f"{hour}:{dt.minute:02d}{ampm}"


def _normalize_trigger_text(tokens: list[str]) -> str:
    return " ".join(tokens).strip()


def parse_trigger(text: str, now: Optional[datetime] = None) -> Trigger:
    """Parse ``at 2pm`` / ``in 1h`` / ``if spy >= 650`` / ``now``."""
    now = now or _now_et()
    if now.tzinfo is None:
        now = now.replace(tzinfo=ET)
    else:
        now = now.astimezone(ET)

    raw = " ".join(text.split()).strip().lower()
    if not raw:
        raise ValueError(_trigger_usage())
    if raw in ("now", "immediately", "asap"):
        return Trigger("now", "now", now)

    at = re.fullmatch(
        r"at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?",
        raw,
    )
    if at:
        hour = int(at.group(1))
        minute = int(at.group(2) or 0)
        ampm = at.group(3)
        if ampm:
            if hour == 12:
                hour = 0 if ampm == "am" else 12
            elif ampm == "pm":
                hour += 12
        elif hour > 23:
            raise ValueError(f"Bad time `{text}` — hour must be 0–23 (or use am/pm).")
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            raise ValueError(f"Bad time `{text}`.")
        when = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        rolled = False
        if when <= now:
            when = when + timedelta(days=1)
            rolled = True
        label = f"at {_fmt_clock(when)} ET"
        if rolled:
            label += f" ({when.strftime('%a %Y-%m-%d')})"
        return Trigger("at", label, when)

    delay = re.fullmatch(
        r"in\s+(\d+(?:\.\d+)?)\s*"
        r"(h|hr|hrs|hour|hours|m|min|mins|minute|minutes)"
        r"(?:\s+from\s+now)?",
        raw,
    )
    if delay:
        amount = float(delay.group(1))
        if amount <= 0:
            raise ValueError("Delay must be > 0.")
        unit = delay.group(2)
        if unit.startswith("h"):
            delta = timedelta(hours=amount)
            pretty = f"in {amount:g}h"
        else:
            delta = timedelta(minutes=amount)
            pretty = f"in {amount:g}m"
        when = now + delta
        return Trigger(
            "in",
            f"{pretty} → {_fmt_clock(when)} ET",
            when,
        )

    price = re.fullmatch(
        r"(?:(?:if|when)\s+)?(?:spy\s+)?"
        r"(>=|<=|>|<|=|hits|hit|reaches|reach)?"
        r"\s*(\d+(?:\.\d+)?)",
        raw,
    )
    if price:
        op_raw = (price.group(1) or "hits").lower()
        level = float(price.group(2))
        if level <= 0:
            raise ValueError("SPY price level must be > 0.")
        if op_raw in ("hits", "hit", "reaches", "reach", "="):
            return Trigger(
                "price",
                f"when SPY hits {level:g}",
                None,
                None,
                level,
                hits_unresolved=True,
            )
        return Trigger(
            "price",
            f"if SPY {op_raw} {level:g}",
            None,
            op_raw,
            level,
            hits_unresolved=False,
        )

    raise ValueError(f"Could not parse trigger `{text}`.\n{_trigger_usage()}")


def resolve_hits_trigger(trigger: Trigger, spy_price: float) -> Trigger:
    """Turn ``when spy hits N`` into ``>=`` or ``<=`` from the current print."""
    if not trigger.hits_unresolved or trigger.price_level is None:
        return trigger
    level = trigger.price_level
    if spy_price >= level:
        op = "<="
    else:
        op = ">="
    return Trigger(
        "price",
        f"when SPY hits {level:g} → {op} {level:g} (spot {spy_price:g})",
        None,
        op,
        level,
        hits_unresolved=False,
    )


def price_condition_met(
    op: Optional[str],
    level: Optional[float],
    spy_price: float,
) -> bool:
    if not op or level is None:
        return False
    if op == ">=":
        return spy_price >= level
    if op == "<=":
        return spy_price <= level
    if op == ">":
        return spy_price > level
    if op == "<":
        return spy_price < level
    return False


def parse_close_spread_args(
    args: list[str],
    now: Optional[datetime] = None,
) -> CloseRequest:
    """Parse ``<id1> <id2> [QTY] <trigger…>``."""
    if len(args) < 3:
        raise ValueError(
            "Usage: `!lia close spread <id1> <id2> [QTY] "
            "at 2pm|in 1h|if spy >= 650|now`\n"
            "Ids come from `!lia opt`. Order of the two legs does not matter."
        )
    id1 = args[0].strip().lower()
    id2 = args[1].strip().lower()
    rest = args[2:]
    qty: Optional[int] = None
    trigger_heads = {
        "at",
        "in",
        "if",
        "when",
        "now",
        "spy",
        "hits",
        "hit",
        "immediately",
        "asap",
        ">=",
        "<=",
        ">",
        "<",
        "=",
        "reaches",
        "reach",
    }
    if (
        len(rest) >= 2
        and re.fullmatch(r"\d+", rest[0])
        and rest[1].lower() in trigger_heads
    ):
        qty = int(rest[0])
        if qty <= 0:
            raise ValueError("quantity must be a positive whole number of contracts")
        rest = rest[1:]
    trigger_tokens = rest
    if not trigger_tokens:
        raise ValueError(
            "Missing trigger. Examples: `at 2pm`, `in 1h`, `if spy >= 650`, `now`."
        )
    trigger = parse_trigger(_normalize_trigger_text(trigger_tokens), now=now)
    return CloseRequest(id1=id1, id2=id2, qty=qty, trigger=trigger)


def _looks_like_id(token: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-fA-F]{4,12}", token))


def _trigger_usage() -> str:
    return (
        "Trigger must be one of:\n"
        "  `at 2pm` / `at 14:00` / `at 2:30pm`\n"
        "  `in 1h` / `in 30m` / `in 1 hour from now`\n"
        "  `if spy >= 650` / `if spy <= 640` / `when spy hits 650`\n"
        "  `now`"
    )


# ---------------------------------------------------------------------------
# Leg validation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpreadLegs:
    symbol: str
    expiration: str
    option_type: str
    long_id: str
    long_strike: float
    short_id: str
    short_strike: float
    qty: int
    long_label: str
    short_label: str


def _pos_label(pos: dict[str, Any]) -> str:
    sid = pos.get("_short_id") or "?"
    exp = pos.get("_expiration") or "?"
    strike = pos.get("_strike") or 0
    otype = (pos.get("_option_type") or "?").upper()
    side = pos.get("_side") or "?"
    return f"{sid} {side} {strike:g} {otype} {exp}"


def validate_spread_legs(
    pos1: dict[str, Any],
    pos2: dict[str, Any],
    qty: Optional[int],
    *,
    symbol: str = SPREAD_SYMBOL,
) -> SpreadLegs:
    """Require a SPY two-leg long+short of the same expiry and type."""
    for pos, name in ((pos1, "first"), (pos2, "second")):
        if not pos:
            raise ValueError(f"Unknown {name} option id. Run `!lia opt`.")

    sym1 = (pos1.get("_symbol") or "").upper()
    sym2 = (pos2.get("_symbol") or "").upper()
    if sym1 != symbol or sym2 != symbol:
        raise ValueError(
            f"Spread close is SPY-only for now "
            f"(got `{sym1 or '?'}` / `{sym2 or '?'}`)."
        )

    exp1 = pos1.get("_expiration") or ""
    exp2 = pos2.get("_expiration") or ""
    if not exp1 or exp1 != exp2:
        raise ValueError(
            f"Both legs must share an expiration (got `{exp1 or '?'}` / `{exp2 or '?'}`)."
        )

    t1 = (pos1.get("_option_type") or "").lower()
    t2 = (pos2.get("_option_type") or "").lower()
    if t1 not in ("call", "put") or t1 != t2:
        raise ValueError(
            f"Both legs must be the same type (got `{t1 or '?'}` / `{t2 or '?'}`)."
        )

    side1 = (pos1.get("_side") or "").lower()
    side2 = (pos2.get("_side") or "").lower()
    if {side1, side2} != {"long", "short"}:
        raise ValueError(
            "Need one long and one short leg "
            f"(got `{side1 or '?'}` / `{side2 or '?'}`)."
        )

    long_pos = pos1 if side1 == "long" else pos2
    short_pos = pos1 if side1 == "short" else pos2
    held = min(
        abs(int(float(long_pos.get("_qty") or 0))),
        abs(int(float(short_pos.get("_qty") or 0))),
    )
    if held <= 0:
        raise ValueError("Both legs need a positive contract quantity.")
    close_qty = held if qty is None else int(qty)
    if close_qty <= 0:
        raise ValueError("quantity must be a positive whole number of contracts")
    if close_qty > held:
        raise ValueError(
            f"Only `{held}` contract(s) available on both legs (asked `{close_qty}`)."
        )

    return SpreadLegs(
        symbol=symbol,
        expiration=str(exp1),
        option_type=t1,
        long_id=str(long_pos.get("_short_id") or ""),
        long_strike=float(long_pos.get("_strike") or 0),
        short_id=str(short_pos.get("_short_id") or ""),
        short_strike=float(short_pos.get("_strike") or 0),
        qty=close_qty,
        long_label=_pos_label(long_pos),
        short_label=_pos_label(short_pos),
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def insert_job(
    *,
    legs: SpreadLegs,
    trigger: Trigger,
    created_by: Optional[str] = None,
    channel_id: Optional[str] = None,
    status: str = STATUS_PENDING,
    db_path: str = DEFAULT_DB_PATH,
) -> dict[str, Any]:
    if trigger.hits_unresolved:
        raise ValueError("Resolve `hits` against a SPY print before inserting.")
    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO spread_close_jobs (
                created_at, created_by, channel_id, symbol, expiration,
                option_type, long_id, long_strike, short_id, short_strike,
                qty, trigger_kind, trigger_text, trigger_at, price_op,
                price_level, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _iso(_now_et()),
                created_by,
                channel_id,
                legs.symbol,
                legs.expiration,
                legs.option_type,
                legs.long_id,
                legs.long_strike,
                legs.short_id,
                legs.short_strike,
                legs.qty,
                trigger.kind,
                trigger.text,
                _iso(trigger.trigger_at) if trigger.trigger_at else None,
                trigger.price_op,
                trigger.price_level,
                status,
            ),
        )
        job_id = int(cur.lastrowid)
    job = get_job(job_id, db_path=db_path)
    assert job is not None
    return job


def get_job(job_id: int, *, db_path: str = DEFAULT_DB_PATH) -> Optional[dict[str, Any]]:
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM spread_close_jobs WHERE id = ?",
            (job_id,),
        ).fetchone()
    return _row_to_job(row) if row else None


def list_jobs(
    *,
    statuses: Optional[tuple[str, ...]] = None,
    limit: int = 25,
    db_path: str = DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    sql = "SELECT * FROM spread_close_jobs"
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
    return list_jobs(
        statuses=(STATUS_PENDING,),
        limit=100,
        db_path=db_path,
    )


def cancel_job(job_id: int, *, db_path: str = DEFAULT_DB_PATH) -> Optional[dict[str, Any]]:
    job = get_job(job_id, db_path=db_path)
    if job is None:
        return None
    if job["status"] != STATUS_PENDING:
        return job
    with _connect(db_path) as conn:
        conn.execute(
            """
            UPDATE spread_close_jobs
            SET status = ?
            WHERE id = ? AND status = ?
            """,
            (STATUS_CANCELLED, job_id, STATUS_PENDING),
        )
    return get_job(job_id, db_path=db_path)


def mark_job(
    job_id: int,
    *,
    status: str,
    result_text: Optional[str] = None,
    last_error: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH,
) -> Optional[dict[str, Any]]:
    fired_at = _iso(_now_et()) if status in {STATUS_FIRED, STATUS_PARTIAL, STATUS_ERROR} else None
    with _connect(db_path) as conn:
        conn.execute(
            """
            UPDATE spread_close_jobs
            SET status = ?, result_text = COALESCE(?, result_text),
                last_error = ?, fired_at = COALESCE(?, fired_at)
            WHERE id = ?
            """,
            (status, result_text, last_error, fired_at, job_id),
        )
    return get_job(job_id, db_path=db_path)


def claim_due_jobs(
    now: Optional[datetime] = None,
    spy_price: Optional[float] = None,
    *,
    db_path: str = DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    """Atomically mark due pending jobs as ``firing`` and return them."""
    now = now or _now_et()
    if now.tzinfo is None:
        now = now.replace(tzinfo=ET)
    else:
        now = now.astimezone(ET)
    pending = list_pending(db_path=db_path)
    due_ids: list[int] = []
    for job in pending:
        kind = job.get("trigger_kind")
        if kind in ("at", "in", "now"):
            when = _parse_iso(job.get("trigger_at"))
            if when is not None and when <= now:
                due_ids.append(int(job["id"]))
        elif kind == "price" and spy_price is not None:
            if price_condition_met(job.get("price_op"), job.get("price_level"), spy_price):
                due_ids.append(int(job["id"]))
    claimed: list[dict[str, Any]] = []
    if not due_ids:
        return claimed
    with _connect(db_path) as conn:
        for job_id in due_ids:
            cur = conn.execute(
                """
                UPDATE spread_close_jobs
                SET status = ?
                WHERE id = ? AND status = ?
                """,
                (STATUS_FIRING, job_id, STATUS_PENDING),
            )
            if cur.rowcount:
                row = conn.execute(
                    "SELECT * FROM spread_close_jobs WHERE id = ?",
                    (job_id,),
                ).fetchone()
                if row:
                    claimed.append(_row_to_job(row))
    return claimed


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def format_job_line(job: dict[str, Any]) -> str:
    jid = job.get("id")
    status = job.get("status") or "?"
    qty = job.get("qty")
    otype = (job.get("option_type") or "?").upper()
    exp = job.get("expiration") or "?"
    long_id = job.get("long_id") or "?"
    short_id = job.get("short_id") or "?"
    long_k = job.get("long_strike")
    short_k = job.get("short_strike")
    trig = job.get("trigger_text") or "?"
    long_s = f"{long_k:g}" if long_k not in (None, "") else "?"
    short_s = f"{short_k:g}" if short_k not in (None, "") else "?"
    return (
        f"`#{jid}` `{status}` {job.get('symbol') or 'SPY'} {exp} {otype} "
        f"x `{qty}` · short `{short_id}` {short_s} / long `{long_id}` {long_s} "
        f"· {trig}"
    )


def format_status(jobs: list[dict[str, Any]]) -> str:
    if not jobs:
        return (
            "No pending spread closes. "
            "Schedule one with `!lia close spread <id1> <id2> at 2pm` "
            "(ids from `!lia opt`)."
        )
    lines = [f"**Spread closes** · {len(jobs)}"]
    for job in jobs:
        lines.append(format_job_line(job))
    lines.append("_Cancel with_ `!lia close cancel <job>`")
    return "\n".join(lines)


def format_scheduled(job: dict[str, Any], legs: SpreadLegs) -> str:
    return (
        f"Watching spread close `#{job['id']}`\n"
        f"Buy-to-close short `{legs.short_label}`\n"
        f"Sell-to-close long `{legs.long_label}`\n"
        f"Qty `{legs.qty}` · trigger: **{job['trigger_text']}**\n"
        f"`!lia close status` to check · `!lia close cancel {job['id']}` to stop."
    )
