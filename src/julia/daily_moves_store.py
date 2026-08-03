"""SQLite store for intraday price paths (today + last N sessions).

Each trading day for a ticker is a session of ``(minutes_from_open, spot,
pct_vs_ref)`` points. ``ref_spot`` is the prior session's close (last
known spot), so 0% is where yesterday left off — same convention as the
implied-vs-actual candles.

Points are upserted from three sources (highest fidelity first):
  * ``tick``     — live prints from ``scripts/price_poller.py`` (seconds)
  * ``market``   — Robinhood 5-minute bars (when authenticated)
  * ``snapshot`` — spot stamps from ``gex_snapshots`` (batch cadence)

``spot_ticks`` additionally keeps the raw sub-minute prints from the
price poller so the live session chart can be rendered at full
resolution instead of collapsed to one point per minute.

Retention keeps today + the most recent ``KEEP_SESSIONS`` completed
sessions (default 30). Older sessions are pruned on sync; raw ticks
are pruned separately (``prune_ticks``) since only recent days need
sub-minute detail.
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, time as dtime, timezone
from typing import Optional

from julia.predictions_store import DEFAULT_DB_PATH

KEEP_SESSIONS = 30
KEEP_TICK_SESSIONS = 10
# NYSE regular session in local (container) time — Dockerfile pins ET.
MARKET_OPEN = dtime(9, 30)
MARKET_CLOSE = dtime(16, 0)

# When two sources land on the same minute, the higher rank wins.
_SOURCE_RANK = {"snapshot": 0, "market": 1, "tick": 2}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS daily_move_sessions (
    ticker       TEXT NOT NULL,
    session_date TEXT NOT NULL,
    ref_spot     REAL NOT NULL,
    PRIMARY KEY (ticker, session_date)
);

CREATE TABLE IF NOT EXISTS daily_move_points (
    ticker            TEXT NOT NULL,
    session_date      TEXT NOT NULL,
    minutes_from_open INTEGER NOT NULL,
    spot              REAL NOT NULL,
    pct               REAL NOT NULL,
    source            TEXT NOT NULL,  -- 'tick' | 'market' | 'snapshot'
    PRIMARY KEY (ticker, session_date, minutes_from_open, source),
    FOREIGN KEY (ticker, session_date)
        REFERENCES daily_move_sessions(ticker, session_date)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_daily_move_points_lookup
    ON daily_move_points(ticker, session_date, minutes_from_open);

CREATE TABLE IF NOT EXISTS spot_ticks (
    ticker       TEXT NOT NULL,
    session_date TEXT NOT NULL,  -- local (ET) trading day
    ts_utc       TEXT NOT NULL,  -- ISO-8601 UTC, seconds precision
    price        REAL NOT NULL,
    PRIMARY KEY (ticker, ts_utc)
);

CREATE INDEX IF NOT EXISTS idx_spot_ticks_session
    ON spot_ticks(ticker, session_date, ts_utc);
"""


@contextmanager
def _connect(db_path: str = DEFAULT_DB_PATH):
    parent = os.path.dirname(db_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    # The price poller, the Streamlit app, and the batch all write to
    # this DB concurrently. WAL lets readers proceed during writes, and
    # the busy timeout turns "database is locked" into a short wait.
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA journal_mode = WAL")
    try:
        conn.executescript(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def minutes_from_open(local_dt: datetime) -> int:
    """Minutes since 09:30 local on ``local_dt``'s date (can be negative)."""
    open_dt = datetime.combine(local_dt.date(), MARKET_OPEN)
    if local_dt.tzinfo is not None:
        open_dt = open_dt.replace(tzinfo=local_dt.tzinfo)
    return int((local_dt - open_dt).total_seconds() // 60)


def upsert_session(
    *,
    ticker: str,
    session_date: date,
    ref_spot: float,
    db_path: str = DEFAULT_DB_PATH,
) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO daily_move_sessions (ticker, session_date, ref_spot)
            VALUES (?, ?, ?)
            ON CONFLICT(ticker, session_date) DO UPDATE SET
                ref_spot = excluded.ref_spot
            """,
            (ticker, session_date.isoformat(), float(ref_spot)),
        )


def rebase_session(
    *,
    ticker: str,
    session_date: date,
    ref_spot: float,
    db_path: str = DEFAULT_DB_PATH,
) -> None:
    """Set ``ref_spot`` and recompute every point's ``pct`` against it.

    Used when we discover the official prior close after points were
    already written against a stale snapshot-based ref.
    """
    ref = float(ref_spot)
    iso = session_date.isoformat()
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO daily_move_sessions (ticker, session_date, ref_spot)
            VALUES (?, ?, ?)
            ON CONFLICT(ticker, session_date) DO UPDATE SET
                ref_spot = excluded.ref_spot
            """,
            (ticker, iso, ref),
        )
        conn.execute(
            """
            UPDATE daily_move_points
            SET pct = (spot - ?) / ? * 100.0
            WHERE ticker = ? AND session_date = ? AND ? != 0
            """,
            (ref, ref, ticker, iso, ref),
        )


def upsert_points(
    *,
    ticker: str,
    session_date: date,
    points: list[tuple[int, float, float, str]],
    db_path: str = DEFAULT_DB_PATH,
) -> None:
    """``points`` = [(minutes_from_open, spot, pct, source), ...]"""
    if not points:
        return
    with _connect(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO daily_move_points
                (ticker, session_date, minutes_from_open, spot, pct, source)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker, session_date, minutes_from_open, source)
            DO UPDATE SET spot = excluded.spot, pct = excluded.pct
            """,
            [
                (
                    ticker,
                    session_date.isoformat(),
                    int(m),
                    float(spot),
                    float(pct),
                    source,
                )
                for m, spot, pct, source in points
            ],
        )


def list_sessions(
    ticker: str,
    *,
    db_path: str = DEFAULT_DB_PATH,
) -> list[sqlite3.Row]:
    """Sessions oldest → newest."""
    with _connect(db_path) as conn:
        return conn.execute(
            """
            SELECT * FROM daily_move_sessions
            WHERE ticker = ?
            ORDER BY session_date ASC
            """,
            (ticker,),
        ).fetchall()


def get_session_path(
    ticker: str,
    session_date: date,
    *,
    prefer_source: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH,
) -> list[dict]:
    """Points for one session, oldest → newest.

    When multiple sources land on the same minute the highest-fidelity
    one wins: ``tick`` > ``market`` > ``snapshot``. ``prefer_source``
    overrides that ordering by bumping one source to the top.
    """
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT minutes_from_open, spot, pct, source
            FROM daily_move_points
            WHERE ticker = ? AND session_date = ?
            ORDER BY minutes_from_open ASC, source ASC
            """,
            (ticker, session_date.isoformat()),
        ).fetchall()

    def _rank(source: str) -> int:
        if prefer_source is not None and source == prefer_source:
            return 99
        return _SOURCE_RANK.get(source, -1)

    by_min: dict[int, dict] = {}
    for r in rows:
        m = int(r["minutes_from_open"])
        entry = {
            "minutes_from_open": m,
            "spot": float(r["spot"]),
            "pct": float(r["pct"]),
            "source": r["source"],
        }
        if m not in by_min or _rank(r["source"]) > _rank(by_min[m]["source"]):
            by_min[m] = entry
    return [by_min[m] for m in sorted(by_min)]


# ---------------------------------------------------------------------------
# Raw ticks — sub-minute prints from the dedicated price poller
# ---------------------------------------------------------------------------

def record_tick(
    *,
    ticker: str,
    ts: datetime,
    price: float,
    db_path: str = DEFAULT_DB_PATH,
) -> None:
    """Store one live price print. ``ts`` must be timezone-aware."""
    if ts.tzinfo is None:
        raise ValueError("record_tick requires a timezone-aware timestamp")
    local = ts.astimezone()
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO spot_ticks (ticker, session_date, ts_utc, price)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(ticker, ts_utc) DO UPDATE SET
                price = excluded.price
            """,
            (
                ticker,
                local.date().isoformat(),
                ts.astimezone(timezone.utc).isoformat(timespec="seconds"),
                float(price),
            ),
        )


def get_ticks(
    ticker: str,
    session_date: date,
    *,
    db_path: str = DEFAULT_DB_PATH,
) -> list[dict]:
    """All prints for one local trading day, oldest → newest.

    Returns ``[{"ts": aware local datetime, "price": float}, ...]``.
    """
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT ts_utc, price FROM spot_ticks
            WHERE ticker = ? AND session_date = ?
            ORDER BY ts_utc ASC
            """,
            (ticker, session_date.isoformat()),
        ).fetchall()
    return [
        {
            "ts": datetime.fromisoformat(r["ts_utc"]).astimezone(),
            "price": float(r["price"]),
        }
        for r in rows
    ]


def latest_tick(
    ticker: str,
    *,
    db_path: str = DEFAULT_DB_PATH,
) -> Optional[dict]:
    """The most recent print for ``ticker`` across all days, or None."""
    with _connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT ts_utc, price, session_date FROM spot_ticks
            WHERE ticker = ?
            ORDER BY ts_utc DESC LIMIT 1
            """,
            (ticker,),
        ).fetchone()
    if row is None:
        return None
    return {
        "ts": datetime.fromisoformat(row["ts_utc"]).astimezone(),
        "price": float(row["price"]),
        "session_date": row["session_date"],
    }


def prune_ticks(
    ticker: str,
    *,
    keep_sessions: int = KEEP_TICK_SESSIONS,
    db_path: str = DEFAULT_DB_PATH,
) -> int:
    """Drop raw ticks for all but the newest ``keep_sessions`` days.

    Per-minute points distilled from those ticks stay in
    ``daily_move_points``, so twin matching keeps its history — only
    the sub-minute detail is discarded. Returns days deleted.
    """
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT session_date FROM spot_ticks
            WHERE ticker = ?
            ORDER BY session_date DESC
            """,
            (ticker,),
        ).fetchall()
        doomed = [r["session_date"] for r in rows[keep_sessions:]]
        for d in doomed:
            conn.execute(
                "DELETE FROM spot_ticks WHERE ticker = ? AND session_date = ?",
                (ticker, d),
            )
        return len(doomed)


def prune(
    ticker: str,
    *,
    keep: int = KEEP_SESSIONS,
    today: Optional[date] = None,
    db_path: str = DEFAULT_DB_PATH,
) -> int:
    """Drop completed sessions beyond the newest ``keep``, always retaining today.

    Returns the number of sessions deleted.
    """
    today = today or date.today()
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT session_date FROM daily_move_sessions
            WHERE ticker = ?
            ORDER BY session_date DESC
            """,
            (ticker,),
        ).fetchall()
        completed = [
            r["session_date"]
            for r in rows
            if r["session_date"] != today.isoformat()
        ]
        doomed = completed[keep:]
        for d in doomed:
            conn.execute(
                "DELETE FROM daily_move_sessions WHERE ticker = ? AND session_date = ?",
                (ticker, d),
            )
        return len(doomed)
