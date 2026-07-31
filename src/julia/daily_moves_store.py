"""SQLite store for intraday % move paths (today + last N sessions).

Each trading day for a ticker is a session of ``(minutes_from_open, spot,
pct_vs_ref)`` points. ``ref_spot`` is the prior session's close (last
known spot), so 0% is where yesterday left off — same convention as the
implied-vs-actual candles.

Points are upserted from two sources:
  * ``snapshot`` — spot stamps from ``gex_snapshots`` (batch cadence)
  * ``market``   — Robinhood 5-minute bars (denser, when authenticated)

Retention keeps today + the most recent ``KEEP_SESSIONS`` completed
sessions (default 30). Older sessions are pruned on sync.
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, time as dtime
from typing import Optional

from julia.predictions_store import DEFAULT_DB_PATH

KEEP_SESSIONS = 30
# NYSE regular session in local (container) time — Dockerfile pins ET.
MARKET_OPEN = dtime(9, 30)
MARKET_CLOSE = dtime(16, 0)

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
    source            TEXT NOT NULL,  -- 'snapshot' | 'market'
    PRIMARY KEY (ticker, session_date, minutes_from_open, source),
    FOREIGN KEY (ticker, session_date)
        REFERENCES daily_move_sessions(ticker, session_date)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_daily_move_points_lookup
    ON daily_move_points(ticker, session_date, minutes_from_open);
"""


@contextmanager
def _connect(db_path: str = DEFAULT_DB_PATH):
    parent = os.path.dirname(db_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
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

    When both ``market`` and ``snapshot`` exist at the same minute,
    ``prefer_source`` wins ('market' recommended). Otherwise market
    points are preferred over snapshot at equal minutes.
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

    prefer = prefer_source or "market"
    by_min: dict[int, dict] = {}
    for r in rows:
        m = int(r["minutes_from_open"])
        entry = {
            "minutes_from_open": m,
            "spot": float(r["spot"]),
            "pct": float(r["pct"]),
            "source": r["source"],
        }
        if m not in by_min:
            by_min[m] = entry
        elif r["source"] == prefer:
            by_min[m] = entry
    return [by_min[m] for m in sorted(by_min)]


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
