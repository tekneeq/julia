"""SQLite-backed store for `lia greeks` snapshots.

Each `greeks` invocation captures a point-in-time picture of an options
chain (one ticker × one expiration). Two snapshots taken at different times
can then be diffed to see what positioning was added.

Important caveat: ``open_interest`` is published once per day by the OCC;
it does *not* tick intraday. Same-day snapshots will show ΔOI = 0 even when
real trading is happening. Use ``volume`` (cumulative since open) for the
intraday signal; ΔOI is meaningful across days.

Tables
------
gex_snapshots
    One row per `greeks` run. Stores portfolio-level aggregates and the
    spot/IV regime so diffs are self-describing.

gex_strike_snapshots
    One row per (snapshot_id, option_type, strike) — the raw chain at the
    moment of the snapshot. Lets any per-strike diff be computed later.

The store reuses ``predictions_store.DEFAULT_DB_PATH`` so both feature
domains share one ``.options_cache/predictions.db`` file.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Optional

import ulid

from julia.predictions_store import DEFAULT_DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS gex_snapshots (
    id                  TEXT PRIMARY KEY,
    captured_at         TEXT NOT NULL,
    ticker              TEXT NOT NULL,
    expiration_date     TEXT NOT NULL,
    spot_price          REAL NOT NULL,
    risk_free_rate      REAL NOT NULL,
    call_oi             INTEGER NOT NULL,
    put_oi              INTEGER NOT NULL,
    call_volume         INTEGER NOT NULL,
    put_volume          INTEGER NOT NULL,
    call_notional_delta REAL,
    put_notional_delta  REAL,
    total_gex           REAL,
    call_gex            REAL,
    put_gex             REAL
);

CREATE INDEX IF NOT EXISTS idx_gex_lookup
    ON gex_snapshots(ticker, expiration_date, captured_at);

CREATE TABLE IF NOT EXISTS gex_strike_snapshots (
    snapshot_id      TEXT NOT NULL REFERENCES gex_snapshots(id),
    option_type      TEXT NOT NULL,  -- 'call' or 'put'
    strike_price     REAL NOT NULL,
    mark_price       REAL,
    implied_vol      REAL,
    delta            REAL,
    gamma            REAL,
    gex_per_contract REAL,
    volume           INTEGER,
    open_interest    INTEGER,
    PRIMARY KEY (snapshot_id, option_type, strike_price)
);
"""


@dataclass
class StrikeSnapshot:
    option_type: str  # 'call' or 'put'
    strike_price: float
    mark_price: Optional[float] = None
    implied_vol: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    gex_per_contract: Optional[float] = None
    volume: Optional[int] = None
    open_interest: Optional[int] = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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


def record_snapshot(
    *,
    ticker: str,
    expiration_date: str,
    spot_price: float,
    risk_free_rate: float,
    strikes: Iterable[StrikeSnapshot],
    db_path: str = DEFAULT_DB_PATH,
) -> str:
    """Insert one snapshot row + N per-strike rows. Returns the snapshot id.

    Portfolio aggregates are computed from ``strikes`` so the caller doesn't
    have to keep them in sync with the per-strike data.
    """
    strikes = list(strikes)
    call_oi = sum(int(s.open_interest or 0) for s in strikes if s.option_type == "call")
    put_oi = sum(int(s.open_interest or 0) for s in strikes if s.option_type == "put")
    call_vol = sum(int(s.volume or 0) for s in strikes if s.option_type == "call")
    put_vol = sum(int(s.volume or 0) for s in strikes if s.option_type == "put")

    # Notional delta in dollars: delta * 100 shares/contract * OI * spot.
    # Put deltas are negative, so put_notional_delta will be negative.
    def _notional(side: str) -> float:
        total = 0.0
        for s in strikes:
            if s.option_type != side:
                continue
            if s.delta is None or s.open_interest is None:
                continue
            total += float(s.delta) * 100.0 * float(s.open_interest) * spot_price
        return total

    call_nd = _notional("call")
    put_nd = _notional("put")

    call_gex = sum(
        float(s.gex_per_contract or 0.0)
        for s in strikes
        if s.option_type == "call"
    )
    put_gex = sum(
        float(s.gex_per_contract or 0.0)
        for s in strikes
        if s.option_type == "put"
    )
    total_gex = call_gex + put_gex

    sid = str(ulid.new())
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO gex_snapshots
                (id, captured_at, ticker, expiration_date, spot_price,
                 risk_free_rate, call_oi, put_oi, call_volume, put_volume,
                 call_notional_delta, put_notional_delta,
                 total_gex, call_gex, put_gex)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sid,
                _now_iso(),
                ticker,
                expiration_date,
                spot_price,
                risk_free_rate,
                call_oi,
                put_oi,
                call_vol,
                put_vol,
                call_nd,
                put_nd,
                total_gex,
                call_gex,
                put_gex,
            ),
        )
        conn.executemany(
            """
            INSERT INTO gex_strike_snapshots
                (snapshot_id, option_type, strike_price, mark_price,
                 implied_vol, delta, gamma, gex_per_contract,
                 volume, open_interest)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    sid,
                    s.option_type,
                    s.strike_price,
                    s.mark_price,
                    s.implied_vol,
                    s.delta,
                    s.gamma,
                    s.gex_per_contract,
                    s.volume,
                    s.open_interest,
                )
                for s in strikes
            ],
        )
    return sid


def recent_snapshots(
    *,
    ticker: Optional[str] = None,
    expiration_date: Optional[str] = None,
    limit: int = 20,
    db_path: str = DEFAULT_DB_PATH,
) -> list[sqlite3.Row]:
    with _connect(db_path) as conn:
        return conn.execute(
            """
            SELECT * FROM gex_snapshots
            WHERE (? IS NULL OR ticker = ?)
              AND (? IS NULL OR expiration_date = ?)
            ORDER BY captured_at DESC
            LIMIT ?
            """,
            (ticker, ticker, expiration_date, expiration_date, limit),
        ).fetchall()


def get_snapshot(
    snapshot_id: str,
    db_path: str = DEFAULT_DB_PATH,
) -> Optional[sqlite3.Row]:
    with _connect(db_path) as conn:
        return conn.execute(
            "SELECT * FROM gex_snapshots WHERE id = ?",
            (snapshot_id,),
        ).fetchone()


def get_strikes(
    snapshot_id: str,
    db_path: str = DEFAULT_DB_PATH,
) -> list[sqlite3.Row]:
    with _connect(db_path) as conn:
        return conn.execute(
            """
            SELECT * FROM gex_strike_snapshots
            WHERE snapshot_id = ?
            ORDER BY option_type, strike_price
            """,
            (snapshot_id,),
        ).fetchall()


def latest_two(
    *,
    ticker: str,
    expiration_date: str,
    db_path: str = DEFAULT_DB_PATH,
) -> tuple[Optional[sqlite3.Row], Optional[sqlite3.Row]]:
    """Return (older, newer) of the two most recent snapshots, or (None, None)."""
    rows = recent_snapshots(
        ticker=ticker, expiration_date=expiration_date, limit=2, db_path=db_path
    )
    if len(rows) < 2:
        return (None, None)
    newer, older = rows[0], rows[1]
    return (older, newer)


def first_snapshot(
    *,
    ticker: str,
    expiration_date: str,
    db_path: str = DEFAULT_DB_PATH,
) -> Optional[sqlite3.Row]:
    """The earliest snapshot for (ticker, expiration_date), if any."""
    with _connect(db_path) as conn:
        return conn.execute(
            """
            SELECT * FROM gex_snapshots
            WHERE ticker = ? AND expiration_date = ?
            ORDER BY captured_at ASC
            LIMIT 1
            """,
            (ticker, expiration_date),
        ).fetchone()


def nth_most_recent(
    *,
    ticker: str,
    expiration_date: str,
    n: int,
    db_path: str = DEFAULT_DB_PATH,
) -> Optional[sqlite3.Row]:
    """N-th most recent snapshot (1 = latest). Returns None if not enough rows."""
    if n < 1:
        return None
    rows = recent_snapshots(
        ticker=ticker, expiration_date=expiration_date, limit=n, db_path=db_path
    )
    return rows[n - 1] if len(rows) >= n else None
