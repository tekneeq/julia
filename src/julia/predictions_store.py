"""SQLite-backed store for `lia emove` predictions and their realized outcomes.

Tables
------
predictions
    One row per `lia emove` invocation. Captures the snapshot that the
    forecast was conditioned on (spot, IV, the option used to source IV).

prediction_bands
    One row per (days, confidence) cell of the printed table. The implied
    move and the resulting low/high price range are stored as raw floats so
    accuracy stats can be recomputed under different definitions later.

outcomes
    One row per (prediction_id, days) once the target date has passed and a
    realized close has been fetched. Whether a band "hit" is derived on
    query (``ABS(actual_move) <= implied_move``), never stored, so the rule
    can evolve without backfills.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Optional

import ulid

DEFAULT_DB_PATH = os.path.join(".options_cache", "predictions.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id              TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL,
    ticker          TEXT NOT NULL,
    spot_price      REAL NOT NULL,
    iv              REAL NOT NULL,
    strike_price    REAL NOT NULL,
    option_type     TEXT NOT NULL,
    expiration_date TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_predictions_ticker_created
    ON predictions(ticker, created_at);

CREATE TABLE IF NOT EXISTS prediction_bands (
    prediction_id   TEXT NOT NULL REFERENCES predictions(id),
    days            INTEGER NOT NULL,
    target_date     TEXT NOT NULL,
    confidence      REAL NOT NULL,
    implied_move    REAL NOT NULL,
    low             REAL NOT NULL,
    high            REAL NOT NULL,
    PRIMARY KEY (prediction_id, days, confidence)
);

CREATE INDEX IF NOT EXISTS idx_bands_target
    ON prediction_bands(target_date);

CREATE TABLE IF NOT EXISTS outcomes (
    prediction_id   TEXT NOT NULL REFERENCES predictions(id),
    days            INTEGER NOT NULL,
    target_date     TEXT NOT NULL,
    actual_price    REAL NOT NULL,
    actual_move     REAL NOT NULL,
    recorded_at     TEXT NOT NULL,
    PRIMARY KEY (prediction_id, days)
);
"""


@dataclass
class Band:
    """A single (days, confidence) cell of the implied-move table."""

    days: int
    target_date: str
    confidence: float
    implied_move: float
    low: float
    high: float


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


def record_prediction(
    *,
    ticker: str,
    spot_price: float,
    iv: float,
    strike_price: float,
    option_type: str,
    expiration_date: str,
    bands: Iterable[Band],
    db_path: str = DEFAULT_DB_PATH,
) -> str:
    """Insert one prediction and its bands; return the new prediction id."""
    pid = str(ulid.new())
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO predictions
                (id, created_at, ticker, spot_price, iv,
                 strike_price, option_type, expiration_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                pid,
                _now_iso(),
                ticker,
                spot_price,
                iv,
                strike_price,
                option_type,
                expiration_date,
            ),
        )
        conn.executemany(
            """
            INSERT INTO prediction_bands
                (prediction_id, days, target_date, confidence,
                 implied_move, low, high)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    pid,
                    b.days,
                    b.target_date,
                    b.confidence,
                    b.implied_move,
                    b.low,
                    b.high,
                )
                for b in bands
            ],
        )
    return pid


def pending_outcomes(
    today: str,
    *,
    ticker: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH,
) -> list[sqlite3.Row]:
    """Distinct (prediction_id, ticker, days, target_date, spot_price) that
    still need a realized close.

    A row is pending when its `target_date <= today` and no matching row
    exists in `outcomes`. The same (prediction_id, days) appears once even
    though it has multiple confidence bands.
    """
    with _connect(db_path) as conn:
        return conn.execute(
            """
            SELECT DISTINCT
                p.id          AS prediction_id,
                p.ticker      AS ticker,
                p.spot_price  AS spot_price,
                b.days        AS days,
                b.target_date AS target_date
            FROM prediction_bands b
            JOIN predictions p ON p.id = b.prediction_id
            LEFT JOIN outcomes o
                ON o.prediction_id = b.prediction_id
                AND o.days = b.days
            WHERE b.target_date <= ?
              AND o.prediction_id IS NULL
              AND (? IS NULL OR p.ticker = ?)
            ORDER BY b.target_date, p.ticker
            """,
            (today, ticker, ticker),
        ).fetchall()


def record_outcome(
    *,
    prediction_id: str,
    days: int,
    target_date: str,
    actual_price: float,
    actual_move: float,
    db_path: str = DEFAULT_DB_PATH,
) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO outcomes
                (prediction_id, days, target_date,
                 actual_price, actual_move, recorded_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                prediction_id,
                days,
                target_date,
                actual_price,
                actual_move,
                _now_iso(),
            ),
        )


def stats_rows(
    *,
    ticker: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH,
) -> list[sqlite3.Row]:
    """Per-(ticker, days, confidence) calibration stats."""
    with _connect(db_path) as conn:
        return conn.execute(
            """
            SELECT
                p.ticker                              AS ticker,
                b.days                                AS days,
                b.confidence                          AS confidence,
                COUNT(*)                              AS n,
                SUM(CASE WHEN ABS(o.actual_move) <= b.implied_move
                         THEN 1 ELSE 0 END)           AS hits,
                AVG(b.implied_move / p.spot_price)    AS avg_implied_pct,
                AVG(ABS(o.actual_move) / p.spot_price) AS avg_abs_move_pct
            FROM prediction_bands b
            JOIN predictions p ON p.id = b.prediction_id
            JOIN outcomes o
                ON o.prediction_id = b.prediction_id
                AND o.days = b.days
            WHERE (? IS NULL OR p.ticker = ?)
            GROUP BY p.ticker, b.days, b.confidence
            ORDER BY p.ticker, b.days, b.confidence
            """,
            (ticker, ticker),
        ).fetchall()


def counts(db_path: str = DEFAULT_DB_PATH) -> dict[str, int]:
    """Summary counts useful for `emove-stats` headers."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM predictions)      AS predictions,
                (SELECT COUNT(*) FROM prediction_bands) AS bands,
                (SELECT COUNT(*) FROM outcomes)         AS outcomes
            """
        ).fetchone()
    return dict(rows)
