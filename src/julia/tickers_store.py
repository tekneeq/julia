"""SQLite cache of Robinhood-tradable tickers grouped by sector / industry.

Pulls the public instruments catalog (``/instruments/``) and fundamentals
(``/fundamentals/``) — no Robinhood login required — and stores every
currently tradable symbol with its Seeking Alpha–style sector / industry
labels for the dashboard Tickers tab.

Cache lives at ``.options_cache/tickers.db`` (alongside the predictions DB).
"""
from __future__ import annotations

import os
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Optional

import requests

DEFAULT_DB_PATH = os.path.join(".options_cache", "tickers.db")

INSTRUMENTS_URL = "https://api.robinhood.com/instruments/"
FUNDAMENTALS_URL = "https://api.robinhood.com/fundamentals/"
USER_AGENT = "julia-dashboard/0.1 (+https://github.com/tekneeq/julia)"

# Fundamentals ``symbols=`` query; RH accepts ~100 but 50 is safer.
FUNDAMENTALS_BATCH = 50

# Equity-like types we surface in the sector browser. Warrants / rights /
# preferreds stay out of the Seeking Alpha–style equity tree.
EQUITY_TYPES = frozenset(
    {"stock", "adr", "etp", "reit", "cef", "mlp", "lp", "tracking"}
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tickers (
    symbol       TEXT PRIMARY KEY,
    name         TEXT,
    simple_name  TEXT,
    instrument_type TEXT,
    country      TEXT,
    list_date    TEXT,
    tradeable    INTEGER NOT NULL DEFAULT 1,
    sector       TEXT,
    industry     TEXT,
    market_cap   REAL,
    pe_ratio     REAL,
    dividend_yield REAL,
    description  TEXT,
    updated_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tickers_sector
    ON tickers(sector, industry, symbol);

CREATE INDEX IF NOT EXISTS idx_tickers_industry
    ON tickers(industry, symbol);

CREATE INDEX IF NOT EXISTS idx_tickers_type
    ON tickers(instrument_type);
"""


ProgressCb = Callable[[str, int, int], None]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        }
    )
    return s


@contextmanager
def _connect(db_path: str = DEFAULT_DB_PATH):
    parent = os.path.dirname(db_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def _set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


def get_meta(key: str, db_path: str = DEFAULT_DB_PATH) -> Optional[str]:
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT value FROM meta WHERE key = ?", (key,)
        ).fetchone()
    return row["value"] if row else None


def ticker_count(db_path: str = DEFAULT_DB_PATH) -> int:
    with _connect(db_path) as conn:
        row = conn.execute("SELECT COUNT(*) AS n FROM tickers").fetchone()
    return int(row["n"]) if row else 0


def _is_tradable(inst: dict[str, Any]) -> bool:
    """Active equity the Robinhood app will let you open a new position in."""
    return (
        bool(inst.get("symbol"))
        and inst.get("state") == "active"
        and inst.get("tradeable") is True
        and inst.get("tradability") == "tradable"
        and inst.get("rhs_tradability") == "tradable"
    )


def _fetch_all_instruments(
    session: requests.Session,
    progress: Optional[ProgressCb] = None,
) -> list[dict[str, Any]]:
    """Paginate Robinhood instruments; return active RH-tradable rows."""
    url: Optional[str] = INSTRUMENTS_URL
    params: Optional[dict[str, str]] = {"active_instruments_only": "true"}
    out: list[dict[str, Any]] = []
    pages = 0
    while url:
        resp = session.get(url, params=params, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        params = None  # cursor is embedded in ``next``
        pages += 1
        for inst in payload.get("results") or []:
            if not _is_tradable(inst):
                continue
            itype = (inst.get("type") or "unknown").lower()
            if itype not in EQUITY_TYPES:
                continue
            out.append(inst)
        if progress and pages % 10 == 0:
            progress("instruments", pages, len(out))
        url = payload.get("next")
    if progress:
        progress("instruments", pages, len(out))
    return out


def _fetch_fundamentals_batch(
    session: requests.Session, symbols: list[str]
) -> list[Optional[dict[str, Any]]]:
    """Return fundamentals aligned 1:1 with ``symbols`` (None on miss)."""
    if not symbols:
        return []
    resp = session.get(
        FUNDAMENTALS_URL,
        params={"symbols": ",".join(symbols)},
        timeout=60,
    )
    resp.raise_for_status()
    results = resp.json().get("results") or []
    # Pad / trim to match request length; RH returns nulls for unknowns.
    out: list[Optional[dict[str, Any]]] = []
    for i, sym in enumerate(symbols):
        item = results[i] if i < len(results) else None
        if item is None:
            out.append(None)
            continue
        item = dict(item)
        item["symbol"] = sym
        out.append(item)
    return out


def _fnum(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def sync_tradable_tickers(
    *,
    db_path: str = DEFAULT_DB_PATH,
    progress: Optional[ProgressCb] = None,
    sleep_s: float = 0.05,
) -> dict[str, Any]:
    """Full refresh of the tradable-ticker cache from Robinhood public APIs.

    Returns a small stats dict: ``{"instruments": N, "with_sector": M, ...}``.
    """
    session = _session()
    instruments = _fetch_all_instruments(session, progress=progress)
    symbols = [i["symbol"].upper() for i in instruments]
    by_symbol = {i["symbol"].upper(): i for i in instruments}

    fundamentals: dict[str, dict[str, Any]] = {}
    total_batches = (len(symbols) + FUNDAMENTALS_BATCH - 1) // FUNDAMENTALS_BATCH
    for bi in range(total_batches):
        chunk = symbols[bi * FUNDAMENTALS_BATCH : (bi + 1) * FUNDAMENTALS_BATCH]
        try:
            batch = _fetch_fundamentals_batch(session, chunk)
        except requests.RequestException:
            # One failed batch shouldn't kill the whole sync — leave sector blank.
            batch = [None] * len(chunk)
        for item in batch:
            if item and item.get("symbol"):
                fundamentals[item["symbol"].upper()] = item
        if progress:
            progress("fundamentals", bi + 1, total_batches)
        if sleep_s:
            time.sleep(sleep_s)

    now = _now_iso()
    rows = []
    with_sector = 0
    for sym in symbols:
        inst = by_symbol[sym]
        fund = fundamentals.get(sym) or {}
        sector = (fund.get("sector") or "").strip() or None
        industry = (fund.get("industry") or "").strip() or None
        if sector:
            with_sector += 1
        rows.append(
            (
                sym,
                inst.get("name"),
                inst.get("simple_name"),
                (inst.get("type") or "unknown").lower(),
                inst.get("country"),
                inst.get("list_date"),
                1,
                sector,
                industry,
                _fnum(fund.get("market_cap")),
                _fnum(fund.get("pe_ratio")),
                _fnum(fund.get("dividend_yield")),
                fund.get("description"),
                now,
            )
        )

    with _connect(db_path) as conn:
        conn.execute("DELETE FROM tickers")
        conn.executemany(
            """
            INSERT INTO tickers (
                symbol, name, simple_name, instrument_type, country, list_date,
                tradeable, sector, industry, market_cap, pe_ratio,
                dividend_yield, description, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        _set_meta(conn, "last_synced_at", now)
        _set_meta(conn, "instrument_count", str(len(rows)))

    return {
        "instruments": len(rows),
        "with_sector": with_sector,
        "synced_at": now,
    }


def list_sectors(
    db_path: str = DEFAULT_DB_PATH,
    instrument_types: Optional[Iterable[str]] = None,
) -> list[dict[str, Any]]:
    """Sector → ticker count, Seeking Alpha–style overview."""
    clauses = ["1=1"]
    params: list[Any] = []
    if instrument_types:
        types = [t.lower() for t in instrument_types]
        placeholders = ",".join("?" for _ in types)
        clauses.append(f"instrument_type IN ({placeholders})")
        params.extend(types)
    where = " AND ".join(clauses)
    with _connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT COALESCE(NULLIF(sector, ''), 'Unclassified') AS sector,
                   COUNT(*) AS n
            FROM tickers
            WHERE {where}
            GROUP BY 1
            ORDER BY n DESC, sector ASC
            """,
            params,
        ).fetchall()
    return [{"sector": r["sector"], "count": int(r["n"])} for r in rows]


def list_industries(
    sector: str,
    db_path: str = DEFAULT_DB_PATH,
    instrument_types: Optional[Iterable[str]] = None,
) -> list[dict[str, Any]]:
    """Industries under a sector with counts."""
    unclassified = sector == "Unclassified"
    clauses: list[str] = []
    params: list[Any] = []
    if unclassified:
        clauses.append("(sector IS NULL OR TRIM(sector) = '')")
    else:
        clauses.append("sector = ?")
        params.append(sector)
    if instrument_types:
        types = [t.lower() for t in instrument_types]
        placeholders = ",".join("?" for _ in types)
        clauses.append(f"instrument_type IN ({placeholders})")
        params.extend(types)
    where = " AND ".join(clauses)
    with _connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT COALESCE(NULLIF(industry, ''), 'Unclassified') AS industry,
                   COUNT(*) AS n
            FROM tickers
            WHERE {where}
            GROUP BY 1
            ORDER BY n DESC, industry ASC
            """,
            params,
        ).fetchall()
    return [{"industry": r["industry"], "count": int(r["n"])} for r in rows]


def list_tickers(
    *,
    sector: Optional[str] = None,
    industry: Optional[str] = None,
    search: Optional[str] = None,
    instrument_types: Optional[Iterable[str]] = None,
    limit: int = 5000,
    db_path: str = DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    """Filter the cached universe for the Tickers table."""
    clauses: list[str] = ["1=1"]
    params: list[Any] = []

    if sector == "Unclassified":
        clauses.append("(sector IS NULL OR TRIM(sector) = '')")
    elif sector:
        clauses.append("sector = ?")
        params.append(sector)

    if industry == "Unclassified":
        clauses.append("(industry IS NULL OR TRIM(industry) = '')")
    elif industry:
        clauses.append("industry = ?")
        params.append(industry)

    if search:
        q = f"%{search.strip().upper()}%"
        clauses.append(
            "(UPPER(symbol) LIKE ? OR UPPER(COALESCE(name, '')) LIKE ? "
            "OR UPPER(COALESCE(simple_name, '')) LIKE ?)"
        )
        params.extend([q, q, q])

    if instrument_types:
        types = [t.lower() for t in instrument_types]
        placeholders = ",".join("?" for _ in types)
        clauses.append(f"instrument_type IN ({placeholders})")
        params.extend(types)

    sql = f"""
        SELECT symbol, name, simple_name, instrument_type, country,
               sector, industry, market_cap, pe_ratio, dividend_yield,
               list_date
        FROM tickers
        WHERE {' AND '.join(clauses)}
        ORDER BY
            CASE WHEN market_cap IS NULL THEN 1 ELSE 0 END,
            market_cap DESC,
            symbol ASC
        LIMIT ?
    """
    params.append(int(limit))

    with _connect(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def sector_industry_tree(
    db_path: str = DEFAULT_DB_PATH,
    instrument_types: Optional[Iterable[str]] = None,
) -> list[dict[str, Any]]:
    """Nested sector → industries structure for the left-nav browser."""
    tree: list[dict[str, Any]] = []
    for sec in list_sectors(db_path, instrument_types=instrument_types):
        industries = list_industries(
            sec["sector"], db_path=db_path, instrument_types=instrument_types
        )
        tree.append(
            {
                "sector": sec["sector"],
                "count": sec["count"],
                "industries": industries,
            }
        )
    return tree
