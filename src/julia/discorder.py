"""Julia Discord bot — ``!lia`` command surface.

Messages that start with ``!lia`` (and ``!spy watch``) are handled:

    !lia help
    !lia buy  AAPL 1
    !lia sell AAPL 0.5
    !lia today              # stock buys submitted/filled today (ET)
    !lia positions          # stock holdings
    !lia opt                # open option positions (with short ids)
    !lia buy  opt TICKER YYYY-MM-DD STRIKE call|put QTY [LIMIT]
    !lia sell opt <id> [QTY] [LIMIT]
    !lia open spread TICKER EXP SHORT/LONG call|put QTY [CREDIT]
    !lia close spread <id1> <id2> [QTY] at 2pm|in 1h|if spy >= 650|now
    !lia close status
    !lia close cancel <job>
    !lia watch SPY 1m|1%|-1%|759
    !lia watch / !spy watch
    !lia watch close <id> / !spy watch close <id>

Buy/sell submit **live Robinhood orders** using the host
``RH_USERNAME`` / ``RH_PASSWORD``.

Env (``.env`` on the julia EC2 host, loaded by docker ``--env-file``):

    DISCORD_BOT_TOKEN         required
    DISCORD_CHANNEL_ID        optional ready greeting
    RH_USERNAME / RH_PASSWORD required for trading / portfolio commands
    LIA_DISCORD_ALLOWLIST     optional comma-separated Discord user IDs
                              allowed to run buy/sell/close (everyone if unset)
"""
from __future__ import annotations

import asyncio
import os
import re
import traceback
from datetime import date, datetime
from typing import Any, Optional
from zoneinfo import ZoneInfo

from julia import price_watch, spread_close, spread_open

from dotenv import load_dotenv

load_dotenv()

try:
    import discord
    from discord import Intents
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "discord package missing — install with `uv sync` / `pip install discord`"
    ) from e


ET = ZoneInfo("America/New_York")

HELP_TEXT = """**Julia · `!lia` commands**
```
!lia help                              this message
!lia buy  TICKER QTY                   stock market buy
!lia sell TICKER QTY                   stock market sell
!lia today                             stock buys placed today (ET)
!lia positions                         stock holdings
!lia own                               alias for positions
!lia opt                               open option positions (+ short ids)
!lia buy  opt TICKER EXP STRIKE call|put QTY [LIMIT]
!lia sell opt <id> [QTY] [LIMIT]       close option by id from !lia opt
!lia open spread TICKER EXP S/L call|put QTY [CREDIT]
!lia close spread <id1> <id2> [QTY] …  close both SPY spread legs
!lia close status                      pending / recent spread closes
!lia close cancel <job>                stop a pending spread close
!lia watch TICKER 1m|1%|-1%|759        price prints / alerts
!lia watch                             list price watches
!lia watch close <id>                  stop a price watch
```
Also: `!spy watch` lists watches · `!spy watch close <id>` stops one ·
`!spy watch 1m` is shorthand for `!lia watch SPY 1m`.
Examples:
`!lia buy AAPL 1`
`!lia buy opt SPY 2026-09-05 550 call 1`
`!lia buy opt SPY 0dte 755 call 1`
`!lia buy opt SPY 0dte atm call 1`
`!lia sell opt a1b2c3`
`!lia open spread SPY 0dte 776/778 call 1`
`!lia open spread SPY 0dte 770/768 put 1 0.35`
`!lia close spread a1b2c3 d4e5f6 at 2pm`
`!lia close spread a1b2c3 d4e5f6 in 1h`
`!lia close spread a1b2c3 d4e5f6 if spy >= 650`
`!lia close spread a1b2c3 d4e5f6 when spy hits 650`
`!lia close spread a1b2c3 d4e5f6 now`
`!lia close status`
`!lia watch SPY 1m`
`!lia watch SPY 1%`
`!lia watch SPY -1%`
`!lia watch SPY 759`
`!spy watch`
`!spy watch close 3`

EXP can be `YYYY-MM-DD`, `0dte`, `1dte`, … (Nth upcoming listed expiration).
STRIKE can be a number or `atm` (closest listed strike to spot).
Stock qty may be fractional. Option qty is whole contracts.
Buy/sell place **real orders**. Option orders are **limit** (default: mark).
`open spread` sells SHORT/buys LONG as one credit-spread order (default
credit: short bid − long ask). Calls: short below long (bear call).
Puts: short above long (bull put).
`close spread` is **SPY options only**. Ids come from `!lia opt` (order of
the two legs does not matter). Trigger: clock (`at 2pm` / `at 14:00`),
delay (`in 1h` / `in 30m`), SPY print (`if spy >= 650` / `when spy hits
650`), or `now`.
`watch` prints last + daily $/% change. Interval (`1m`) also shows change
since the last print. `%` and price watches fire once when hit.
"""

# Short-id → enriched option position, refreshed by ``!lia opt``.
_OPT_POS_BY_ID: dict[str, dict[str, Any]] = {}

# Discord hard limit is 2000; leave room for reply chrome.
_DISCORD_SAFE_LEN = 1800


def _ensure_rh_login() -> bool:
    try:
        from julia.rh_auth import ensure_robinhood_login

        return ensure_robinhood_login()
    except Exception:  # noqa: BLE001
        return False


def _trader_allowed(user_id: int) -> bool:
    raw = (os.getenv("LIA_DISCORD_ALLOWLIST") or "").strip()
    if not raw:
        return True
    allowed = {p.strip() for p in raw.split(",") if p.strip()}
    return str(user_id) in allowed


def _parse_qty(raw: str) -> float:
    qty = float(raw)
    if qty <= 0:
        raise ValueError("quantity must be > 0")
    # Robinhood fractional shares support up to 6 decimal places.
    return round(qty, 6)


def _fnum(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return "—"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    text = f"{v:,.{digits}f}".rstrip("0").rstrip(".")
    return text or "0"


def _clip(text: str) -> str:
    if len(text) <= _DISCORD_SAFE_LEN:
        return text
    return text[: _DISCORD_SAFE_LEN - 20] + "\n… _(truncated)_"


def _today_et() -> date:
    return datetime.now(ET).date()


def _order_created_date(order: dict[str, Any]) -> Optional[date]:
    raw = order.get("created_at") or order.get("last_transaction_at") or ""
    if not raw:
        return None
    try:
        # RH timestamps look like 2026-08-27T14:32:01.123456Z
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return dt.astimezone(ET).date()
    except ValueError:
        return None


def _symbol_for_order(order: dict[str, Any]) -> str:
    sym = (order.get("symbol") or "").upper()
    if sym:
        return sym
    # Fall back: resolve instrument URL → ticker.
    url = order.get("instrument")
    if not url:
        return "?"
    try:
        import robin_stocks.robinhood as rh

        resolved = rh.stocks.get_symbol_by_url(url)
        return (resolved or "?").upper()
    except Exception:  # noqa: BLE001
        return "?"


def _format_order(side: str, symbol: str, qty: float, result: Any) -> str:
    if result is None:
        return f"❌ {side} **{symbol}** x `{qty}` — no response from Robinhood."
    if isinstance(result, list):
        # Some robin_stocks paths return [None] / error strings on failure.
        if not result or result[0] is None:
            return f"❌ {side} **{symbol}** x `{qty}` — order rejected (empty response)."
        result = result[0]
    if isinstance(result, str):
        return f"❌ {side} **{symbol}** x `{qty}` — `{result}`"
    if not isinstance(result, dict):
        return f"❌ {side} **{symbol}** x `{qty}` — unexpected response: `{result!r}`"

    # robin_stocks sometimes returns {"detail": "..."} on API errors.
    if result.get("detail") and not result.get("id"):
        return f"❌ {side} **{symbol}** x `{qty}` — `{result['detail']}`"

    oid = result.get("id") or result.get("order_id") or "—"
    state = result.get("state") or result.get("status") or "—"
    filled = result.get("cumulative_quantity") or result.get("quantity") or qty
    price = result.get("average_price") or result.get("price")
    if price is None:
        tn = result.get("total_notional")
        price = tn.get("amount") if isinstance(tn, dict) else tn
    lines = [
        f"✅ **{side.upper()}** `{symbol}` x `{qty}` submitted",
        f"Order id: `{oid}`",
        f"State: `{state}` · filled qty: `{filled}`",
    ]
    if price not in (None, ""):
        lines.append(f"Price / notional: `{price}`")
    return "\n".join(lines)


def _cmd_buy(symbol: str, qty: float) -> str:
    if not _ensure_rh_login():
        return "Robinhood login failed — check `RH_USERNAME` / `RH_PASSWORD`."
    import robin_stocks.robinhood as rh

    result = rh.orders.order_buy_fractional_by_quantity(symbol, qty)
    return _format_order("buy", symbol, qty, result)


def _cmd_sell(symbol: str, qty: float) -> str:
    if not _ensure_rh_login():
        return "Robinhood login failed — check `RH_USERNAME` / `RH_PASSWORD`."
    import robin_stocks.robinhood as rh

    result = rh.orders.order_sell_fractional_by_quantity(symbol, qty)
    return _format_order("sell", symbol, qty, result)


def _cmd_today() -> str:
    """Buys placed today (America/New_York calendar day)."""
    if not _ensure_rh_login():
        return "Robinhood login failed — check `RH_USERNAME` / `RH_PASSWORD`."
    import robin_stocks.robinhood as rh

    today = _today_et()
    start = today.isoformat()
    try:
        orders = rh.orders.get_all_stock_orders(start_date=start) or []
    except TypeError:
        # Older robin_stocks without start_date — pull all and filter.
        orders = rh.orders.get_all_stock_orders() or []
    except Exception as e:  # noqa: BLE001
        return f"Failed to load orders: `{type(e).__name__}: {e}`"

    if isinstance(orders, dict):
        orders = [orders]

    buys: list[dict[str, Any]] = []
    sells_today = 0
    for order in orders:
        if not isinstance(order, dict):
            continue
        created = _order_created_date(order)
        if created != today:
            continue
        side = (order.get("side") or "").lower()
        if side == "sell":
            sells_today += 1
            continue
        if side != "buy":
            continue
        buys.append(order)

    if not buys:
        extra = (
            f" ({sells_today} sell(s) today)" if sells_today else ""
        )
        return f"No **buys** today ({today.isoformat()} ET){extra}."

    # Newest first.
    buys.sort(
        key=lambda o: o.get("created_at") or "",
        reverse=True,
    )

    lines = [f"**Buys today** ({today.isoformat()} ET) · {len(buys)}"]
    for order in buys[:25]:
        sym = _symbol_for_order(order)
        qty = order.get("cumulative_quantity") or order.get("quantity") or "?"
        state = order.get("state") or "?"
        px = order.get("average_price") or order.get("price") or "—"
        created = order.get("created_at") or ""
        hhmm = ""
        try:
            dt = datetime.fromisoformat(created.replace("Z", "+00:00")).astimezone(ET)
            hhmm = dt.strftime("%H:%M")
        except ValueError:
            pass
        lines.append(
            f"`{hhmm or '??:??'}` **{sym}** x `{_fnum(qty)}` @ `{_fnum(px)}` · `{state}`"
        )
    if len(buys) > 25:
        lines.append(f"_…and {len(buys) - 25} more_")
    if sells_today:
        lines.append(f"_Also {sells_today} sell order(s) today._")
    return _clip("\n".join(lines))


def _cmd_positions() -> str:
    """Open stock holdings — what you can sell."""
    if not _ensure_rh_login():
        return "Robinhood login failed — check `RH_USERNAME` / `RH_PASSWORD`."
    import robin_stocks.robinhood as rh

    try:
        holdings = rh.account.build_holdings() or {}
    except Exception as e:  # noqa: BLE001
        return f"Failed to load holdings: `{type(e).__name__}: {e}`"

    if not isinstance(holdings, dict) or not holdings:
        return "No open stock positions."

    rows: list[tuple[float, str, dict[str, Any]]] = []
    for symbol, info in holdings.items():
        if not isinstance(info, dict):
            continue
        try:
            qty = float(info.get("quantity") or 0)
        except (TypeError, ValueError):
            qty = 0.0
        if qty <= 0:
            continue
        try:
            equity = float(info.get("equity") or 0)
        except (TypeError, ValueError):
            equity = 0.0
        rows.append((equity, symbol.upper(), info))

    if not rows:
        return "No open stock positions."

    rows.sort(key=lambda r: r[0], reverse=True)
    lines = [f"**Positions** · {len(rows)} holdings (sorted by equity)"]
    for equity, symbol, info in rows[:40]:
        qty = info.get("quantity")
        avg = info.get("average_buy_price")
        price = info.get("price")
        pct = info.get("percent_change")
        name = info.get("name") or ""
        pct_s = ""
        if pct not in (None, ""):
            try:
                p = float(pct)
                sign = "+" if p >= 0 else ""
                pct_s = f" · `{sign}{p:.2f}%`"
            except (TypeError, ValueError):
                pct_s = f" · `{pct}`"
        label = f"  {name}" if name and name.upper() != symbol else ""
        lines.append(
            f"`{symbol:<6}` qty `{_fnum(qty)}` · "
            f"avg `{_fnum(avg)}` · last `{_fnum(price)}` · "
            f"eq `${_fnum(equity, 2)}`{pct_s}{label}"
        )
    if len(rows) > 40:
        lines.append(f"_…and {len(rows) - 40} more_")
    lines.append("_Sell with_ `!lia sell TICKER QTY`")
    return _clip("\n".join(lines))


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

def _option_instrument_id(pos: dict[str, Any]) -> Optional[str]:
    if pos.get("option_id"):
        return str(pos["option_id"])
    url = pos.get("option") or ""
    if isinstance(url, str) and url:
        # https://api.robinhood.com/options/instruments/<uuid>/
        parts = url.rstrip("/").split("/")
        if parts and parts[-1]:
            return parts[-1]
    return None


def _short_opt_id(instrument_id: str) -> str:
    """Stable short id from the option instrument UUID (first 6 hex chars)."""
    hex_id = re.sub(r"[^0-9a-fA-F]", "", instrument_id).lower()
    return (hex_id[:6] if hex_id else instrument_id[:6]).lower()


def _enrich_option_position(pos: dict[str, Any]) -> dict[str, Any]:
    """Attach symbol / strike / expiry / type / short id for display + sell."""
    import robin_stocks.robinhood as rh

    out = dict(pos)
    instrument_id = _option_instrument_id(pos)
    out["_instrument_id"] = instrument_id
    out["_short_id"] = _short_opt_id(instrument_id) if instrument_id else "?"

    if instrument_id:
        try:
            inst = rh.options.get_option_instrument_data_by_id(instrument_id) or {}
            if isinstance(inst, list):
                inst = inst[0] if inst else {}
            if isinstance(inst, dict):
                out["_symbol"] = (
                    inst.get("chain_symbol")
                    or out.get("chain_symbol")
                    or "?"
                ).upper()
                out["_strike"] = float(inst.get("strike_price") or 0)
                out["_expiration"] = inst.get("expiration_date") or "?"
                out["_option_type"] = (inst.get("type") or "?").lower()
        except Exception:  # noqa: BLE001
            pass

    out.setdefault("_symbol", (out.get("chain_symbol") or "?").upper())
    out.setdefault("_strike", float(out.get("strike_price") or 0))
    out.setdefault("_expiration", out.get("expiration_date") or "?")
    # Don't use pos["type"] as call/put — on positions that field is often long/short.
    if out.get("_option_type") in (None, "", "?", "long", "short"):
        out["_option_type"] = "?"
    try:
        qty = float(out.get("quantity") or 0)
    except (TypeError, ValueError):
        qty = 0.0
    out["_qty"] = qty
    # Prefer explicit direction from RH when present.
    ptype = (pos.get("type") or "").lower()
    if ptype in ("long", "short"):
        out["_side"] = ptype
    else:
        out["_side"] = "long" if qty >= 0 else "short"
    return out


def _load_open_option_positions() -> list[dict[str, Any]]:
    """Fetch + enrich open option positions; refresh short-id map."""
    global _OPT_POS_BY_ID
    import robin_stocks.robinhood as rh

    raw = rh.options.get_open_option_positions() or []
    if isinstance(raw, dict):
        raw = [raw]
    enriched: list[dict[str, Any]] = []
    for pos in raw:
        if not isinstance(pos, dict):
            continue
        try:
            qty = float(pos.get("quantity") or 0)
        except (TypeError, ValueError):
            qty = 0.0
        if qty == 0:
            continue
        enriched.append(_enrich_option_position(pos))

    # Stable sort for consistent listing.
    enriched.sort(
        key=lambda p: (
            p.get("_symbol") or "",
            p.get("_expiration") or "",
            float(p.get("_strike") or 0),
            p.get("_option_type") or "",
        )
    )

    by_id: dict[str, dict[str, Any]] = {}
    for p in enriched:
        sid = p.get("_short_id") or "?"
        # Disambiguate rare collisions with a numeric suffix.
        key = sid
        n = 2
        while key in by_id:
            key = f"{sid}{n}"
            n += 1
        p["_short_id"] = key
        by_id[key] = p
    _OPT_POS_BY_ID = by_id
    return enriched


def _lookup_opt_position(short_id: str) -> Optional[dict[str, Any]]:
    sid = short_id.lower().strip()
    if sid in _OPT_POS_BY_ID:
        return _OPT_POS_BY_ID[sid]
    # Rebuild map (bot may have restarted since the user listed positions).
    _load_open_option_positions()
    if sid in _OPT_POS_BY_ID:
        return _OPT_POS_BY_ID[sid]
    # Prefix match against full instrument ids.
    for key, pos in _OPT_POS_BY_ID.items():
        inst = (pos.get("_instrument_id") or "").replace("-", "").lower()
        if inst.startswith(sid) or key.startswith(sid):
            return pos
    return None


def _option_quote_prices(
    symbol: str, expiration: str, strike: float, option_type: str
) -> dict[str, Optional[float]]:
    """Return bid/ask/mark for a contract (best-effort)."""
    import robin_stocks.robinhood as rh

    try:
        rows = rh.options.find_options_by_expiration_and_strike(
            symbol,
            expiration,
            strike,
            optionType=option_type.lower(),
        ) or []
    except Exception:  # noqa: BLE001
        rows = []
    if isinstance(rows, dict):
        rows = [rows]
    row = rows[0] if rows else {}
    if not isinstance(row, dict):
        row = {}

    def _pf(key: str) -> Optional[float]:
        v = row.get(key)
        try:
            return float(v) if v not in (None, "") else None
        except (TypeError, ValueError):
            return None

    mark = _pf("mark_price") or _pf("adjusted_mark_price")
    bid = _pf("bid_price")
    ask = _pf("ask_price")
    # Fallback via instrument market data if we have an id.
    if mark is None and row.get("id"):
        try:
            md = rh.options.get_option_market_data_by_id(row["id"]) or {}
            if isinstance(md, list):
                md = md[0] if md else {}
            if isinstance(md, dict):
                for k in ("adjusted_mark_price", "mark_price"):
                    if md.get(k) not in (None, ""):
                        mark = float(md[k])
                        break
                if bid is None and md.get("bid_price") not in (None, ""):
                    bid = float(md["bid_price"])
                if ask is None and md.get("ask_price") not in (None, ""):
                    ask = float(md["ask_price"])
        except Exception:  # noqa: BLE001
            pass
    return {"bid": bid, "ask": ask, "mark": mark}


def _cmd_opt_positions() -> str:
    if not _ensure_rh_login():
        return "Robinhood login failed — check `RH_USERNAME` / `RH_PASSWORD`."
    try:
        positions = _load_open_option_positions()
    except Exception as e:  # noqa: BLE001
        return f"Failed to load option positions: `{type(e).__name__}: {e}`"

    if not positions:
        return "No open option positions."

    lines = [
        f"**Option positions** · {len(positions)}",
        "_Close one leg with_ `!lia sell opt <id>` · "
        "_both spread legs with_ `!lia close spread <id1> <id2> at 2pm`",
    ]
    for p in positions[:40]:
        sid = p.get("_short_id") or "?"
        sym = p.get("_symbol") or "?"
        exp = p.get("_expiration") or "?"
        strike = p.get("_strike") or 0
        otype = (p.get("_option_type") or "?").upper()
        qty = abs(float(p.get("_qty") or 0))
        side = p.get("_side") or "?"
        avg = p.get("average_price")
        lines.append(
            f"`{sid}` **{sym}** {exp} `{_fnum(strike, 2)}` {otype} "
            f"· {side} x `{_fnum(qty, 0)}` · avg `{_fnum(avg)}`"
        )
    if len(positions) > 40:
        lines.append(f"_…and {len(positions) - 40} more_")
    return _clip("\n".join(lines))


def _cmd_buy_opt(
    symbol: str,
    expiration: str,
    strike: float,
    option_type: str,
    qty: int,
    limit: Optional[float],
) -> str:
    if not _ensure_rh_login():
        return "Robinhood login failed — check `RH_USERNAME` / `RH_PASSWORD`."
    import robin_stocks.robinhood as rh

    option_type = option_type.lower()
    if option_type not in ("call", "put"):
        return "option type must be `call` or `put`."
    if qty <= 0:
        return "quantity must be a positive whole number of contracts."

    if limit is None:
        q = _option_quote_prices(symbol, expiration, strike, option_type)
        limit = q.get("ask") or q.get("mark") or q.get("bid")
        if limit is None:
            return (
                f"No quote for {symbol} {expiration} {strike} {option_type} — "
                "pass an explicit LIMIT price."
            )
        # Pad a tick so the debit limit is more likely to fill.
        limit = round(float(limit) + 0.01, 2)
    else:
        limit = round(float(limit), 2)

    result = rh.orders.order_buy_option_limit(
        positionEffect="open",
        creditOrDebit="debit",
        price=limit,
        symbol=symbol,
        quantity=qty,
        expirationDate=expiration,
        strike=strike,
        optionType=option_type,
        timeInForce="gtc",
    )
    label = f"{symbol} {expiration} {strike:g} {option_type.upper()}"
    return _format_order("buy-opt", label, float(qty), result) + f"\nLimit: `${limit}`"


def _cmd_sell_opt(
    short_id: str,
    qty: Optional[int],
    limit: Optional[float],
) -> str:
    """Close an option position referenced by the short id from ``!lia opt``."""
    if not _ensure_rh_login():
        return "Robinhood login failed — check `RH_USERNAME` / `RH_PASSWORD`."
    import robin_stocks.robinhood as rh

    pos = _lookup_opt_position(short_id)
    if not pos:
        return (
            f"Unknown option id `{short_id}`. "
            "Run `!lia opt` to list positions and ids."
        )

    symbol = pos.get("_symbol") or "?"
    expiration = pos.get("_expiration") or "?"
    strike = float(pos.get("_strike") or 0)
    option_type = (pos.get("_option_type") or "").lower()
    held = abs(float(pos.get("_qty") or 0))
    side = pos.get("_side") or "long"

    if held <= 0:
        return f"Position `{short_id}` has zero quantity."
    if option_type not in ("call", "put"):
        return f"Position `{short_id}` missing call/put type."

    close_qty = int(held) if qty is None else int(qty)
    if close_qty <= 0:
        return "quantity must be a positive whole number of contracts."
    if close_qty > int(held):
        return f"Only `{_fnum(held, 0)}` contract(s) available (asked `{close_qty}`)."

    if limit is None:
        q = _option_quote_prices(symbol, expiration, strike, option_type)
        if side == "long":
            # Sell to close → credit; use bid/mark.
            limit = q.get("bid") or q.get("mark") or q.get("ask")
        else:
            # Buy to close short → debit; use ask/mark.
            limit = q.get("ask") or q.get("mark") or q.get("bid")
        if limit is None:
            return (
                f"No quote for `{short_id}` — pass an explicit LIMIT price: "
                f"`!lia sell opt {short_id} {close_qty} LIMIT`"
            )
        limit = round(float(limit), 2)
        if side == "long":
            limit = max(0.01, round(limit - 0.01, 2))
        else:
            limit = round(limit + 0.01, 2)
    else:
        limit = round(float(limit), 2)

    label = (
        f"{short_id} {symbol} {expiration} {strike:g} {option_type.upper()} "
        f"({side})"
    )

    if side == "long":
        result = rh.orders.order_sell_option_limit(
            positionEffect="close",
            creditOrDebit="credit",
            price=limit,
            symbol=symbol,
            quantity=close_qty,
            expirationDate=expiration,
            strike=strike,
            optionType=option_type,
            timeInForce="gtc",
        )
        return (
            _format_order("sell-opt", label, float(close_qty), result)
            + f"\nLimit: `${limit}` (sell to close)"
        )

    # Short position → buy to close.
    result = rh.orders.order_buy_option_limit(
        positionEffect="close",
        creditOrDebit="debit",
        price=limit,
        symbol=symbol,
        quantity=close_qty,
        expirationDate=expiration,
        strike=strike,
        optionType=option_type,
        timeInForce="gtc",
    )
    return (
        _format_order("buy-to-close", label, float(close_qty), result)
        + f"\nLimit: `${limit}` (buy to close short)"
    )


def _cmd_open_spread(req: spread_open.SpreadOpenRequest) -> str:
    """Open a credit spread: sell the short leg, buy the long hedge."""
    if not _ensure_rh_login():
        return "Robinhood login failed — check `RH_USERNAME` / `RH_PASSWORD`."
    import robin_stocks.robinhood as rh

    try:
        expiration, exp_note = _resolve_expiration(req.symbol, req.exp_token)
    except ValueError as e:
        return f"❌ {e}"

    credit = req.credit
    quote_note = ""
    if credit is None:
        short_q = _option_quote_prices(
            req.symbol, expiration, req.short_strike, req.option_type
        )
        long_q = _option_quote_prices(
            req.symbol, expiration, req.long_strike, req.option_type
        )
        credit = spread_open.default_credit(short_q, long_q)
        if credit is None:
            return (
                f"No quotes for {req.symbol} {expiration} "
                f"{req.short_strike:g}/{req.long_strike:g} "
                f"{req.option_type} — pass an explicit CREDIT:\n"
                "`!lia open spread TICKER EXP SHORT/LONG call|put QTY CREDIT`"
            )
        quote_note = (
            f"\nQuotes: short bid `{_fnum(short_q.get('bid'))}` / "
            f"long ask `{_fnum(long_q.get('ask'))}` → natural credit"
        )
    credit = round(float(credit), 2)
    if credit >= req.width:
        return (
            f"❌ Credit `${credit:.2f}` ≥ spread width `${req.width:g}` — "
            "that can't fill (max value of the spread is its width). "
            "Check the strikes / credit."
        )

    legs = [
        {
            "expirationDate": expiration,
            "strike": req.short_strike,
            "optionType": req.option_type,
            "effect": "open",
            "action": "sell",
        },
        {
            "expirationDate": expiration,
            "strike": req.long_strike,
            "optionType": req.option_type,
            "effect": "open",
            "action": "buy",
        },
    ]
    result = rh.orders.order_option_credit_spread(
        price=credit,
        symbol=req.symbol,
        quantity=req.qty,
        spread=legs,
        timeInForce="gtc",
    )

    s = spread_open.spread_summary(req, credit)
    label = (
        f"{req.symbol} {expiration} "
        f"-{req.short_strike:g}/+{req.long_strike:g} "
        f"{req.option_type.upper()} ({req.kind})"
    )
    return (
        f"Resolved `{req.exp_token}` → **{exp_note}**\n"
        + _format_order("open-spread", label, float(req.qty), result)
        + f"\nCredit: `${credit:.2f}`{quote_note}"
        + (
            f"\nWidth `${s['width']:g}` · max gain `${s['max_gain']:,.0f}` · "
            f"max loss `${s['max_loss']:,.0f}` · breakeven `{s['breakeven']:g}`"
        )
        + "\n_Legs show in_ `!lia opt` _after fill — close with_ "
        "`!lia close spread <id1> <id2> …`"
    )


# ---------------------------------------------------------------------------
# Conditional SPY spread close
# ---------------------------------------------------------------------------

_SPREAD_WATCH_SECONDS = 15


def _spy_spot() -> Optional[float]:
    """Best-effort SPY last: poller ticks first, then Robinhood."""
    try:
        from julia.daily_moves_store import latest_tick

        tick = latest_tick("SPY")
        if tick and tick.get("price"):
            return float(tick["price"])
    except Exception:  # noqa: BLE001
        pass
    if not _ensure_rh_login():
        return None
    try:
        import robin_stocks.robinhood as rh

        spot_list = rh.stocks.get_latest_price(
            "SPY", priceType=None, includeExtendedHours=True
        )
        if spot_list and spot_list[0] is not None:
            return float(spot_list[0])
    except Exception:  # noqa: BLE001
        return None
    return None


def _session_ref_spot(symbol: str) -> Optional[float]:
    try:
        from julia.daily_moves_store import list_sessions

        today = _today_et().isoformat()
        for row in list_sessions(symbol):
            if str(row["session_date"]) == today:
                return float(row["ref_spot"])
    except Exception:  # noqa: BLE001
        return None
    return None


def _rh_previous_close(symbol: str) -> Optional[float]:
    try:
        import robin_stocks.robinhood as rh

        quotes = rh.stocks.get_quotes(symbol) or []
        if isinstance(quotes, dict):
            quotes = [quotes]
        row = quotes[0] if quotes else {}
        if not isinstance(row, dict):
            return None
        for key in ("previous_close", "adjusted_previous_close", "last_close"):
            if row.get(key) not in (None, ""):
                return float(row[key])
    except Exception:  # noqa: BLE001
        return None
    return None


def _stock_quote(symbol: str) -> Optional[dict[str, Any]]:
    """Last price + daily change. Poller ticks first, then Robinhood."""
    symbol = symbol.upper()
    price: Optional[float] = None
    ts = datetime.now(ET)
    try:
        from julia.daily_moves_store import latest_tick

        tick = latest_tick(symbol)
        if tick and tick.get("price"):
            price = float(tick["price"])
            ts = tick.get("ts") or ts
    except Exception:  # noqa: BLE001
        pass
    if price is None:
        if not _ensure_rh_login():
            return None
        try:
            import robin_stocks.robinhood as rh

            spot_list = rh.stocks.get_latest_price(
                symbol, priceType=None, includeExtendedHours=True
            )
            if spot_list and spot_list[0] is not None:
                price = float(spot_list[0])
        except Exception:  # noqa: BLE001
            return None
    if price is None:
        return None
    ref = _session_ref_spot(symbol)
    if ref is None and _ensure_rh_login():
        ref = _rh_previous_close(symbol)
    daily_chg = (price - ref) if ref else None
    daily_pct = (daily_chg / ref * 100.0) if ref and ref != 0 else None
    return {
        "symbol": symbol,
        "price": price,
        "ref": ref,
        "daily_chg": daily_chg,
        "daily_pct": daily_pct,
        "ts": ts,
    }


def _cmd_watch_status() -> str:
    jobs = price_watch.list_watches(
        statuses=(price_watch.STATUS_PENDING,),
        limit=40,
    )
    if not jobs:
        recent = price_watch.list_watches(limit=8)
        if not recent:
            return price_watch.format_watch_status([])
        lines = [
            "No **active** price watches. Recent:",
            *(price_watch.format_watch_line(j) for j in recent),
            "_New:_ `!lia watch SPY 1m`",
        ]
        return _clip("\n".join(lines))
    return _clip(price_watch.format_watch_status(jobs))


def _cmd_watch_close(raw_id: str) -> str:
    token = raw_id.strip().lstrip("#")
    if not token.isdigit():
        return "Usage: `!lia watch close <id>` or `!spy watch close <id>`."
    watch_id = int(token)
    job = price_watch.cancel_watch(watch_id)
    if job is None:
        return f"No price watch `#{watch_id}`."
    if job["status"] == price_watch.STATUS_CANCELLED:
        return f"Closed `{price_watch.format_watch_line(job)}`."
    return (
        f"Watch `#{watch_id}` is `{job['status']}` — "
        "only **pending** watches can be closed."
    )


def _schedule_price_watch(
    args: list[str],
    *,
    created_by: str,
    channel_id: Optional[str],
) -> str:
    try:
        req = price_watch.parse_watch_args(args)
    except ValueError as e:
        return str(e)

    quote = _stock_quote(req.symbol)
    if quote is None:
        return (
            f"Could not read a last price for `{req.symbol}`. "
            "Is the price poller running, or RH login available?"
        )

    spec = req.spec
    fire_now = False
    if spec.kind == price_watch.KIND_PRICE:
        spec = price_watch.resolve_price_op(spec, float(quote["price"]))
        fire_now = price_watch.price_condition_met(
            spec.price_op, spec.price_level, float(quote["price"])
        )
    elif spec.kind == price_watch.KIND_PCT:
        fire_now = price_watch.pct_condition_met(spec.pct_level, quote.get("daily_pct"))
        if quote.get("daily_pct") is None and not fire_now:
            # Still schedule — watcher will fire once daily % is known.
            pass
    elif spec.kind == price_watch.KIND_INTERVAL:
        fire_now = True  # first print immediately, then every interval

    now = datetime.now(ET)
    if fire_now and spec.kind != price_watch.KIND_INTERVAL:
        job = price_watch.insert_watch(
            symbol=req.symbol,
            spec=spec,
            created_by=created_by,
            channel_id=channel_id,
            last_price=float(quote["price"]),
            last_printed_at=now,
            print_count=1,
            status=price_watch.STATUS_FIRED,
        )
        return (
            f"Watch `#{job['id']}` already there — printing now.\n"
            + price_watch.format_quote_message(
                job, {**quote, "prev_print_price": None}
            )
        )

    job = price_watch.insert_watch(
        symbol=req.symbol,
        spec=spec,
        created_by=created_by,
        channel_id=channel_id,
        last_price=float(quote["price"]) if spec.kind == price_watch.KIND_INTERVAL else None,
        last_printed_at=now if spec.kind == price_watch.KIND_INTERVAL else None,
        print_count=1 if spec.kind == price_watch.KIND_INTERVAL else 0,
        status=price_watch.STATUS_PENDING,
    )
    if spec.kind == price_watch.KIND_INTERVAL:
        return (
            f"Watching `#{job['id']}` **{req.symbol}** {spec.text}. "
            f"`!spy watch close {job['id']}` to stop.\n"
            + price_watch.format_quote_message(
                job, {**quote, "prev_print_price": None}
            )
        )
    return (
        f"Watching `#{job['id']}` **{req.symbol}** {spec.text}. "
        f"`!spy watch close {job['id']}` to stop.\n"
        f"Now `{req.symbol}` `${float(quote['price']):,.2f}` · "
        f"day `{price_watch.signed_money(quote.get('daily_chg'))}` "
        f"(`{price_watch.signed_pct(quote.get('daily_pct'))}`)"
    )


_WATCH_HELP = (
    "Price watches:\n"
    "`!lia watch SPY 1m` — print last + day change every minute "
    "(also % since last print)\n"
    "`!lia watch SPY 1%` / `!lia watch SPY -1%` — print when today's % hits\n"
    "`!lia watch SPY 759` — print when last hits 759\n"
    "`!lia watch` or `!spy watch` — list with ids\n"
    "`!lia watch close <id>` or `!spy watch close <id>` — stop"
)


def _handle_watch_args(
    args: list[str],
    *,
    created_by: str,
    channel_id: Optional[str],
    default_symbol: Optional[str] = None,
) -> str:
    if not args or args[0].lower() in ("status", "list", "jobs"):
        return _cmd_watch_status()
    if args[0].lower() in ("help", "?"):
        return _WATCH_HELP
    if args[0].lower() in ("close", "cancel", "stop", "rm", "delete"):
        if len(args) < 2:
            return "Usage: `!lia watch close <id>` or `!spy watch close <id>`."
        return _cmd_watch_close(args[1])
    watch_args = list(args)
    if default_symbol:
        try:
            price_watch.parse_watch_spec(" ".join(watch_args))
        except ValueError:
            pass
        else:
            watch_args = [default_symbol, *watch_args]
    return _schedule_price_watch(
        watch_args,
        created_by=created_by,
        channel_id=channel_id,
    )


def _fire_spread_job(job: dict[str, Any]) -> str:
    """Buy-to-close the short leg, then sell-to-close the long leg."""
    qty = int(job.get("qty") or 0)
    short_id = str(job.get("short_id") or "")
    long_id = str(job.get("long_id") or "")
    header = (
        f"**Spread close `#{job.get('id')}` fired** · "
        f"{job.get('trigger_text') or '?'}"
    )
    if qty <= 0 or not short_id or not long_id:
        err = "Job missing qty or leg ids."
        spread_close.mark_job(
            int(job["id"]),
            status=spread_close.STATUS_ERROR,
            last_error=err,
            result_text=err,
        )
        return f"{header}\n❌ {err}"

    short_result = _cmd_sell_opt(short_id, qty, None)
    long_result = _cmd_sell_opt(long_id, qty, None)
    short_ok = short_result.startswith("✅")
    long_ok = long_result.startswith("✅")
    body = (
        f"**Short `{short_id}` (buy to close)**\n{short_result}\n"
        f"**Long `{long_id}` (sell to close)**\n{long_result}"
    )
    if short_ok and long_ok:
        status = spread_close.STATUS_FIRED
        last_error = None
    elif short_ok or long_ok:
        status = spread_close.STATUS_PARTIAL
        last_error = "one leg failed"
    else:
        status = spread_close.STATUS_ERROR
        last_error = "both legs failed"
    spread_close.mark_job(
        int(job["id"]),
        status=status,
        result_text=body,
        last_error=last_error,
    )
    return f"{header}\n{body}"


def _cmd_close_status() -> str:
    jobs = spread_close.list_jobs(
        statuses=(
            spread_close.STATUS_PENDING,
            spread_close.STATUS_FIRING,
        ),
        limit=25,
    )
    if not jobs:
        recent = spread_close.list_jobs(limit=8)
        if not recent:
            return spread_close.format_status([])
        lines = [
            "No **pending** spread closes. Recent:",
            *(spread_close.format_job_line(j) for j in recent),
            "_New watch:_ `!lia close spread <id1> <id2> at 2pm`",
        ]
        return _clip("\n".join(lines))
    return _clip(spread_close.format_status(jobs))


def _cmd_close_cancel(raw_id: str) -> str:
    token = raw_id.strip().lstrip("#")
    if not token.isdigit():
        return "Usage: `!lia close cancel <job>` — job ids from `!lia close status`."
    job_id = int(token)
    job = spread_close.cancel_job(job_id)
    if job is None:
        return f"No spread-close job `#{job_id}`."
    if job["status"] == spread_close.STATUS_CANCELLED:
        return f"Cancelled `{spread_close.format_job_line(job)}`."
    return (
        f"Job `#{job_id}` is `{job['status']}` — only **pending** watches can be cancelled."
    )


def _schedule_spread_close(
    args: list[str],
    *,
    created_by: str,
    channel_id: Optional[str],
) -> str:
    if not _ensure_rh_login():
        return "Robinhood login failed — check `RH_USERNAME` / `RH_PASSWORD`."
    try:
        req = spread_close.parse_close_spread_args(args)
    except ValueError as e:
        return str(e)

    pos1 = _lookup_opt_position(req.id1)
    pos2 = _lookup_opt_position(req.id2)
    if not pos1:
        return f"Unknown option id `{req.id1}`. Run `!lia opt` to list positions and ids."
    if not pos2:
        return f"Unknown option id `{req.id2}`. Run `!lia opt` to list positions and ids."
    if req.id1 == req.id2 or pos1 is pos2:
        return "Need two different legs."

    try:
        legs = spread_close.validate_spread_legs(pos1, pos2, req.qty)
    except ValueError as e:
        return str(e)

    trigger = req.trigger
    if trigger.hits_unresolved:
        spot = _spy_spot()
        if spot is None:
            return (
                "Could not read SPY to resolve `hits`. "
                "Use `if spy >= LEVEL` or `if spy <= LEVEL` instead."
            )
        trigger = spread_close.resolve_hits_trigger(trigger, spot)
        if spread_close.price_condition_met(
            trigger.price_op, trigger.price_level, spot
        ):
            # Already at the level — fire now rather than sitting pending.
            trigger = spread_close.Trigger("now", trigger.text + " · already there", None)

    if trigger.kind == "now":
        job = spread_close.insert_job(
            legs=legs,
            trigger=trigger,
            created_by=created_by,
            channel_id=channel_id,
            status=spread_close.STATUS_FIRING,
        )
        return (
            f"Closing spread `#{job['id']}` now — "
            f"buy-to-close `{legs.short_label}`, "
            f"sell-to-close `{legs.long_label}` x `{legs.qty}`.\n"
            + _fire_spread_job(job)
        )

    job = spread_close.insert_job(
        legs=legs,
        trigger=trigger,
        created_by=created_by,
        channel_id=channel_id,
    )
    return spread_close.format_scheduled(job, legs)


async def _notify_channel(client: discord.Client, channel_id: Optional[str], text: str) -> None:
    if not channel_id:
        print(f"[discord] spread close (no channel): {text[:200]}")
        return
    try:
        channel = client.get_channel(int(channel_id))
        if channel is None:
            channel = await client.fetch_channel(int(channel_id))
        if channel is not None:
            await channel.send(_clip(text))
    except Exception as e:  # noqa: BLE001
        print(f"[discord] spread-close notify failed: {e!r}")


async def _tick_spread_closes(client: discord.Client) -> None:
    pending = spread_close.list_pending()
    if not pending:
        return
    need_price = any(j.get("trigger_kind") == "price" for j in pending)
    spy = await asyncio.to_thread(_spy_spot) if need_price else None
    due = await asyncio.to_thread(spread_close.claim_due_jobs, None, spy)
    for job in due:
        result = await asyncio.to_thread(_fire_spread_job, job)
        await _notify_channel(client, job.get("channel_id"), result)


async def _tick_price_watches(client: discord.Client) -> None:
    pending = price_watch.list_pending()
    if not pending:
        return
    symbols = sorted(
        {
            str(j.get("symbol") or "").upper()
            for j in pending
            if j.get("symbol")
        }
    )
    quotes: dict[str, dict[str, Any]] = {}
    for symbol in symbols:
        quote = await asyncio.to_thread(_stock_quote, symbol)
        if quote:
            quotes[symbol] = quote
    due = await asyncio.to_thread(price_watch.claim_due_watches, quotes, None)
    for job, quote in due:
        await _notify_channel(
            client,
            job.get("channel_id"),
            price_watch.format_quote_message(job, quote),
        )


async def _watch_background(client: discord.Client) -> None:
    await client.wait_until_ready()
    print("[discord] spread-close + price-watch loop started")
    while not client.is_closed():
        try:
            await _tick_spread_closes(client)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
        try:
            await _tick_price_watches(client)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
        await asyncio.sleep(_SPREAD_WATCH_SECONDS)


def _parse_lia(content: str) -> tuple[str, list[str]]:
    """Return (subcommand, args) for a ``!lia ...`` message.

    ``!lia`` alone → ("help", []). Unknown prefix → ("", []).
    """
    parts = content.split()
    if not parts:
        return "", []
    if parts[0].lower() != "!lia":
        return "", []
    if len(parts) == 1:
        return "help", []
    return parts[1].lower(), parts[2:]


def _parse_opt_buy_args(
    args: list[str],
) -> tuple[str, str, str, str, int, Optional[float]]:
    """Parse: TICKER EXP STRIKE call|put QTY [LIMIT].

    EXP may be ``YYYY-MM-DD`` or ``Ndte`` (e.g. ``0dte``).
    STRIKE may be a number or ``atm``.
    Resolution of aliases happens later in ``_resolve_opt_buy``.
    """
    if len(args) < 5:
        raise ValueError(
            "Usage: `!lia buy opt TICKER EXP STRIKE call|put QTY [LIMIT]`\n"
            "EXP: `YYYY-MM-DD` or `0dte`/`1dte`/…  ·  "
            "STRIKE: number or `atm`"
        )
    symbol = args[0].upper().strip()
    expiration = args[1].strip().lower()
    strike_token = args[2].strip().lower()
    option_type = args[3].lower().strip()
    qty = int(float(args[4]))
    limit = float(args[5]) if len(args) >= 6 else None
    if option_type not in ("call", "put", "c", "p"):
        raise ValueError("option type must be `call` or `put`")
    if option_type == "c":
        option_type = "call"
    if option_type == "p":
        option_type = "put"

    # Light validation — full resolve needs RH login.
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", expiration) and not re.fullmatch(
        r"\d+dte", expiration
    ):
        raise ValueError(
            f"Bad EXP `{expiration}` — use `YYYY-MM-DD` or `0dte`/`1dte`/…"
        )
    if strike_token != "atm":
        try:
            float(strike_token)
        except ValueError as e:
            raise ValueError(
                f"Bad STRIKE `{strike_token}` — use a number or `atm`"
            ) from e
    if qty <= 0:
        raise ValueError("quantity must be a positive whole number of contracts")
    return symbol, expiration, strike_token, option_type, qty, limit


def _listed_expirations(symbol: str) -> list[str]:
    """Upcoming expiration dates for ``symbol`` from the RH option chain."""
    import robin_stocks.robinhood as rh

    chain = rh.options.get_chains(symbol) or {}
    if isinstance(chain, list):
        chain = chain[0] if chain else {}
    dates = list(chain.get("expiration_dates") or [])
    today = _today_et().isoformat()
    upcoming = sorted(d for d in dates if isinstance(d, str) and d >= today)
    if not upcoming:
        raise ValueError(f"No upcoming option expirations listed for `{symbol}`")
    return upcoming


def _resolve_expiration(symbol: str, exp_token: str) -> tuple[str, str]:
    """Return ``(YYYY-MM-DD, note)`` for a date or ``Ndte`` token."""
    token = exp_token.strip().lower()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", token):
        datetime.strptime(token, "%Y-%m-%d")  # validate
        return token, token

    m = re.fullmatch(r"(\d+)dte", token)
    if not m:
        raise ValueError(f"Bad EXP `{exp_token}`")
    n = int(m.group(1))
    upcoming = _listed_expirations(symbol)
    if n >= len(upcoming):
        raise ValueError(
            f"`{token}` needs upcoming[{n}] but `{symbol}` only has "
            f"{len(upcoming)} listed date(s): {', '.join(upcoming[:5])}"
            + ("…" if len(upcoming) > 5 else "")
        )
    resolved = upcoming[n]
    note = f"{token} → {resolved}"
    if n == 0 and resolved != _today_et().isoformat():
        note += " _(no same-day expiry; nearest listed)_"
    return resolved, note


def _resolve_strike(
    symbol: str,
    expiration: str,
    strike_token: str,
    option_type: str,
) -> tuple[float, str]:
    """Return ``(strike, note)`` for a numeric strike or ``atm``."""
    token = strike_token.strip().lower()
    if token != "atm":
        strike = float(token)
        return strike, f"{strike:g}"

    import robin_stocks.robinhood as rh

    spot_list = rh.stocks.get_latest_price(
        symbol, priceType=None, includeExtendedHours=True
    )
    if not spot_list or spot_list[0] is None:
        raise ValueError(f"Could not fetch spot for `{symbol}` to resolve atm")
    spot = float(spot_list[0])

    rows = rh.options.find_options_by_expiration(
        symbol, expirationDate=expiration, optionType=option_type
    ) or []
    if isinstance(rows, dict):
        rows = [rows]
    strikes: list[float] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            strikes.append(float(row["strike_price"]))
        except (KeyError, TypeError, ValueError):
            continue
    strikes = sorted(set(strikes))
    if not strikes:
        raise ValueError(
            f"No {option_type} strikes listed for `{symbol}` {expiration}"
        )
    atm = min(strikes, key=lambda s: abs(s - spot))
    return atm, f"atm → {atm:g} (spot `{spot:g}`)"


def _resolve_opt_buy(
    symbol: str,
    exp_token: str,
    strike_token: str,
    option_type: str,
    qty: int,
    limit: Optional[float],
) -> tuple[str, float, str]:
    """Resolve aliases then place the buy. Returns ``(reply, strike, expiration)``."""
    if not _ensure_rh_login():
        return "Robinhood login failed — check `RH_USERNAME` / `RH_PASSWORD`.", 0.0, ""
    try:
        expiration, exp_note = _resolve_expiration(symbol, exp_token)
        strike, strike_note = _resolve_strike(
            symbol, expiration, strike_token, option_type
        )
    except ValueError as e:
        return f"❌ {e}", 0.0, ""

    result = _cmd_buy_opt(symbol, expiration, strike, option_type, qty, limit)
    header = f"Resolved `{exp_token}`/`{strike_token}` → **{exp_note}**, **{strike_note}**"
    return f"{header}\n{result}", strike, expiration


def _parse_opt_sell_args(
    args: list[str],
) -> tuple[str, Optional[int], Optional[float]]:
    """Parse: <id> [QTY] [LIMIT]."""
    if not args:
        raise ValueError(
            "Usage: `!lia sell opt <id> [QTY] [LIMIT]` — ids from `!lia opt`"
        )
    short_id = args[0].strip()
    qty: Optional[int] = None
    limit: Optional[float] = None
    if len(args) >= 2:
        qty = int(float(args[1]))
    if len(args) >= 3:
        limit = float(args[2])
    return short_id, qty, limit


async def _handle_spy_message(msg: discord.Message, content: str) -> None:
    """``!spy watch`` list / ``!spy watch 1m`` / ``!spy watch close <id>``."""
    parts = content.split()
    rest = parts[1:]
    if rest and rest[0].lower() in ("watch", "watches"):
        rest = rest[1:]
    elif rest:
        await msg.reply(
            "SPY watches: `!spy watch` · `!spy watch 1m` · "
            "`!spy watch close <id>`"
        )
        return
    await msg.reply(
        _handle_watch_args(
            rest,
            created_by=str(msg.author.id),
            channel_id=str(msg.channel.id),
            default_symbol="SPY",
        )
    )


async def _handle_message(msg: discord.Message) -> None:
    content = (msg.content or "").strip()
    lower = content.lower()
    if lower.startswith("!spy"):
        try:
            await _handle_spy_message(msg, content)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            await msg.reply(f"Error: `{type(e).__name__}: {e}`")
        return
    if not lower.startswith("!lia"):
        # Ignore every other message (including old !price / !help).
        return

    sub, args = _parse_lia(content)
    if not sub:
        return

    try:
        if sub in ("help", "commands", "?"):
            await msg.reply(HELP_TEXT)
            return

        if sub in ("watch", "watches"):
            await msg.reply(
                _handle_watch_args(
                    args,
                    created_by=str(msg.author.id),
                    channel_id=str(msg.channel.id),
                )
            )
            return

        if sub in ("today", "buys"):
            await msg.reply(_cmd_today())
            return

        if sub in ("opt", "opts", "options"):
            # `!lia opt` / `!lia options` → list option positions.
            # `!lia opt positions` also accepted.
            await msg.reply(_cmd_opt_positions())
            return

        if sub in ("spreads", "spread"):
            # `!lia spread open …` is an alias for `!lia open spread …`;
            # bare `!lia spread` shows pending/recent closes.
            if args and args[0].lower() == "open":
                sub, args = "open", ["spread", *args[1:]]
            else:
                await msg.reply(_cmd_close_status())
                return

        if sub == "close":
            if not args or args[0].lower() in (
                "status",
                "jobs",
                "list",
                "pending",
            ):
                await msg.reply(_cmd_close_status())
                return
            if args[0].lower() in ("help", "?"):
                await msg.reply(
                    "Close both SPY spread legs (buy the short, sell the long).\n"
                    "`!lia close spread <id1> <id2> [QTY] at 2pm`\n"
                    "`!lia close spread <id1> <id2> [QTY] in 1h`\n"
                    "`!lia close spread <id1> <id2> [QTY] if spy >= 650`\n"
                    "`!lia close spread <id1> <id2> [QTY] when spy hits 650`\n"
                    "`!lia close spread <id1> <id2> [QTY] now`\n"
                    "`!lia close status` · `!lia close cancel <job>`\n"
                    "Ids from `!lia opt`. SPY options only."
                )
                return
            if not _trader_allowed(msg.author.id):
                await msg.reply(
                    f"Not authorized to trade "
                    f"(Discord user `{msg.author.id}` not in "
                    f"`LIA_DISCORD_ALLOWLIST`)."
                )
                return
            if args[0].lower() in ("cancel", "stop", "rm", "delete"):
                if len(args) < 2:
                    await msg.reply(
                        "Usage: `!lia close cancel <job>` — ids from `!lia close status`."
                    )
                    return
                await msg.reply(_cmd_close_cancel(args[1]))
                return
            spread_args = args[1:] if args[0].lower() in ("spread", "spr") else args
            await msg.reply(
                _schedule_spread_close(
                    spread_args,
                    created_by=str(msg.author.id),
                    channel_id=str(msg.channel.id),
                )
            )
            return

        if sub == "open":
            if args and args[0].lower() in ("help", "?"):
                await msg.reply(spread_open.USAGE)
                return
            if not _trader_allowed(msg.author.id):
                await msg.reply(
                    f"Not authorized to trade "
                    f"(Discord user `{msg.author.id}` not in "
                    f"`LIA_DISCORD_ALLOWLIST`)."
                )
                return
            spread_args = (
                args[1:] if args and args[0].lower() in ("spread", "spr")
                else args
            )
            try:
                req = spread_open.parse_open_spread_args(spread_args)
            except ValueError as e:
                await msg.reply(str(e))
                return
            await msg.reply(
                f"Submitting **{req.kind} credit spread** "
                f"`{req.symbol} {req.exp_token} "
                f"-{req.short_strike:g}/+{req.long_strike:g} "
                f"{req.option_type}` x `{req.qty}`…"
            )
            await msg.reply(_cmd_open_spread(req))
            return

        if sub in ("positions", "position", "own", "holdings", "portfolio"):
            # `!lia positions opt` → option positions; else stocks.
            if args and args[0].lower() in ("opt", "opts", "options", "option"):
                await msg.reply(_cmd_opt_positions())
            else:
                await msg.reply(_cmd_positions())
            return

        if sub in ("buy", "sell"):
            if not _trader_allowed(msg.author.id):
                await msg.reply(
                    f"Not authorized to trade "
                    f"(Discord user `{msg.author.id}` not in "
                    f"`LIA_DISCORD_ALLOWLIST`)."
                )
                return

            # Options branch: `!lia buy opt ...` / `!lia sell opt <id>`
            if args and args[0].lower() in ("opt", "option", "opts", "options"):
                opt_args = args[1:]
                if sub == "buy":
                    try:
                        (
                            symbol,
                            exp_token,
                            strike_token,
                            otype,
                            qty,
                            limit,
                        ) = _parse_opt_buy_args(opt_args)
                    except ValueError as e:
                        await msg.reply(str(e))
                        return
                    await msg.reply(
                        f"Submitting **buy-opt** `{symbol} {exp_token} "
                        f"{strike_token} {otype}` x `{qty}`…"
                    )
                    await msg.reply(
                        _resolve_opt_buy(
                            symbol, exp_token, strike_token, otype, qty, limit
                        )[0]
                    )
                    return

                try:
                    short_id, qty, limit = _parse_opt_sell_args(opt_args)
                except ValueError as e:
                    await msg.reply(str(e))
                    return
                await msg.reply(
                    f"Submitting **sell-opt** `{short_id}`"
                    + (f" x `{qty}`" if qty is not None else " (all)")
                    + "…"
                )
                await msg.reply(_cmd_sell_opt(short_id, qty, limit))
                return

            # Stock branch.
            if len(args) != 2:
                await msg.reply(
                    f"Usage: `!lia {sub} TICKER QTY` "
                    f"or `!lia {sub} opt …` (see `!lia help`)"
                )
                return
            symbol = args[0].upper().strip()
            try:
                qty = _parse_qty(args[1])
            except ValueError as e:
                await msg.reply(f"Bad quantity: {e}")
                return
            if not symbol.isalnum():
                await msg.reply(f"Bad ticker `{symbol}`.")
                return
            await msg.reply(f"Submitting **{sub}** `{symbol}` x `{qty}`…")
            if sub == "buy":
                await msg.reply(_cmd_buy(symbol, qty))
            else:
                await msg.reply(_cmd_sell(symbol, qty))
            return

        await msg.reply(
            f"Unknown `!lia` command `{sub}`. Try `!lia help`."
        )
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        await msg.reply(f"Error: `{type(e).__name__}: {e}`")


def build_client() -> discord.Client:
    intents = Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)
    watcher_started = {"value": False}

    @client.event
    async def on_ready() -> None:
        print(
            f"[discord] logged in as {client.user} "
            f"(id={client.user and client.user.id})"
        )
        if not watcher_started["value"]:
            watcher_started["value"] = True
            client.loop.create_task(_watch_background(client))
        channel_id = os.getenv("DISCORD_CHANNEL_ID")
        if not channel_id:
            return
        try:
            channel = client.get_channel(int(channel_id))
            if channel is None:
                channel = await client.fetch_channel(int(channel_id))
            if channel is not None:
                await channel.send(
                    "Julia online. Commands start with `!lia` — try `!lia help`."
                )
        except Exception as e:  # noqa: BLE001
            print(f"[discord] ready greeting failed: {e!r}")

    @client.event
    async def on_message(msg: discord.Message) -> None:
        if msg.author.bot:
            return
        await _handle_message(msg)

    return client


def run_bot(token: Optional[str] = None) -> None:
    """Block forever on the Discord gateway."""
    token = token or os.getenv("DISCORD_BOT_TOKEN")
    if not token:
        raise SystemExit(
            "DISCORD_BOT_TOKEN is not set. Create a Discord application at "
            "https://discord.com/developers/applications and put the bot "
            "token in the julia host `.env`."
        )
    client = build_client()
    try:
        client.run(token)
    except discord.errors.PrivilegedIntentsRequired as e:
        raise SystemExit(
            "Discord rejected privileged intents.\n"
            "In https://discord.com/developers/applications → your app → "
            "Bot → Privileged Gateway Intents, enable **Message Content "
            "Intent**, save, then re-run ./restart-discord-bot.sh.\n"
            f"({e})"
        ) from e


if __name__ == "__main__":
    run_bot()
