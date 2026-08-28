"""Julia Discord bot — ``!lia`` command surface.

Only messages that start with ``!lia`` are handled:

    !lia help
    !lia buy  AAPL 1
    !lia sell AAPL 0.5
    !lia today              # buys submitted/filled today (ET)
    !lia positions          # what you own (sellable qty)

Buy/sell submit **live Robinhood market orders** (fractional-capable) using
the host ``RH_USERNAME`` / ``RH_PASSWORD``.

Env (``.env`` on the julia EC2 host, loaded by docker ``--env-file``):

    DISCORD_BOT_TOKEN         required
    DISCORD_CHANNEL_ID        optional ready greeting
    RH_USERNAME / RH_PASSWORD required for trading / portfolio commands
    LIA_DISCORD_ALLOWLIST     optional comma-separated Discord user IDs
                              allowed to run buy/sell (everyone if unset)
"""
from __future__ import annotations

import os
import traceback
from datetime import date, datetime
from typing import Any, Optional
from zoneinfo import ZoneInfo

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
!lia help                 this message
!lia buy  TICKER QTY      market buy  (live Robinhood)
!lia sell TICKER QTY      market sell (live Robinhood)
!lia today                buys you placed today (ET)
!lia positions            what you own / can sell
!lia own                  alias for positions
```
Examples: `!lia buy AAPL 1` · `!lia sell SPY 0.5` · `!lia today`

Qty may be fractional (up to 6 decimals). Buy/sell place **real orders**.
"""

# Discord hard limit is 2000; leave room for reply chrome.
_DISCORD_SAFE_LEN = 1800


def _ensure_rh_login() -> bool:
    try:
        from julia.main import is_logged_in, login_robinhood

        if is_logged_in():
            return True
        username = os.getenv("RH_USERNAME")
        password = os.getenv("RH_PASSWORD")
        if not username or not password:
            return False
        login_robinhood(username, password)
        return is_logged_in()
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


async def _handle_message(msg: discord.Message) -> None:
    content = (msg.content or "").strip()
    if not content.lower().startswith("!lia"):
        # Ignore every non-!lia message (including old !price / !help).
        return

    sub, args = _parse_lia(content)
    if not sub:
        return

    try:
        if sub in ("help", "commands", "?"):
            await msg.reply(HELP_TEXT)
            return

        if sub in ("today", "buys"):
            await msg.reply(_cmd_today())
            return

        if sub in ("positions", "position", "own", "holdings", "portfolio"):
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
            if len(args) != 2:
                await msg.reply(f"Usage: `!lia {sub} TICKER QTY`")
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

    @client.event
    async def on_ready() -> None:
        print(
            f"[discord] logged in as {client.user} "
            f"(id={client.user and client.user.id})"
        )
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
