"""Julia Discord stock bot — query price, quote, sector/industry, search.

Run via ``scripts/discord_bot.py`` (PID-managed) or:

    uv run python -m julia.discorder

Env (``.env`` on the julia EC2 host, loaded by docker ``--env-file``):

    DISCORD_BOT_TOKEN      required — bot token from a *new* Discord app
    DISCORD_CHANNEL_ID     optional — channel for the ready greeting
    RH_USERNAME / RH_PASSWORD   for live ``!price`` / ``!quote``
"""
from __future__ import annotations

import os
import traceback
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()

# Discord is an optional install unless folded into core deps / --extra chat.
try:
    import discord
    from discord import Intents
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "discord package missing — install with "
        "`uv sync --extra chat` or `pip install discord`"
    ) from e


HELP_TEXT = """**Julia stock bot**
```
!help                         this message
!ping                         latency check
!price  TICKER                latest Robinhood price
!quote  TICKER                price + day change / volume
!info   TICKER                sector, industry, market cap, P/E
!sectors                      sector overview (counts)
!industries <sector>          industries under a sector
!tickers <sector> [industry]  top tickers by market cap
!search <query>               symbol / name search
```
Universe is Robinhood-tradable symbols cached by `lia tickers-sync`.
"""


def _fmt_market_cap(value: Any) -> str:
    if value is None:
        return "—"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    abs_v = abs(v)
    if abs_v >= 1e12:
        return f"${v / 1e12:.2f}T"
    if abs_v >= 1e9:
        return f"${v / 1e9:.2f}B"
    if abs_v >= 1e6:
        return f"${v / 1e6:.2f}M"
    return f"${v:,.0f}"


def _fmt_num(value: Any, digits: int = 2) -> str:
    if value is None or value == "":
        return "—"
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return "—"


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


def _cmd_price(symbol: str) -> str:
    if not _ensure_rh_login():
        return "Robinhood login failed — check `RH_USERNAME` / `RH_PASSWORD`."
    import robin_stocks.robinhood as rh

    vals = rh.stocks.get_latest_price(
        symbol, priceType=None, includeExtendedHours=True
    )
    if not vals or vals[0] is None:
        return f"No price for **{symbol}**."
    return f"**{symbol}** ${float(vals[0]):,.4f}".rstrip("0").rstrip(".")


def _cmd_quote(symbol: str) -> str:
    if not _ensure_rh_login():
        return "Robinhood login failed — check `RH_USERNAME` / `RH_PASSWORD`."
    import robin_stocks.robinhood as rh

    quotes = rh.stocks.get_quotes(symbol)
    if not quotes or not quotes[0]:
        return f"No quote for **{symbol}**."
    q = quotes[0]
    last = float(q.get("last_trade_price") or 0)
    prev = float(q.get("previous_close") or q.get("adjusted_previous_close") or 0)
    ext = q.get("last_extended_hours_trade_price")
    chg = (last - prev) if prev else None
    pct = (chg / prev * 100.0) if prev and chg is not None else None
    lines = [
        f"**{symbol}** quote",
        f"Last: `${last:,.4f}`".rstrip("0").rstrip("."),
    ]
    if prev:
        sign = "+" if (chg or 0) >= 0 else ""
        lines.append(
            f"Prev close: `${prev:,.2f}`  ·  "
            f"Change: `{sign}{_fmt_num(chg)}` (`{sign}{_fmt_num(pct)}%`)"
        )
    if ext:
        lines.append(f"Extended: `${float(ext):,.4f}`")
    if q.get("trading_halted") in (True, "true", "True"):
        lines.append("⚠️ Trading halted")
    return "\n".join(lines)


def _lookup_cached(symbol: str) -> Optional[dict[str, Any]]:
    try:
        from julia import tickers_store

        rows = tickers_store.list_tickers(search=symbol, limit=20)
    except Exception:  # noqa: BLE001
        return None
    symbol = symbol.upper()
    for row in rows:
        if row.get("symbol") == symbol:
            return row
    return None


def _cmd_info(symbol: str) -> str:
    row = _lookup_cached(symbol)
    fund: dict[str, Any] = {}
    if _ensure_rh_login():
        try:
            import robin_stocks.robinhood as rh

            data = rh.stocks.get_fundamentals(symbol)
            if data and data[0]:
                fund = data[0]
        except Exception:  # noqa: BLE001
            fund = {}

    name = None
    if row:
        name = row.get("simple_name") or row.get("name")
    sector = (fund.get("sector") or (row or {}).get("sector") or "—")
    industry = (fund.get("industry") or (row or {}).get("industry") or "—")
    mcap = fund.get("market_cap") or (row or {}).get("market_cap")
    pe = fund.get("pe_ratio") or (row or {}).get("pe_ratio")
    div = fund.get("dividend_yield") or (row or {}).get("dividend_yield")
    desc = (fund.get("description") or (row or {}).get("description") or "")
    itype = (row or {}).get("instrument_type") or ""

    lines = [f"**{symbol}**" + (f" — {name}" if name else "")]
    if itype:
        lines.append(f"Type: `{itype.upper()}`")
    lines.append(f"Sector: **{sector}**")
    lines.append(f"Industry: **{industry}**")
    lines.append(
        f"Market cap: `{_fmt_market_cap(mcap)}`  ·  "
        f"P/E: `{_fmt_num(pe)}`  ·  Div yield: `{_fmt_num(div)}%`"
    )
    if desc:
        snippet = desc.strip().replace("\n", " ")
        if len(snippet) > 280:
            snippet = snippet[:277] + "…"
        lines.append(snippet)
    if not row and not fund:
        lines.append(
            "_No cache / fundamentals — run `lia tickers-sync` or check RH login._"
        )
    return "\n".join(lines)


def _cmd_sectors() -> str:
    from julia import tickers_store

    n = tickers_store.ticker_count()
    if n == 0:
        return "Ticker cache empty — run `lia tickers-sync` (or Sync on the Tickers tab)."
    rows = tickers_store.list_sectors()
    lines = [f"**Sectors** ({n:,} tradable symbols)"]
    for r in rows[:25]:
        lines.append(f"• {r['sector']} — `{r['count']:,}`")
    if len(rows) > 25:
        lines.append(f"_…and {len(rows) - 25} more_")
    return "\n".join(lines)


def _cmd_industries(sector: str) -> str:
    from julia import tickers_store

    rows = tickers_store.list_industries(sector)
    if not rows:
        # Case-insensitive sector match
        for s in tickers_store.list_sectors():
            if s["sector"].lower() == sector.lower():
                sector = s["sector"]
                rows = tickers_store.list_industries(sector)
                break
    if not rows:
        return (
            f"No industries for sector `{sector}`. "
            "Try `!sectors` for names."
        )
    lines = [f"**{sector}** industries"]
    for r in rows:
        lines.append(f"• {r['industry']} — `{r['count']:,}`")
    return "\n".join(lines)


def _cmd_tickers(sector: str, industry: Optional[str] = None) -> str:
    from julia import tickers_store

    # Fuzzy sector match
    sectors = {s["sector"].lower(): s["sector"] for s in tickers_store.list_sectors()}
    sector_key = sector.lower()
    if sector_key in sectors:
        sector = sectors[sector_key]
    else:
        matches = [name for low, name in sectors.items() if sector_key in low]
        if len(matches) == 1:
            sector = matches[0]
        elif not matches:
            return f"Unknown sector `{sector}`. Try `!sectors`."

    if industry:
        inds = {
            i["industry"].lower(): i["industry"]
            for i in tickers_store.list_industries(sector)
        }
        ind_key = industry.lower()
        if ind_key in inds:
            industry = inds[ind_key]
        else:
            matches = [name for low, name in inds.items() if ind_key in low]
            if len(matches) == 1:
                industry = matches[0]
            elif not matches:
                return (
                    f"Unknown industry `{industry}` in **{sector}**. "
                    f"Try `!industries {sector}`."
                )

    rows = tickers_store.list_tickers(
        sector=sector, industry=industry, limit=15
    )
    if not rows:
        return "No tickers matched."
    title = f"**{industry}** in {sector}" if industry else f"**{sector}**"
    lines = [f"{title} — top by market cap"]
    for r in rows:
        name = r.get("simple_name") or r.get("name") or ""
        lines.append(
            f"`{r['symbol']:<6}` {_fmt_market_cap(r.get('market_cap'))}"
            + (f"  {name}" if name else "")
        )
    return "\n".join(lines)


def _cmd_search(query: str) -> str:
    from julia import tickers_store

    rows = tickers_store.list_tickers(search=query, limit=15)
    if not rows:
        return f"No matches for `{query}`."
    lines = [f"**Search** `{query}`"]
    for r in rows:
        name = r.get("simple_name") or r.get("name") or ""
        sector = r.get("sector") or "—"
        lines.append(
            f"`{r['symbol']:<6}` {name}  ·  {sector}  ·  "
            f"{_fmt_market_cap(r.get('market_cap'))}"
        )
    return "\n".join(lines)


def _parse_args(content: str, command: str) -> list[str]:
    rest = content[len(command) :].strip()
    if not rest:
        return []
    return rest.split()


async def _handle_message(msg: discord.Message) -> None:
    content = (msg.content or "").strip()
    if not content.startswith("!"):
        return

    lower = content.lower()
    try:
        if lower == "!ping":
            await msg.reply("pong!")
            return
        if lower in ("!help", "!commands"):
            await msg.reply(HELP_TEXT)
            return
        if lower.startswith("!price"):
            args = _parse_args(content, "!price")
            if not args:
                await msg.reply("Usage: `!price TICKER`")
                return
            await msg.reply(_cmd_price(args[0].upper()))
            return
        if lower.startswith("!quote"):
            args = _parse_args(content, "!quote")
            if not args:
                await msg.reply("Usage: `!quote TICKER`")
                return
            await msg.reply(_cmd_quote(args[0].upper()))
            return
        if lower.startswith("!info"):
            args = _parse_args(content, "!info")
            if not args:
                await msg.reply("Usage: `!info TICKER`")
                return
            await msg.reply(_cmd_info(args[0].upper()))
            return
        if lower in ("!sectors", "!sector"):
            await msg.reply(_cmd_sectors())
            return
        if lower.startswith("!industries") or lower.startswith("!industry"):
            cmd = "!industries" if lower.startswith("!industries") else "!industry"
            args = _parse_args(content, cmd)
            if not args:
                await msg.reply("Usage: `!industries <sector>`")
                return
            await msg.reply(_cmd_industries(" ".join(args)))
            return
        if lower.startswith("!tickers"):
            args = _parse_args(content, "!tickers")
            if not args:
                await msg.reply("Usage: `!tickers <sector> [industry]`")
                return
            from julia import tickers_store

            sectors = [s["sector"] for s in tickers_store.list_sectors()]
            joined = " ".join(args)
            sector = None
            industry = None
            # Prefer longest sector name match against the full arg string.
            for candidate in sorted(sectors, key=len, reverse=True):
                if joined.lower() == candidate.lower():
                    sector = candidate
                    break
                prefix = candidate.lower() + " "
                if joined.lower().startswith(prefix):
                    sector = candidate
                    industry = joined[len(candidate) :].strip() or None
                    break
            if sector is None:
                # Fall back: first token as sector, rest as industry.
                sector = args[0]
                industry = " ".join(args[1:]) or None
            await msg.reply(_cmd_tickers(sector, industry))
            return
        if lower.startswith("!search"):
            args = _parse_args(content, "!search")
            if not args:
                await msg.reply("Usage: `!search <query>`")
                return
            await msg.reply(_cmd_search(" ".join(args)))
            return
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        await msg.reply(f"Error: `{type(e).__name__}: {e}`")


def build_client() -> discord.Client:
    intents = Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)

    @client.event
    async def on_ready() -> None:
        print(f"[discord] logged in as {client.user} (id={client.user and client.user.id})")
        channel_id = os.getenv("DISCORD_CHANNEL_ID")
        if not channel_id:
            return
        try:
            channel = client.get_channel(int(channel_id))
            if channel is None:
                channel = await client.fetch_channel(int(channel_id))
            if channel is not None:
                await channel.send(
                    "Julia stock bot online. Type `!help` for commands."
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
