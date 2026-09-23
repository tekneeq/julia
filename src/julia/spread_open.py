"""Parse + price ``!lia open spread`` — open an options credit spread.

Command shape (mirrors ``!lia buy opt``):

    !lia open spread TICKER EXP SHORT/LONG call|put QTY [CREDIT]

    !lia open spread SPY 0dte 776/778 call 1        # bear call spread
    !lia open spread SPY 2026-09-25 770/768 put 1   # bull put spread
    !lia open spread SPY 0dte 770/768 put 2 0.35    # explicit credit

The first strike is always the SHORT leg (sold), the second the LONG
hedge (bought). Geometry is validated so the order really is a credit
spread: calls need short < long, puts need short > long.

Only pure parsing / validation / pricing lives here so it can be unit
tested without the ``discord`` or ``robin_stocks`` packages; order
submission stays in ``julia.discorder``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

USAGE = (
    "Usage: `!lia open spread TICKER EXP SHORT/LONG call|put QTY [CREDIT]`\n"
    "EXP: `YYYY-MM-DD` or `0dte`/`1dte`/…  ·  strikes: `SHORT/LONG` "
    "(short leg first)\n"
    "Calls: short **below** long (bear call). Puts: short **above** long "
    "(bull put).\n"
    "`!lia open spread SPY 0dte 776/778 call 1` · "
    "`!lia open spread SPY 0dte 770/768 put 1 0.35`"
)


@dataclass
class SpreadOpenRequest:
    symbol: str
    exp_token: str
    short_strike: float
    long_strike: float
    option_type: str  # 'call' | 'put'
    qty: int
    credit: Optional[float]  # None → price off the live quotes

    @property
    def width(self) -> float:
        return abs(self.short_strike - self.long_strike)

    @property
    def kind(self) -> str:
        return "bear call" if self.option_type == "call" else "bull put"


def parse_open_spread_args(args: list[str]) -> SpreadOpenRequest:
    """Parse ``TICKER EXP SHORT/LONG call|put QTY [CREDIT]``."""
    if len(args) < 5:
        raise ValueError(USAGE)

    symbol = args[0].upper().strip()
    if not symbol.isalnum():
        raise ValueError(f"Bad ticker `{symbol}`.\n{USAGE}")

    exp_token = args[1].strip().lower()
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", exp_token) and not re.fullmatch(
        r"\d+dte", exp_token
    ):
        raise ValueError(
            f"Bad EXP `{exp_token}` — use `YYYY-MM-DD` or `0dte`/`1dte`/…"
        )

    strikes_token = args[2].strip()
    m = re.fullmatch(
        r"(\d+(?:\.\d+)?)\s*[/|:-]\s*(\d+(?:\.\d+)?)", strikes_token
    )
    if not m:
        raise ValueError(
            f"Bad strikes `{strikes_token}` — use `SHORT/LONG`, e.g. `776/778`"
        )
    short_strike = float(m.group(1))
    long_strike = float(m.group(2))

    option_type = args[3].lower().strip()
    if option_type in ("c", "call", "calls"):
        option_type = "call"
    elif option_type in ("p", "put", "puts"):
        option_type = "put"
    else:
        raise ValueError("option type must be `call` or `put`")

    try:
        qty = int(float(args[4]))
    except ValueError as e:
        raise ValueError(f"Bad QTY `{args[4]}` — whole contracts.") from e
    if qty <= 0:
        raise ValueError("quantity must be a positive whole number of contracts")

    credit: Optional[float] = None
    if len(args) >= 6:
        try:
            credit = float(args[5])
        except ValueError as e:
            raise ValueError(f"Bad CREDIT `{args[5]}` — dollars, e.g. `0.35`.") from e
        if credit <= 0:
            raise ValueError("CREDIT must be positive (it's what you collect).")

    validate_credit_geometry(short_strike, long_strike, option_type)
    return SpreadOpenRequest(
        symbol=symbol,
        exp_token=exp_token,
        short_strike=short_strike,
        long_strike=long_strike,
        option_type=option_type,
        qty=qty,
        credit=credit,
    )


def validate_credit_geometry(
    short_strike: float, long_strike: float, option_type: str,
) -> None:
    """Short leg must be the expensive one or the order isn't a credit."""
    if short_strike == long_strike:
        raise ValueError("SHORT and LONG strikes must differ.")
    if option_type == "call" and not short_strike < long_strike:
        raise ValueError(
            "Call credit spread needs SHORT **below** LONG "
            f"(got `{short_strike:g}/{long_strike:g}`) — e.g. `776/778`."
        )
    if option_type == "put" and not short_strike > long_strike:
        raise ValueError(
            "Put credit spread needs SHORT **above** LONG "
            f"(got `{short_strike:g}/{long_strike:g}`) — e.g. `770/768`."
        )


def default_credit(
    short_quote: dict, long_quote: dict,
) -> Optional[float]:
    """Natural credit from live quotes: sell the short bid, pay the long ask.

    Falls back to marks when a side is missing. Returns None when there's
    nothing to price off, and never less than $0.01.
    """
    short_px = short_quote.get("bid") or short_quote.get("mark")
    long_px = long_quote.get("ask") or long_quote.get("mark")
    if short_px is None or long_px is None:
        return None
    credit = round(float(short_px) - float(long_px), 2)
    return max(0.01, credit)


def spread_summary(req: SpreadOpenRequest, credit: float) -> dict:
    """Risk numbers for the reply: width, max gain/loss, breakeven."""
    width = req.width
    max_gain = credit * 100.0 * req.qty
    max_loss = max(width - credit, 0.0) * 100.0 * req.qty
    if req.option_type == "call":
        breakeven = req.short_strike + credit
    else:
        breakeven = req.short_strike - credit
    return {
        "width": width,
        "max_gain": max_gain,
        "max_loss": max_loss,
        "breakeven": breakeven,
    }
