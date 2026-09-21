"""Spot prints where net GEX (put+/call−) changes environment.

GEX is restated at trial spots from the snapshot's OI and IV. Black-Scholes
gamma's 1/S cancels the S in julia's GEX formula, so the remaining
spot-dependence is n(d1): ATM gamma migrates as price prints. Puts stay
positive and calls negative, so a lower print typically adds GEX+ and a
higher print adds GEX−.

The $1M |net| band matches the dashboard GEX≈ cut.
"""
from __future__ import annotations

import math
from datetime import date
from typing import Iterable, Mapping, Optional

import numpy as np

NEUTRAL_ABS = 1_000_000.0
_SCAN_PCT = 0.12
_SCAN_POINTS = 241  # ~10 bps steps across ±12%


def years_to_expiry(expiration: str, as_of: Optional[date] = None) -> float:
    """Same floor as ``calculate_time_to_expiry`` (min 1/365 year)."""
    as_of = as_of or date.today()
    exp = date.fromisoformat(str(expiration)[:10])
    return max((exp - as_of).days / 365.0, 1.0 / 365.0)


def contracts_from_strike_rows(rows: Iterable[Mapping]) -> list[dict]:
    """Normalize snapshot strike rows into pricer inputs."""
    contracts: list[dict] = []
    for row in rows:
        try:
            oi = int(row["open_interest"] or 0)
            iv = row["implied_vol"]
            k = row["strike_price"]
            typ = str(row["option_type"] or "").lower()
        except (KeyError, TypeError, ValueError):
            continue
        if oi <= 0 or iv is None or k is None:
            continue
        try:
            sigma = float(iv)
            strike = float(k)
        except (TypeError, ValueError):
            continue
        if sigma <= 0 or strike <= 0:
            continue
        if typ not in ("call", "put"):
            continue
        contracts.append({
            "K": strike,
            "sigma": sigma,
            "oi": oi,
            "sign": -1.0 if typ == "call" else 1.0,
        })
    return contracts


def net_gex_at_spots(
    contracts: list[dict],
    spots: np.ndarray,
    *,
    T: float,
    r: float,
    q: float = 0.0,
) -> np.ndarray:
    """Vectorized net GEX at each spot (julia put+/call− dollars).

    Matches ``OptionPricer.gex_per_contract``: gamma * OI * 100 * S * 0.01
    with calls flipped negative. The S terms cancel, leaving
    sign * exp(-qT) * n(d1) * OI / (sigma * sqrt(T)).
    """
    spots = np.asarray(spots, dtype=float)
    net = np.zeros(spots.shape, dtype=float)
    if T <= 0 or not contracts:
        return net
    sqrt_t = math.sqrt(T)
    exp_q = math.exp(-q * T)
    inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)
    for c in contracts:
        sigma = float(c["sigma"])
        oi = float(c["oi"])
        k = float(c["K"])
        if sigma <= 0 or oi <= 0 or k <= 0:
            continue
        s = np.maximum(spots, 1e-8)
        d1 = (np.log(s / k) + (r - q + 0.5 * sigma * sigma) * T) / (
            sigma * sqrt_t
        )
        pdf = np.exp(-0.5 * d1 * d1) * inv_sqrt_2pi
        net += float(c["sign"]) * exp_q * pdf * oi / (sigma * sqrt_t)
    return net


def crossings_of(
    spots: np.ndarray, values: np.ndarray, level: float,
) -> list[float]:
    """Linearly interpolated spots where ``values`` crosses ``level``."""
    y = np.asarray(values, dtype=float) - float(level)
    xs = np.asarray(spots, dtype=float)
    hits: list[float] = []
    for i in range(len(y) - 1):
        a, b = float(y[i]), float(y[i + 1])
        if a == 0.0:
            hits.append(float(xs[i]))
            continue
        if a * b < 0.0:
            t = a / (a - b)
            hits.append(float(xs[i] + t * (xs[i + 1] - xs[i])))
        elif b == 0.0:
            hits.append(float(xs[i + 1]))
    return _dedupe_near(hits)


def _dedupe_near(hits: list[float], rel: float = 1e-6) -> list[float]:
    if not hits:
        return []
    out: list[float] = []
    for h in hits:
        if not out or abs(h - out[-1]) > max(abs(h) * rel, 1e-8):
            out.append(h)
    return out


def nearest(hits: list[float], ref: float) -> Optional[float]:
    if not hits:
        return None
    return min(hits, key=lambda x: abs(x - float(ref)))


def find_gex_flips(
    contracts: list[dict],
    *,
    spot: float,
    T: float,
    r: float,
    q: float = 0.0,
    scan_pct: float = _SCAN_PCT,
) -> dict:
    """Locate zero-gamma and ±$1M env-band prints nearest ``spot``.

    Returns a dict with:
      net_at_spot, zero, to_neutral, to_opposite,
      to_gex_plus, to_gex_minus, scan_lo, scan_hi
    """
    empty = {
        "net_at_spot": None,
        "zero": None,
        "to_neutral": None,
        "to_opposite": None,
        "to_gex_plus": None,
        "to_gex_minus": None,
        "scan_lo": None,
        "scan_hi": None,
    }
    if spot is None or float(spot) <= 0 or not contracts:
        return empty

    spot = float(spot)
    lo = spot * (1.0 - scan_pct)
    hi = spot * (1.0 + scan_pct)
    spots = np.linspace(lo, hi, _SCAN_POINTS)
    nets = net_gex_at_spots(contracts, spots, T=T, r=r, q=q)
    net0 = float(np.interp(spot, spots, nets))

    zero = nearest(crossings_of(spots, nets, 0.0), spot)
    plus_band = nearest(crossings_of(spots, nets, NEUTRAL_ABS), spot)
    minus_band = nearest(crossings_of(spots, nets, -NEUTRAL_ABS), spot)

    if net0 > NEUTRAL_ABS:
        to_neutral = plus_band
        to_opposite = minus_band
    elif net0 < -NEUTRAL_ABS:
        to_neutral = minus_band
        to_opposite = plus_band
    else:
        to_neutral = None
        to_opposite = None

    return {
        "net_at_spot": net0,
        "zero": zero,
        "to_neutral": to_neutral,
        "to_opposite": to_opposite,
        "to_gex_plus": plus_band,
        "to_gex_minus": minus_band,
        "scan_lo": lo,
        "scan_hi": hi,
    }
