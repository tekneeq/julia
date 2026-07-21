import json
import numpy as np
from scipy.stats import norm
import pandas as pd
import robin_stocks.robinhood as rh
from dotenv import load_dotenv
import os
from scipy.optimize import brentq
import click
from datetime import datetime, timedelta
from julia.options import OptionPricer
from julia.options_cache import get_cache_instance
from julia import predictions_store
from julia import gex_store
import functools

# looks for a .env file in the current directory
# RH_USERNAME and RH_PASSWORD should be set in the .env file
load_dotenv()


def is_logged_in():
    """
    Check if user is already logged in to Robinhood.

    Returns:
    - bool: True if logged in, False otherwise
    """
    try:
        # Try to load account profile - this will fail if not logged in
        rh.profiles.load_account_profile()
        return True
    except:
        return False


def ensure_logged_in(func):
    """
    Decorator to ensure user is logged in before executing command.
    If not logged in, will call login_robinhood with credentials from environment.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not is_logged_in():
            username = os.getenv("RH_USERNAME")
            password = os.getenv("RH_PASSWORD")
            if not username or not password:
                click.echo(
                    "❌ Error: RH_USERNAME and RH_PASSWORD must be set in environment variables or .env file"
                )
                click.echo("Please set these credentials and try again.")
                return

            click.echo("🔐 Logging in to Robinhood...")
            try:
                login_robinhood(username, password)
                click.echo("✅ Successfully logged in to Robinhood")
            except Exception as e:
                click.echo(f"❌ Login failed: {e}")
                return

        return func(*args, **kwargs)

    return wrapper


def login_robinhood(username, password):
    """
    Login to Robinhood account.

    Parameters:
    - username: str, Robinhood username
    - password: str, Robinhood password

    Returns:
    - None
    """
    rh.login(username=username, password=password)


def implied_move(stock_price, iv, days_list, confidence_levels):
    """
    Calculate implied move over varying days and confidence levels.

    Parameters:
    - stock_price: float, current stock price
    - iv: float, annualized implied volatility (e.g., 0.25 for 25%)
    - days_list: list of ints, number of days
    - confidence_levels: list of floats, confidence levels (e.g., 0.68, 0.95)

    Returns:
    - DataFrame with implied moves
    """
    results = []
    for days in days_list:
        for conf in confidence_levels:
            z = norm.ppf((1 + conf) / 2)  # two-tailed z-score
            move = stock_price * iv * np.sqrt(days / 252) * z
            results.append(
                {
                    "Days": days,
                    "Confidence Level": f"{int(conf * 100)}%",
                    "Implied Move ($)": round(move, 2),
                    "Implied Move (%)": f"{round((move / stock_price) * 100, 2)}%",
                    "Price Range": f"{round(stock_price - move, 2)} - {round(stock_price + move, 2)}",
                }
            )

    return pd.DataFrame(results)


# Black-Scholes Call Price
def black_scholes_call_price(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


# Function to solve: difference between market price and BS price
def implied_volatility_call(S, K, T, r, market_price):
    objective = (
        lambda sigma: black_scholes_call_price(S, K, T, r, sigma) - market_price
    )
    return brentq(
        objective, 1e-5, 3
    )  # search in a reasonable range for sigma (0.001% - 300%)


def business_days_from_today(n):
    current_date = datetime.today()
    direction = 1 if n >= 0 else -1
    days_remaining = abs(n)

    while days_remaining > 0:
        current_date += timedelta(days=direction)
        if current_date.weekday() < 5:  # Weekdays are 0-4 (Mon-Fri)
            days_remaining -= 1
    return current_date.strftime("%Y-%m-%d")


def calculate_time_to_expiry(expiration_date):
    """
    Calculate time to expiry in years from current date to expiration date.

    Parameters:
    - expiration_date: str, expiration date in 'YYYY-MM-DD' format

    Returns:
    - float, time to expiry in years
    """
    current_date = datetime.now()
    exp_date = datetime.strptime(expiration_date, "%Y-%m-%d")
    days_diff = (exp_date - current_date).days
    return max(
        days_diff / 365.0, 1 / 365
    )  # Minimum 1 day to avoid division by zero


def calculate_options_greeks(
    ticker,
    expiration_date,
    risk_free_rate=0.02,
    use_cache=True,
    refresh_cache=False,
):
    """
    Calculate delta and gamma for all options of a given ticker on a specific expiration date.

    Parameters:
    - ticker: str, stock ticker symbol
    - expiration_date: str, expiration date in 'YYYY-MM-DD' format
    - risk_free_rate: float, risk-free interest rate (default 2%)
    - use_cache: bool, whether to use cached data (default True)
    - refresh_cache: bool, whether to force refresh cache (default False)

    Returns:
    - pandas.DataFrame with options data and calculated Greeks
    """
    cache = get_cache_instance()

    # Create cache parameters
    cache_params = {"risk_free_rate": risk_free_rate}

    # Try to get cached data first (unless refresh is requested)
    if use_cache and not refresh_cache:
        cached_data = cache.get(ticker, expiration_date, cache_params)
        if cached_data:
            # Convert cached data back to DataFrame
            df = pd.DataFrame(cached_data["data"]["options"])

            # Restore DataFrame attributes
            if "portfolio_gex" in cached_data["data"]:
                df.attrs["portfolio_gex"] = cached_data["data"]["portfolio_gex"]
            if "gamma_levels" in cached_data["data"]:
                df.attrs["gamma_levels"] = cached_data["data"]["gamma_levels"]
            if "stock_price" in cached_data["data"]:
                df.attrs["stock_price"] = cached_data["data"]["stock_price"]
            calc_params = cached_data["data"].get("calculation_params") or {}
            if calc_params.get("calculated_at"):
                df.attrs["cached_calculated_at"] = calc_params["calculated_at"]

            return df

    try:
        # Get current stock price
        stock_price_list = rh.stocks.get_latest_price(
            ticker, priceType=None, includeExtendedHours=True
        )
        if not stock_price_list:
            raise ValueError(f"Could not retrieve price for {ticker}")

        stock_price = float(stock_price_list[0])
        print(f"Current {ticker} price: ${stock_price:.2f}")

        # Calculate time to expiry
        time_to_expiry = calculate_time_to_expiry(expiration_date)
        print(
            f"Time to expiry: {time_to_expiry:.4f} years ({time_to_expiry*365:.1f} days)"
        )

        # Get all available options for the expiration date
        print(
            f"🔄 Fetching options data from Robinhood for {ticker} {expiration_date}..."
        )
        options_data = rh.options.find_options_by_expiration(
            ticker, expiration_date, info=None
        )

        if not options_data:
            raise ValueError(
                f"No options found for {ticker} on {expiration_date}"
            )

        print(f"Found {len(options_data)} options contracts")

        results = []

        for option in options_data:
            try:
                # Extract option details
                strike_price = float(option.get("strike_price", 0))
                option_type = option.get("type", "").lower()  # 'call' or 'put'

                # Get market price (use mark price if available, otherwise midpoint of bid/ask)
                mark_price = option.get("mark_price")
                bid_price = option.get("bid_price")
                ask_price = option.get("ask_price")

                if mark_price and float(mark_price) > 0:
                    market_price = float(mark_price)
                elif (
                    bid_price
                    and ask_price
                    and float(bid_price) > 0
                    and float(ask_price) > 0
                ):
                    market_price = (float(bid_price) + float(ask_price)) / 2
                else:
                    continue  # Skip options without valid pricing

                # Get implied volatility
                implied_vol = option.get("implied_volatility")
                if not implied_vol or float(implied_vol) <= 0:
                    continue  # Skip options without IV

                implied_vol = float(implied_vol)

                # Create OptionPricer instance
                pricer = OptionPricer(
                    S=stock_price,
                    K=strike_price,
                    T=time_to_expiry,
                    r=risk_free_rate,
                    market_price=market_price,
                )

                # Calculate Greeks
                delta = pricer.delta(implied_vol, option_type)
                gamma = pricer.gamma(implied_vol)
                vega = pricer.vega(implied_vol)
                theta = pricer.theta(implied_vol, option_type)

                # Calculate GEX (Gamma Exposure)
                open_interest_val = option.get("open_interest")
                open_interest_val = (
                    int(open_interest_val) if open_interest_val else 0
                )

                gex_per_contract = pricer.gex_per_contract(
                    implied_vol, open_interest_val, option_type
                )
                gex_notional = pricer.gex_notional(
                    implied_vol, open_interest_val, option_type
                )

                # Calculate theoretical price using Black-Scholes
                if option_type == "call":
                    theoretical_price = pricer.black_scholes_call(implied_vol)
                else:
                    theoretical_price = pricer.black_scholes_put(implied_vol)

                # Store results
                results.append(
                    {
                        "ticker": ticker,
                        "expiration_date": expiration_date,
                        "option_type": option_type.upper(),
                        "strike_price": strike_price,
                        "market_price": market_price,
                        "theoretical_price": theoretical_price,
                        "bid_price": float(bid_price) if bid_price else None,
                        "ask_price": float(ask_price) if ask_price else None,
                        "implied_volatility": implied_vol,
                        "delta": delta,
                        "gamma": gamma,
                        "vega": vega,
                        "theta": theta,
                        "gex_per_contract": gex_per_contract,
                        "gex_notional": gex_notional,
                        "volume": option.get("volume"),
                        "open_interest": open_interest_val,
                        "time_to_expiry_days": time_to_expiry * 365,
                    }
                )

            except (ValueError, TypeError, KeyError) as e:
                print(
                    f"Error processing option {option.get('strike_price', 'unknown')}: {e}"
                )
                continue

        if not results:
            raise ValueError(
                f"No valid options data found for {ticker} on {expiration_date}"
            )

        # Create DataFrame and sort by strike price and option type
        df = pd.DataFrame(results)
        df = df.sort_values(["option_type", "strike_price"])

        # Calculate portfolio-level GEX analysis
        if not df.empty:
            # Portfolio GEX analysis
            all_gex = df["gex_per_contract"].tolist()
            portfolio_gex = OptionPricer.calculate_portfolio_gex(all_gex)

            # GEX by strike analysis
            gex_by_strike = (
                df.groupby("strike_price")["gex_per_contract"].sum().to_dict()
            )
            gamma_levels = OptionPricer.get_gamma_levels(
                stock_price, gex_by_strike
            )

            # Add portfolio analysis to DataFrame as metadata
            df.attrs["portfolio_gex"] = portfolio_gex
            df.attrs["gamma_levels"] = gamma_levels
            df.attrs["stock_price"] = stock_price

        # Cache the results if caching is enabled
        if use_cache:
            cache_data = {
                "options": df.to_dict("records"),
                "portfolio_gex": df.attrs.get("portfolio_gex", {}),
                "gamma_levels": df.attrs.get("gamma_levels", {}),
                "stock_price": df.attrs.get("stock_price", stock_price),
                "calculation_params": {
                    "risk_free_rate": risk_free_rate,
                    "time_to_expiry": time_to_expiry,
                    "calculated_at": datetime.now().isoformat(),
                },
            }
            cache.set(ticker, expiration_date, cache_data, cache_params)

        return df

    except Exception as e:
        print(f"Error calculating options Greeks: {e}")
        return pd.DataFrame()


@click.group()
def cli():
    """
    Command line interface for Robinhood options trading.

    This CLI allows you to login to your Robinhood account and perform various options trading calculations.
    """
    pass


@cli.command()
@click.option(
    "--ticker", default="SPY", help="Ticker symbol of the stock (default: SPY)"
)
@click.option(
    "--days",
    default="1,3,5",
    help="Comma-separated list of days for implied move calculation (default: 1,3,5,10,21,30,60)",
)
@click.option(
    "--confidence",
    default="0.6827,0.9545,0.997",
    help="Comma-separated list of confidence levels (default: 0.6827,0.9545,0.997)",
)
@click.option(
    "--expiration",
    default=None,
    help="Expiration date YYYY-MM-DD to source IV from "
    "(default: next business day). Use today's date for 0DTE IV.",
)
@click.option(
    "--today",
    is_flag=True,
    help="Shortcut for --expiration=<today> (0DTE IV).",
)
@click.option(
    "--no-record",
    is_flag=True,
    help="Don't persist this prediction to the local SQLite store",
)
@ensure_logged_in
def emove(ticker, days, confidence, expiration, today, no_record):

    stock_price = rh.stocks.get_latest_price(
        ticker, priceType=None, includeExtendedHours=True
    )
    if stock_price is None:
        click.echo(
            f"Could not retrieve price for {ticker}. Please check the ticker symbol."
        )
        return

    stock_price = float(stock_price[0])
    click.echo(f"Current {ticker} price: {stock_price}")

    if today and expiration:
        click.echo("❌ Pass either --today or --expiration, not both.")
        return
    if today:
        expirationDate = business_days_from_today(0)
    elif expiration:
        try:
            datetime.strptime(expiration, "%Y-%m-%d")
        except ValueError:
            click.echo(f"❌ Invalid --expiration '{expiration}'. Use YYYY-MM-DD.")
            return
        expirationDate = expiration
    else:
        expirationDate = business_days_from_today(1)
    strikePrice = int(stock_price)
    options_data_list = rh.options.find_options_by_expiration_and_strike(
        ticker, expirationDate, strikePrice, optionType="call", info=None
    )
    options_data = options_data_list[0]

    # print(json.dumps(options_data, indent=4))
    iv = options_data.get("implied_volatility", None)
    if iv is None:
        click.echo(
            f"No implied volatility data available for {ticker} on {expirationDate} with strike {strikePrice}."
        )
        click.echo("Please check the options data or try a different ticker.")
        return

    iv = float(iv) if iv else 0.1956  # Default to 19.56% if not available
    click.echo(
        f"Implied Volatility for {ticker} on {expirationDate} with strike {strikePrice}: {iv:.4%}"
    )

    # implied_volatility = 0.1956  # 30% IV
    # days = [1, 3, 5, 10, 21, 30, 60]

    # 1 sig = 68% confidence,
    # 2 sig = 95% confidence
    # 3 sig = 99.7% confidence
    # confidence_levels = [0.6827, 0.9545, 0.997]
    confidence_levels = [float(c) for c in confidence.split(",")]
    days_list = [int(d) for d in days.split(",")]

    df = implied_move(stock_price, iv, days_list, confidence_levels)
    print(df)

    if not no_record:
        bands = _build_bands(stock_price, iv, days_list, confidence_levels)
        try:
            pid = predictions_store.record_prediction(
                ticker=ticker,
                spot_price=stock_price,
                iv=iv,
                strike_price=float(strikePrice),
                option_type="call",
                expiration_date=expirationDate,
                bands=bands,
            )
            click.echo(f"📒 Recorded prediction {pid}")
        except Exception as e:
            click.echo(f"⚠️  Could not record prediction: {e}")


def _build_bands(stock_price, iv, days_list, confidence_levels):
    """Build raw-float Band rows mirroring the printed implied_move table."""
    bands = []
    for days in days_list:
        target_date = business_days_from_today(days)
        for conf in confidence_levels:
            z = norm.ppf((1 + conf) / 2)
            move = stock_price * iv * np.sqrt(days / 252) * z
            bands.append(
                predictions_store.Band(
                    days=days,
                    target_date=target_date,
                    confidence=conf,
                    implied_move=round(float(move), 4),
                    low=round(float(stock_price - move), 4),
                    high=round(float(stock_price + move), 4),
                )
            )
    return bands


def _daily_close_by_date(ticker):
    """Fetch a year of daily bars for `ticker` and return {YYYY-MM-DD: close}."""
    historicals = rh.stocks.get_stock_historicals(
        ticker, interval="day", span="year"
    )
    closes = {}
    for bar in historicals or []:
        begins_at = bar.get("begins_at") or ""
        close = bar.get("close_price")
        if not begins_at or close is None:
            continue
        # begins_at looks like '2026-06-20T13:30:00Z'
        date = begins_at[:10]
        try:
            closes[date] = float(close)
        except (TypeError, ValueError):
            continue
    return closes


@cli.command("emove-check")
@click.option(
    "--ticker",
    default=None,
    help="Only check predictions for this ticker (default: all)",
)
@ensure_logged_in
def emove_check(ticker):
    """Backfill realized closes for `emove` predictions whose target date has passed."""
    today = datetime.now().strftime("%Y-%m-%d")
    pending = predictions_store.pending_outcomes(today, ticker=ticker)
    if not pending:
        click.echo("✅ Nothing to check — no pending outcomes")
        return

    click.echo(f"Found {len(pending)} pending outcome(s)")

    close_cache: dict[str, dict[str, float]] = {}
    recorded = 0
    skipped = 0
    for row in pending:
        sym = row["ticker"]
        if sym not in close_cache:
            try:
                close_cache[sym] = _daily_close_by_date(sym)
            except Exception as e:
                click.echo(f"  ⚠️  {sym}: failed to fetch historicals ({e})")
                close_cache[sym] = {}
        close = close_cache[sym].get(row["target_date"])
        if close is None:
            click.echo(
                f"  ⏭  {sym} {row['target_date']} (+{row['days']}d): no close available yet"
            )
            skipped += 1
            continue
        actual_move = close - float(row["spot_price"])
        predictions_store.record_outcome(
            prediction_id=row["prediction_id"],
            days=row["days"],
            target_date=row["target_date"],
            actual_price=close,
            actual_move=actual_move,
        )
        sign = "+" if actual_move >= 0 else ""
        click.echo(
            f"  📌 {sym} +{row['days']}d {row['target_date']}: "
            f"close={close:.2f} move={sign}{actual_move:.2f} "
            f"({sign}{actual_move / float(row['spot_price']) * 100:.2f}%)"
        )
        recorded += 1

    click.echo(f"\nRecorded {recorded}, skipped {skipped}")


@cli.command("emove-stats")
@click.option(
    "--ticker", default=None, help="Filter to one ticker (default: all)"
)
def emove_stats(ticker):
    """Show calibration of recorded `emove` predictions vs realized moves."""
    summary = predictions_store.counts()
    click.echo(
        f"📊 predictions={summary['predictions']} "
        f"bands={summary['bands']} outcomes={summary['outcomes']}"
    )

    rows = predictions_store.stats_rows(ticker=ticker)
    if not rows:
        click.echo(
            "No realized outcomes yet — run `lia emove-check` once a "
            "prediction's target date has passed."
        )
        return

    df = pd.DataFrame(
        [
            {
                "ticker": r["ticker"],
                "days": r["days"],
                "conf": f"{r['confidence'] * 100:.2f}%",
                "n": r["n"],
                "hits": r["hits"],
                "hit_rate": f"{(r['hits'] / r['n']) * 100:.1f}%",
                "avg_implied": f"{r['avg_implied_pct'] * 100:.2f}%",
                "avg_realized_abs": f"{r['avg_abs_move_pct'] * 100:.2f}%",
            }
            for r in rows
        ]
    )
    click.echo("")
    click.echo(df.to_string(index=False))


@cli.command()
@click.option(
    "--ticker", default="SPY", help="Ticker symbol of the stock (default: SPY)"
)
@click.option(
    "--days", default=1, help="Days from now to get options data (default: 1)"
)
@ensure_logged_in
def option(ticker="SPY", days=1):
    expiration_date = business_days_from_today(days)
    options_data = rh.options.find_options_by_expiration(
        ticker, expiration_date, info=None
    )
    print(json.dumps(options_data, indent=4))


@cli.command()
@click.option(
    "--ticker", default="SPY", help="Ticker symbol of the stock (default: SPY)"
)
@ensure_logged_in
def range(ticker):

    def todays_high_low_intraday(
        symbol: str, bounds: str = "regular"
    ) -> tuple[float | None, float | None, float | None]:
        candles = rh.stocks.get_stock_historicals(
            symbol, interval="5minute", span="day", bounds=bounds
        )
        if not candles:
            return None, None, None
        highs = [
            float(candle["high_price"])
            for candle in candles
            if candle.get("high_price")
        ]
        lows = [
            float(candle["low_price"])
            for candle in candles
            if candle.get("low_price")
        ]
        # Opening price: first candle's open where available
        open_price: float | None = None
        for candle in candles:
            if candle.get("open_price") is not None:
                try:
                    open_price = float(candle["open_price"])
                except (TypeError, ValueError):
                    open_price = None
                break
        return (
            max(highs) if highs else None,
            min(lows) if lows else None,
            open_price,
        )

    def todays_high_low(
        symbol: str,
    ) -> tuple[float | None, float | None, float | None]:
        hi, lo, op = todays_high_low_intraday(symbol)
        if hi is None or lo is None or op is None:
            hi, lo, op = todays_high_low_intraday(symbol, bounds="trading")

        return hi, lo, op

    hi, lo, op = todays_high_low(ticker)
    range_value = (hi - lo) if (hi is not None and lo is not None) else None

    todays_date = business_days_from_today(0)

    if (
        op is not None
        and hi is not None
        and lo is not None
        and range_value is not None
        and op != 0
    ):
        hi_pct = (hi - op) / op
        lo_pct = (lo - op) / op
        range_pct = range_value / op
        print(
            f"{ticker} {todays_date} High: {hi} ({hi_pct:.2%}), Low: {lo} ({lo_pct:.2%}), Range: {range_value:.2f} ({range_pct:.2%})"
        )
    else:
        print(
            f"{ticker} {todays_date} High: {hi}, Low: {lo}, Range: {range_value if range_value is not None else 0.0:.2%}"
        )


@cli.command()
@click.option(
    "--ticker", default="SPY", help="Ticker symbol of the stock (default: SPY)"
)
@click.option(
    "--expiration",
    help="Expiration date in YYYY-MM-DD format (default: next business day)",
)
@click.option(
    "--rate", default=0.02, help="Risk-free interest rate (default: 0.02 or 2%)"
)
@click.option(
    "--output", help="Output file path to save results as CSV (optional)"
)
@click.option(
    "--min-volume", default=0, help="Minimum volume filter (default: 0)"
)
@click.option(
    "--show-all",
    is_flag=True,
    help="Show all Greeks (delta, gamma, vega, theta) instead of just delta and gamma",
)
@click.option(
    "--show-gex",
    is_flag=True,
    help="Show Gamma Exposure (GEX) analysis and positioning",
)
@click.option(
    "--no-cache",
    is_flag=True,
    help="Disable caching and always fetch fresh data",
)
@click.option(
    "--refresh-cache",
    is_flag=True,
    help="Force refresh cached data with fresh API call",
)
@click.option(
    "--no-record",
    is_flag=True,
    help="Don't snapshot this run to the local SQLite store (used by greeks-diff)",
)
@ensure_logged_in
def greeks(
    ticker,
    expiration,  # YYYY-MM-DD, e.g. "2026-06-23"
    rate,
    output,
    min_volume,
    show_all,
    show_gex,
    no_cache,
    refresh_cache,
    no_record,
):
    """
    Calculate delta, gamma, GEX (and optionally other Greeks) for all options on a given expiration date.

    This command fetches all available options contracts for the specified ticker and expiration date,
    then calculates the Black-Scholes Greeks and Gamma Exposure (GEX) for each contract.

    GEX Analysis helps determine if the market is in a Long Gamma, Short Gamma, or Gamma Neutral environment,
    which affects expected market maker hedging flows and volatility patterns.
    """
    try:
        # Set default expiration to next business day if not provided
        if not expiration:
            expiration = business_days_from_today(1)
            click.echo(f"Using default expiration date: {expiration}")

        click.echo(
            f"Calculating Greeks for {ticker} options expiring on {expiration}"
        )
        click.echo(f"Using risk-free rate: {rate:.2%}")

        # Calculate options Greeks
        df = calculate_options_greeks(
            ticker,
            expiration,
            rate,
            use_cache=not no_cache,
            refresh_cache=refresh_cache,
        )

        if df.empty:
            click.echo("No options data found or error occurred.")
            return

        if not no_record:
            try:
                sid = _record_greeks_snapshot(df, ticker, expiration, rate)
                click.echo(f"📒 Recorded snapshot {sid}")
                if not refresh_cache and not no_cache:
                    click.echo(
                        "   (data may be from cache; pass --refresh-cache for a "
                        "fresh intraday snapshot)"
                    )
            except Exception as e:
                click.echo(f"⚠️  Could not record snapshot: {e}")

        # Apply volume filter
        if min_volume > 0:
            df = df[df["volume"].fillna(0) >= min_volume]
            click.echo(f"Filtered to options with volume >= {min_volume}")

        # Select columns to display
        if show_all:
            display_columns = [
                "option_type",
                "strike_price",
                "market_price",
                "theoretical_price",
                "bid_price",
                "ask_price",
                "implied_volatility",
                "delta",
                "gamma",
                "vega",
                "theta",
                "gex_per_contract",
                "gex_notional",
                "volume",
                "open_interest",
            ]
        else:
            display_columns = [
                "option_type",
                "strike_price",
                "market_price",
                "implied_volatility",
                "delta",
                "gamma",
                "gex_per_contract",
                "gex_notional",
                "volume",
                "open_interest",
            ]

        # Remove GEX columns if not requested for display
        if not show_gex:
            display_columns = [
                col for col in display_columns if not col.startswith("gex_")
            ]

        # Format and display results
        display_df = df[display_columns].copy()

        # Round numerical columns for better display
        numeric_columns = [
            "market_price",
            "theoretical_price",
            "bid_price",
            "ask_price",
            "implied_volatility",
            "delta",
            "gamma",
            "vega",
            "theta",
            "gex_per_contract",
            "gex_notional",
        ]
        for col in numeric_columns:
            if col in display_df.columns:
                if col == "implied_volatility":
                    display_df[col] = display_df[col].round(4)
                elif col in ["delta", "gamma", "vega", "theta"]:
                    display_df[col] = display_df[col].round(6)
                elif col in ["gex_per_contract", "gex_notional"]:
                    display_df[col] = display_df[col].round(0).astype(int)
                else:
                    display_df[col] = display_df[col].round(2)

        click.echo(f"\nFound {len(display_df)} options contracts:")
        click.echo(
            f"Calls: {len(display_df[display_df['option_type'] == 'CALL'])}"
        )
        click.echo(
            f"Puts: {len(display_df[display_df['option_type'] == 'PUT'])}"
        )

        # Display summary statistics
        click.echo(
            f"\nDelta range: {df['delta'].min():.4f} to {df['delta'].max():.4f}"
        )
        click.echo(
            f"Gamma range: {df['gamma'].min():.6f} to {df['gamma'].max():.6f}"
        )

        # GEX debugging summary
        if "gex_per_contract" in df.columns:
            total_options = len(df)
            options_with_gex = len(df[df["gex_per_contract"] != 0])
            options_with_oi = len(df[df["open_interest"] > 0])

            # Analyze call vs put breakdown
            calls = df[df["option_type"] == "CALL"]
            puts = df[df["option_type"] == "PUT"]
            call_gex_positive = len(calls[calls["gex_per_contract"] > 0])
            put_gex_positive = len(puts[puts["gex_per_contract"] > 0])
            call_gex_negative = len(calls[calls["gex_per_contract"] < 0])
            put_gex_negative = len(puts[puts["gex_per_contract"] < 0])

            click.echo(f"\nGEX Debug Summary:")
            click.echo(f"  Total options: {total_options}")
            click.echo(f"  Calls: {len(calls)} | Puts: {len(puts)}")
            click.echo(f"  Options with open interest > 0: {options_with_oi}")
            click.echo(f"  Options with non-zero GEX: {options_with_gex}")
            click.echo(
                f"  GEX range: ${df['gex_per_contract'].min():,.0f} to ${df['gex_per_contract'].max():,.0f}"
            )

            click.echo(f"\nGEX Sign Analysis:")
            click.echo(
                f"  Calls with negative GEX: {call_gex_negative} ✅ (expected)"
            )
            click.echo(
                f"  Calls with positive GEX: {call_gex_positive} ❌ (unexpected)"
            )
            click.echo(
                f"  Puts with positive GEX: {put_gex_positive} ✅ (expected)"
            )
            click.echo(
                f"  Puts with negative GEX: {put_gex_negative} ❌ (unexpected)"
            )

            if options_with_gex == 0:
                click.echo(
                    f"  ⚠️  All options have $0 GEX - likely due to zero open interest"
                )
            elif put_gex_negative > 0:
                click.echo(
                    f"  🚨 PROBLEM: {put_gex_negative} puts have negative GEX (should be positive)"
                )
            elif len(puts) == 0:
                click.echo(
                    f"  🚨 PROBLEM: No PUT options found - check API data"
                )

        # Display GEX Analysis
        if hasattr(df, "attrs") and "portfolio_gex" in df.attrs and show_gex:
            portfolio_gex = df.attrs["portfolio_gex"]
            gamma_levels = df.attrs["gamma_levels"]
            stock_price = df.attrs["stock_price"]

            click.echo(f"\n{'='*60}")
            click.echo(f"GAMMA EXPOSURE (GEX) ANALYSIS")
            click.echo(f"{'='*60}")

            # Portfolio GEX Summary
            click.echo(f"\nPortfolio GEX Summary:")
            click.echo(f"  Current Position: {portfolio_gex['gamma_position']}")
            click.echo(f"  Total GEX: ${portfolio_gex['total_gex']:,.0f}")
            click.echo(f"  Call GEX: ${portfolio_gex['call_gex']:,.0f}")
            click.echo(f"  Put GEX: ${portfolio_gex['put_gex']:,.0f}")
            click.echo(f"  Net GEX: ${portfolio_gex['net_gex']:,.0f}")
            click.echo(f"\nMarket Impact:")
            click.echo(f"  {portfolio_gex['position_description']}")

            # Key Gamma Levels
            if gamma_levels and "top_3_gamma_strikes" in gamma_levels:
                click.echo(f"\nKey Gamma Levels (Top 3 by absolute GEX):")
                for i, (strike, gex) in enumerate(
                    gamma_levels["top_3_gamma_strikes"], 1
                ):
                    distance = ((strike - stock_price) / stock_price) * 100
                    click.echo(
                        f"  {i}. ${strike:.0f} - GEX: ${gex:,.0f} ({distance:+.1f}% from spot)"
                    )

                # Support/Resistance levels
                if gamma_levels.get("nearest_support"):
                    support_strike, support_gex = gamma_levels[
                        "nearest_support"
                    ]
                    click.echo(
                        f"\nNearest Support: ${support_strike:.0f} (GEX: ${support_gex:,.0f})"
                    )

                if gamma_levels.get("nearest_resistance"):
                    resistance_strike, resistance_gex = gamma_levels[
                        "nearest_resistance"
                    ]
                    click.echo(
                        f"Nearest Resistance: ${resistance_strike:.0f} (GEX: ${resistance_gex:,.0f})"
                    )

            # GEX interpretation
            total_gex_millions = portfolio_gex["total_gex"] / 1000000
            click.echo(f"\nGEX Interpretation:")
            if abs(total_gex_millions) < 1:
                click.echo(
                    f"  Low GEX environment (${abs(total_gex_millions):.1f}M) - Expect normal volatility"
                )
            elif abs(total_gex_millions) < 10:
                click.echo(
                    f"  Moderate GEX environment (${abs(total_gex_millions):.1f}M) - Some dealer flow impact"
                )
            else:
                click.echo(
                    f"  High GEX environment (${abs(total_gex_millions):.1f}M) - Significant dealer flow expected"
                )
        elif show_gex and hasattr(df, "attrs") and "portfolio_gex" in df.attrs:
            # Show basic GEX summary even if detailed analysis failed
            portfolio_gex = df.attrs["portfolio_gex"]
            click.echo(
                f"\nGEX Summary: {portfolio_gex['gamma_position']} - Net GEX: ${portfolio_gex['net_gex']:,.0f}"
            )

        # Display the data
        click.echo("\nOptions Greeks Data:")
        click.echo(display_df.to_string(index=False))

        # Save to CSV if output file specified
        if output:
            df.to_csv(output, index=False)
            click.echo(f"\nFull results saved to: {output}")

    except Exception as e:
        click.echo(f"Error: {e}")


def _record_greeks_snapshot(df, ticker, expiration, rate):
    """Persist a `greeks` run as one gex_snapshots row + N strike rows.

    Reads the unfiltered DataFrame returned by ``calculate_options_greeks``.
    Strike rows with no IV or pricing are already excluded upstream.
    """
    spot_price = float(df.attrs.get("stock_price") or 0.0)
    strikes = []
    for _, row in df.iterrows():
        oi = row.get("open_interest")
        vol = row.get("volume")
        strikes.append(
            gex_store.StrikeSnapshot(
                option_type=str(row["option_type"]).lower(),
                strike_price=float(row["strike_price"]),
                mark_price=_safe_float(row.get("market_price")),
                implied_vol=_safe_float(row.get("implied_volatility")),
                delta=_safe_float(row.get("delta")),
                gamma=_safe_float(row.get("gamma")),
                gex_per_contract=_safe_float(row.get("gex_per_contract")),
                volume=int(vol) if vol not in (None, "") else None,
                open_interest=int(oi) if oi not in (None, "") else None,
            )
        )
    return gex_store.record_snapshot(
        ticker=ticker,
        expiration_date=expiration,
        spot_price=spot_price,
        risk_free_rate=float(rate),
        strikes=strikes,
    )


def _safe_float(v):
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


@cli.command("greeks-history")
@click.option(
    "--ticker", default=None, help="Filter to one ticker (default: all)"
)
@click.option(
    "--expiration", default=None, help="Filter to one expiration YYYY-MM-DD"
)
@click.option("--limit", default=20, help="Max snapshots to show (default: 20)")
def greeks_history(ticker, expiration, limit):
    """List recent gex snapshots (most recent first).

    The `idx` column is what to pass to `greeks-diff --from/--to`
    (1 = most recent).
    """
    rows = gex_store.recent_snapshots(
        ticker=ticker, expiration_date=expiration, limit=limit
    )
    if not rows:
        click.echo("No greeks snapshots yet — run `lia greeks` to record one.")
        return

    df = pd.DataFrame(
        [
            {
                "idx": i + 1,
                "id": r["id"],
                "captured_at": r["captured_at"],
                "ticker": r["ticker"],
                "exp": r["expiration_date"],
                "spot": f"{r['spot_price']:.2f}",
                "C_OI": r["call_oi"],
                "P_OI": r["put_oi"],
                "C_vol": r["call_volume"],
                "P_vol": r["put_volume"],
                "P/C_vol": f"{(r['put_volume'] / r['call_volume']):.2f}"
                if r["call_volume"]
                else "n/a",
                "net_gex": f"{(r['total_gex'] or 0):,.0f}",
            }
            for i, r in enumerate(rows)
        ]
    )
    click.echo(df.to_string(index=False))


@cli.command("greeks-diff")
@click.option(
    "--ticker", required=True, help="Ticker (required)"
)
@click.option(
    "--expiration", required=True, help="Expiration date YYYY-MM-DD (required)"
)
@click.option(
    "--from", "from_ref", default=None,
    help=(
        "Snapshot to compare FROM. Accepts a ULID, an integer index "
        "(1 = most recent, 2 = next, ...), or 'first'. "
        "Default: 2nd most recent."
    ),
)
@click.option(
    "--to", "to_ref", default=None,
    help=(
        "Snapshot to compare TO. Accepts a ULID, an integer index "
        "(1 = most recent), or 'last'/'latest'. Default: most recent."
    ),
)
@click.option(
    "--full-day", is_flag=True,
    help="Shortcut: diff the first snapshot of the day vs the latest "
    "(overrides --from/--to).",
)
@click.option(
    "--top", default=5, type=int,
    help="Show top N per-strike movers by Δvolume per side (default: 5, 0 to skip)",
)
def greeks_diff(ticker, expiration, from_ref, to_ref, full_day, top):
    """Diff two gex snapshots for the same (ticker, expiration).

    Shows what changed between two points in time — most useful for spotting
    where call vs put positioning was added through the day or week.

    Pick snapshots with `--from` / `--to`. Examples:
      `--from 4 --to 1`     (4th-most-recent vs latest)
      `--from first --to 1` (first of the day vs latest)
      `--full-day`          (same as above, shortcut)
      `--from <ULID> --to <ULID>` (explicit)

    Use `lia greeks-history` to see the indices.
    """
    if full_day:
        from_ref = "first"
        to_ref = "1"

    if from_ref or to_ref:
        older = _resolve_snapshot_ref(
            from_ref or "2", ticker=ticker, expiration=expiration
        )
        newer = _resolve_snapshot_ref(
            to_ref or "1", ticker=ticker, expiration=expiration
        )
        if not older:
            click.echo(f"❌ --from snapshot '{from_ref}' not found.")
            return
        if not newer:
            click.echo(f"❌ --to snapshot '{to_ref}' not found.")
            return
        # Make sure (older, newer) is in chronological order even if the user
        # passed them backwards — diffing always reads "older → newer".
        if older["captured_at"] > newer["captured_at"]:
            older, newer = newer, older
    else:
        older, newer = gex_store.latest_two(
            ticker=ticker, expiration_date=expiration
        )
        if not older or not newer:
            click.echo(
                f"Need at least 2 snapshots for {ticker} {expiration}. "
                "Run `lia greeks` twice (use --refresh-cache for fresh data)."
            )
            return

    # Header
    click.echo(
        f"\n{ticker} {expiration}  "
        f"{older['captured_at']}  →  {newer['captured_at']}"
    )
    dt = _iso_diff_minutes(older["captured_at"], newer["captured_at"])
    if dt is not None:
        click.echo(f"Elapsed: {dt:.0f} min")

    # Spot
    d_spot = newer["spot_price"] - older["spot_price"]
    pct = (d_spot / older["spot_price"] * 100) if older["spot_price"] else 0.0
    click.echo(
        f"\nSpot: {older['spot_price']:.2f}  →  {newer['spot_price']:.2f}  "
        f"({_sign(d_spot)}{d_spot:.2f}, {_sign(d_spot)}{pct:.2f}%)"
    )

    # Headline: intraday "more calls vs more puts being added" — by VOLUME
    click.echo("\nIntraday signal (volume — cumulative since open):")
    _print_pair(
        "  Calls",
        older["call_volume"], newer["call_volume"],
        is_int=True,
    )
    _print_pair(
        "  Puts ",
        older["put_volume"], newer["put_volume"],
        is_int=True,
    )
    pc_old = (older["put_volume"] / older["call_volume"]) if older["call_volume"] else None
    pc_new = (newer["put_volume"] / newer["call_volume"]) if newer["call_volume"] else None
    if pc_old is not None and pc_new is not None:
        click.echo(
            f"  P/C vol ratio: {pc_old:.2f}  →  {pc_new:.2f}  "
            f"({_sign(pc_new - pc_old)}{pc_new - pc_old:.2f})"
        )

    # OI — meaningful day-over-day only
    d_call_oi = newer["call_oi"] - older["call_oi"]
    d_put_oi = newer["put_oi"] - older["put_oi"]
    click.echo("\nDay-over-day signal (open interest — updates EOD):")
    _print_pair("  Calls", older["call_oi"], newer["call_oi"], is_int=True)
    _print_pair("  Puts ", older["put_oi"], newer["put_oi"], is_int=True)
    if d_call_oi == 0 and d_put_oi == 0:
        click.echo("  (ΔOI = 0 — expected within same trading day)")

    # Notional delta — dollar exposure being added
    click.echo("\nNotional delta ($, derived from OI × delta × 100 × spot):")
    _print_pair(
        "  Calls", older["call_notional_delta"], newer["call_notional_delta"],
        is_money=True,
    )
    _print_pair(
        "  Puts ", older["put_notional_delta"], newer["put_notional_delta"],
        is_money=True,
    )

    # GEX
    click.echo("\nGEX:")
    _print_pair(
        "  Net  ", older["total_gex"], newer["total_gex"], is_money=True
    )
    _print_pair(
        "  Calls", older["call_gex"], newer["call_gex"], is_money=True
    )
    _print_pair(
        "  Puts ", older["put_gex"], newer["put_gex"], is_money=True
    )

    if top > 0:
        _print_strike_movers(older["id"], newer["id"], top)


def _resolve_snapshot_ref(ref, *, ticker, expiration):
    """Resolve a user-supplied --from/--to value to a snapshot row.

    Accepts:
      - integer / digit string  → N-th most recent (1 = latest)
      - 'first'                 → earliest snapshot for (ticker, expiration)
      - 'last' / 'latest'       → most recent
      - 26-char ULID            → direct lookup by id
    """
    if ref is None:
        return None
    ref_str = str(ref).strip()
    if ref_str.lower() == "first":
        return gex_store.first_snapshot(
            ticker=ticker, expiration_date=expiration
        )
    if ref_str.lower() in ("last", "latest"):
        return gex_store.nth_most_recent(
            ticker=ticker, expiration_date=expiration, n=1
        )
    if ref_str.isdigit():
        return gex_store.nth_most_recent(
            ticker=ticker, expiration_date=expiration, n=int(ref_str)
        )
    return gex_store.get_snapshot(ref_str)


def _iso_diff_minutes(a, b):
    try:
        return (
            datetime.fromisoformat(b.replace("Z", "+00:00"))
            - datetime.fromisoformat(a.replace("Z", "+00:00"))
        ).total_seconds() / 60.0
    except Exception:
        return None


def _sign(x):
    if x is None:
        return ""
    return "+" if x >= 0 else ""


def _print_pair(label, old, new, *, is_int=False, is_money=False):
    if old is None or new is None:
        click.echo(f"{label}: n/a")
        return
    d = new - old
    if is_money:
        click.echo(
            f"{label}: {old:>15,.0f}  →  {new:>15,.0f}  "
            f"(Δ {_sign(d)}{d:,.0f})"
        )
    elif is_int:
        click.echo(
            f"{label}: {old:>10,d}  →  {new:>10,d}  (Δ {_sign(d)}{d:,d})"
        )
    else:
        click.echo(
            f"{label}: {old:>10.2f}  →  {new:>10.2f}  (Δ {_sign(d)}{d:.2f})"
        )


def _print_strike_movers(older_id, newer_id, top):
    """Top-N per-strike movers by Δvolume, split by call/put."""
    older_strikes = {
        (r["option_type"], r["strike_price"]): r
        for r in gex_store.get_strikes(older_id)
    }
    newer_strikes = {
        (r["option_type"], r["strike_price"]): r
        for r in gex_store.get_strikes(newer_id)
    }
    keys = set(older_strikes) | set(newer_strikes)
    movers = []
    for k in keys:
        o = older_strikes.get(k)
        n = newer_strikes.get(k)
        o_vol = (o["volume"] if o and o["volume"] is not None else 0)
        n_vol = (n["volume"] if n and n["volume"] is not None else 0)
        o_oi = (o["open_interest"] if o and o["open_interest"] is not None else 0)
        n_oi = (n["open_interest"] if n and n["open_interest"] is not None else 0)
        movers.append(
            {
                "type": k[0],
                "strike": k[1],
                "d_vol": n_vol - o_vol,
                "d_oi": n_oi - o_oi,
                "vol_now": n_vol,
                "oi_now": n_oi,
            }
        )

    for side in ("call", "put"):
        side_movers = [m for m in movers if m["type"] == side and m["d_vol"] != 0]
        if not side_movers:
            continue
        side_movers.sort(key=lambda m: abs(m["d_vol"]), reverse=True)
        click.echo(f"\nTop {side} Δvolume:")
        df = pd.DataFrame(side_movers[:top])
        click.echo(df.to_string(index=False))


@cli.command("oi-value")
@click.option(
    "--ticker", default="SPY", help="Ticker symbol (default: SPY)"
)
@click.option(
    "--expiration", required=True, help="Expiration date YYYY-MM-DD (required)"
)
@click.option(
    "--rate", default=0.02, help="Risk-free rate (default: 0.02)"
)
@click.option(
    "--range", "range_pct", default=5.0, type=float,
    help="Only include strikes within +/- N percent of spot (default: 5). "
    "Pass 0 to include the whole chain.",
)
@click.option(
    "--out", default=None,
    help="Output PNG path (default: plots/oi-value-{ticker}-{expiration}[-cumulative].png)",
)
@click.option(
    "--cumulative", is_flag=True,
    help="Plot cumulative sums (running total by strike) instead of per-strike bars.",
)
@click.option(
    "--metric", type=click.Choice(["dollars", "contracts"]), default="dollars",
    show_default=True,
    help="Y-axis metric: 'dollars' = market_price × OI × 100, "
    "'contracts' = raw open interest count.",
)
@click.option(
    "--no-open", is_flag=True, help="Don't auto-open the PNG when done",
)
@click.option("--no-cache", is_flag=True, help="Disable options-chain cache")
@click.option("--refresh-cache", is_flag=True, help="Force refresh options-chain cache")
@ensure_logged_in
def oi_value(
    ticker, expiration, rate, range_pct, out, cumulative, metric,
    no_open, no_cache, refresh_cache,
):
    """Plot dollar OI or raw open interest per strike.

    `--metric dollars` (default): market_price × open_interest × 100 — a
    "money profile" showing where dollar exposure sits at each strike.

    `--metric contracts`: raw open-interest count per strike. Useful for
    spotting positioning "walls" independent of premium level (great for
    dated options where premium is tiny but OI walls still pin price).

    Bar mode: calls point up, puts point down.

    With `--cumulative`: plots the running total, direction aligned to each
    side's payoff:
      - Calls cumulate low→high: at strike X, y = Σ(strike ≤ X).
      - Puts  cumulate high→low: at strike X, y = Σ(strike ≥ X).
    Both curves peak toward spot and fall off toward their OTM tail.
    """
    try:
        datetime.strptime(expiration, "%Y-%m-%d")
    except ValueError:
        click.echo(f"❌ Invalid --expiration '{expiration}'. Use YYYY-MM-DD.")
        return

    # Deferred import so the CLI stays fast when this command isn't used
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = calculate_options_greeks(
        ticker, expiration, rate,
        use_cache=not no_cache, refresh_cache=refresh_cache,
    )
    if df.empty:
        click.echo("No options data found.")
        return

    # Always fetch a fresh spot price. The options chain may come from cache
    # (which is fine for OI/premium — it doesn't move much intraday), but the
    # spot line on the chart MUST reflect right-now.
    spot = float(df.attrs.get("stock_price") or 0.0)
    try:
        fresh_spot_list = rh.stocks.get_latest_price(
            ticker, priceType=None, includeExtendedHours=True
        )
        if fresh_spot_list and fresh_spot_list[0]:
            spot = float(fresh_spot_list[0])
    except Exception as e:
        click.echo(f"⚠️  Could not refresh spot ({e}); using cached value.")
    if spot <= 0:
        click.echo("Could not determine spot price.")
        return

    # Figure out how stale the options data itself is.
    cached_at_iso = df.attrs.get("cached_calculated_at")
    cache_age_min = None
    data_as_of_local = None
    if cached_at_iso:
        try:
            cached_dt = datetime.fromisoformat(cached_at_iso)
            cache_age_min = (datetime.now() - cached_dt).total_seconds() / 60
            data_as_of_local = cached_dt.strftime("%Y-%m-%d %H:%M")
        except ValueError:
            pass

    fig, ax = plt.subplots(figsize=(14, 7))
    scope = f"±{range_pct:.0f}% of spot" if range_pct > 0 else "full chain"
    as_of = (
        f"OI as of {data_as_of_local}"
        if data_as_of_local
        else f"OI as of {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )

    result = _render_oi_panel(
        ax, df=df, ticker=ticker, expiration=expiration, spot=spot,
        metric=metric, cumulative=cumulative, range_pct=range_pct,
        scope=scope, as_of=as_of,
    )

    if result["empty"]:
        click.echo(
            f"No strikes within ±{range_pct}% of ${spot:.2f}. "
            "Try a larger --range."
        )
        plt.close(fig)
        return

    plt.tight_layout()

    if not out:
        os.makedirs("plots", exist_ok=True)
        cum_suffix = "-cumulative" if cumulative else ""
        metric_suffix = "" if metric == "dollars" else "-contracts"
        out = (
            f"plots/oi-value-{ticker}-{expiration}"
            f"{cum_suffix}{metric_suffix}.png"
        )
    plt.savefig(out, dpi=120)
    plt.close(fig)

    total_call = result["total_call"]
    total_put = result["total_put"]
    fmt_total = result["fmt_total"]
    fmt_point = result["fmt_point"]
    crossing_x = result["crossing_x"]
    crossing_y = result["crossing_y"]

    click.echo(f"📊 Saved: {out}")
    click.echo(
        f"    Spot ${spot:.2f} (fresh) | Call {fmt_total(total_call)} | "
        f"Put {fmt_total(total_put)} | strikes plotted: {result['strikes_count']}"
    )
    if cache_age_min is not None:
        stale_hint = (
            "  ⚠️  pass --refresh-cache for live premium/OI"
            if cache_age_min >= 15
            else ""
        )
        click.echo(
            f"    OI/premium data age: {cache_age_min:,.0f} min "
            f"(cached at {data_as_of_local}){stale_hint}"
        )
    if cumulative and crossing_x is not None:
        rel = (crossing_x - spot) / spot * 100 if spot else 0.0
        click.echo(
            f"    Crossing strike: ${crossing_x:.2f}  "
            f"({fmt_point(crossing_y)} each, {rel:+.2f}% vs spot)"
        )
    if not no_open:
        try:
            import subprocess
            subprocess.run(["open", out], check=False)
        except Exception:
            pass


def _find_curve_crossing(strikes, call_cum, put_cum):
    """Return (strike, y) where the ascending call cumsum crosses the
    descending put cumsum, interpolated linearly between the two bracketing
    strike points. Returns (None, None) if no crossing exists.
    """
    call_arr = np.asarray(call_cum, dtype=float)
    put_arr = np.asarray(put_cum, dtype=float)
    strike_arr = np.asarray(strikes, dtype=float)

    diff = call_arr - put_arr
    if len(diff) < 2:
        return None, None
    # Detect the first index i where sign(diff) flips between i and i+1.
    sign = np.sign(diff)
    flips = np.where(sign[:-1] != sign[1:])[0]
    if len(flips) == 0:
        return None, None
    i = int(flips[0])
    x1, x2 = float(strike_arr[i]), float(strike_arr[i + 1])
    d1, d2 = float(diff[i]), float(diff[i + 1])
    if d2 == d1:
        return x1, float(call_arr[i])
    t = -d1 / (d2 - d1)
    cross_x = x1 + t * (x2 - x1)
    cross_y = float(call_arr[i]) + t * (float(call_arr[i + 1]) - float(call_arr[i]))
    return cross_x, cross_y


def _prepare_oi_series(df, spot, metric, range_pct):
    """Compute per-strike sums for OI plotting.

    Filters by spot ± ``range_pct``%, computes the chosen metric per
    strike, and returns everything both single-panel and diff renderers
    need. Returns ``None`` if no strikes fall within range.
    """
    df = df.copy()
    if metric == "dollars":
        df["metric_value"] = (
            df["market_price"].fillna(0).astype(float)
            * df["open_interest"].fillna(0).astype(float)
            * 100.0
        )
    else:
        df["metric_value"] = df["open_interest"].fillna(0).astype(float)

    if range_pct > 0:
        lo = spot * (1 - range_pct / 100)
        hi = spot * (1 + range_pct / 100)
        df = df[(df["strike_price"] >= lo) & (df["strike_price"] <= hi)]

    if df.empty:
        return None

    calls = (
        df[df["option_type"].str.upper() == "CALL"]
        .groupby("strike_price")["metric_value"].sum()
    )
    puts = (
        df[df["option_type"].str.upper() == "PUT"]
        .groupby("strike_price")["metric_value"].sum()
    )
    strikes = sorted(set(calls.index) | set(puts.index))
    calls_by_strike = {float(k): float(v) for k, v in calls.items()}
    puts_by_strike = {float(k): float(v) for k, v in puts.items()}
    call_vals = [calls_by_strike.get(k, 0.0) for k in strikes]
    put_vals_positive = [puts_by_strike.get(k, 0.0) for k in strikes]
    put_vals_negative = [-v for v in put_vals_positive]

    return {
        "strikes": strikes,
        "call_vals": call_vals,
        "put_vals_positive": put_vals_positive,
        "put_vals_negative": put_vals_negative,
        "calls_by_strike": calls_by_strike,
        "puts_by_strike": puts_by_strike,
        "total_call": sum(call_vals),
        "total_put": sum(put_vals_positive),
    }


def _oi_metric_style(metric, ticker, expiration, scope):
    """Formatters, labels, and titles for one metric.

    Kept out of `_render_oi_panel` so the diff renderer can reuse it
    without importing plotting internals.
    """
    import matplotlib.pyplot as plt

    if metric == "dollars":
        return {
            "y_fmt_signed": plt.FuncFormatter(lambda x, _: f"${x/1e6:,.1f}M"),
            "y_fmt_abs": plt.FuncFormatter(lambda x, _: f"${abs(x)/1e6:,.1f}M"),
            "fmt_total": lambda v: f"${v/1e6:,.1f}M",
            "fmt_point": lambda v: f"${v/1e6:,.2f}M",
            "y_label_bar": (
                "Market Price × OI × 100  ($, puts shown negative)"
            ),
            "y_label_cum": "Cumulative Market Price × OI × 100  ($)",
            "title_bar": (
                f"{ticker} {expiration} — Dollar OI per strike  ({scope})"
            ),
            "title_cum": (
                f"{ticker} {expiration} — Cumulative Dollar OI  "
                f"(calls low→high, puts high→low)  ({scope})"
            ),
        }
    return {
        "y_fmt_signed": plt.FuncFormatter(lambda x, _: f"{x:,.0f}"),
        "y_fmt_abs": plt.FuncFormatter(lambda x, _: f"{abs(x):,.0f}"),
        "fmt_total": lambda v: f"{v:,.0f} contracts",
        "fmt_point": lambda v: f"{v:,.0f}",
        "y_label_bar": "Open Interest  (contracts, puts shown negative)",
        "y_label_cum": "Cumulative Open Interest  (contracts)",
        "title_bar": (
            f"{ticker} {expiration} — Open Interest per strike  ({scope})"
        ),
        "title_cum": (
            f"{ticker} {expiration} — Cumulative Open Interest  "
            f"(calls low→high, puts high→low)  ({scope})"
        ),
    }


def _bar_width(strikes, fraction=0.8):
    step = float(np.min(np.diff(strikes))) if len(strikes) > 1 else 1.0
    if step <= 0:
        step = 1.0
    return step * fraction, step


def _apply_strike_ticks(ax, strikes):
    """Set finer x-axis ticks so individual strikes are easy to read.

    Chooses major/minor tick intervals based on the strike step so any
    strike price can be located without counting bars. Draws a faint
    dotted grid at the minor gridlines so the chart still reads clean.
    """
    from matplotlib.ticker import MultipleLocator

    if len(strikes) < 2:
        return
    step = float(np.min(np.diff(strikes)))
    if step <= 0:
        return
    if step <= 1:
        major, minor = 5, 1
    elif step <= 2.5:
        major, minor = 10, step
    else:
        major, minor = step * 5, step
    ax.xaxis.set_major_locator(MultipleLocator(major))
    ax.xaxis.set_minor_locator(MultipleLocator(minor))
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", alpha=0.15, linestyle=":")
    ax.tick_params(axis="x", which="minor", length=3)


def _load_oi_snapshot_df(snapshot_row):
    """Reconstruct a DataFrame from a stored snapshot.

    Returned columns match what `_prepare_oi_series` expects
    (``strike_price``, ``option_type``, ``market_price``,
    ``open_interest``). Returns ``None`` if the snapshot has no strikes.
    """
    rows = gex_store.get_strikes(snapshot_row["id"])
    if not rows:
        return None
    data = [
        {
            "strike_price": float(r["strike_price"]),
            "option_type": str(r["option_type"]).lower(),
            "market_price": (
                float(r["mark_price"]) if r["mark_price"] is not None else 0.0
            ),
            "open_interest": (
                int(r["open_interest"])
                if r["open_interest"] is not None
                else 0
            ),
        }
        for r in rows
    ]
    return pd.DataFrame(data)


def _local_ts(iso_ts):
    """Format an ISO-UTC timestamp for a chart title/legend (local wall time)."""
    if not iso_ts:
        return "?"
    try:
        dt = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
        return dt.astimezone().strftime("%Y-%m-%d %H:%M")
    except Exception:
        return iso_ts


def _render_oi_panel(
    ax,
    *,
    df,
    ticker,
    expiration,
    spot,
    metric,
    cumulative,
    range_pct,
    scope,
    as_of,
    show_title=True,
):
    """Render one OI panel into a matplotlib axes.

    Shared between `oi-value` (single-chart) and `oi-dashboard`
    (multi-panel). Returns a dict with computed totals so the caller
    can echo a summary.
    """
    series = _prepare_oi_series(df, spot, metric, range_pct)
    if series is None:
        return {
            "empty": True, "strikes_count": 0,
            "total_call": 0.0, "total_put": 0.0,
            "crossing_x": None, "crossing_y": None,
            "fmt_total": lambda v: str(v), "fmt_point": lambda v: str(v),
        }

    style = _oi_metric_style(metric, ticker, expiration, scope)
    strikes = series["strikes"]
    call_vals = series["call_vals"]
    put_vals_positive = series["put_vals_positive"]
    put_vals_negative = series["put_vals_negative"]
    total_call = series["total_call"]
    total_put = series["total_put"]
    fmt_total = style["fmt_total"]
    fmt_point = style["fmt_point"]

    bar_width, _ = _bar_width(strikes)

    crossing_x, crossing_y = None, None
    if cumulative:
        call_cum = np.cumsum(call_vals)
        put_arr = np.array(put_vals_positive)
        put_cum = np.cumsum(put_arr[::-1])[::-1]

        crossing_x, crossing_y = _find_curve_crossing(
            strikes, call_cum, put_cum
        )

        ax.step(
            strikes, call_cum, where="post", color="#2ca02c",
            linewidth=2.2,
            label=f"Call cum ≤ strike  (total {fmt_total(total_call)})",
        )
        ax.step(
            strikes, put_cum, where="post", color="#d62728",
            linewidth=2.2,
            label=f"Put  cum ≥ strike  (total {fmt_total(total_put)})",
        )
        ax.axvline(
            spot, color="black", linestyle="--", linewidth=1.2,
            label=f"Spot ${spot:.2f}",
        )
        if crossing_x is not None:
            ax.axvline(
                crossing_x, color="#1f77b4", linestyle=":", linewidth=1.6,
                label=(
                    f"Crossing ${crossing_x:.2f}  "
                    f"({fmt_point(crossing_y)})"
                ),
            )
            ax.plot(
                [crossing_x], [crossing_y], marker="o", markersize=8,
                markerfacecolor="#1f77b4", markeredgecolor="white",
                markeredgewidth=1.5, zorder=5,
            )
            ax.annotate(
                f"${crossing_x:.2f}",
                xy=(crossing_x, crossing_y),
                xytext=(8, 10), textcoords="offset points",
                fontsize=10, fontweight="bold", color="#1f77b4",
            )
        ax.set_xlabel("Strike price ($)")
        ax.set_ylabel(style["y_label_cum"])
        if show_title:
            ax.set_title(
                f"{style['title_cum']}\nSpot ${spot:.2f}   ·   {as_of}"
            )
        ax.yaxis.set_major_formatter(style["y_fmt_signed"])
    else:
        ax.bar(
            strikes, call_vals,
            width=bar_width, color="#2ca02c", alpha=0.85,
            label=f"Call  (total {fmt_total(total_call)})",
        )
        ax.bar(
            strikes, put_vals_negative,
            width=bar_width, color="#d62728", alpha=0.85,
            label=f"Put  (total {fmt_total(total_put)})",
        )
        ax.axvline(
            spot, color="black", linestyle="--", linewidth=1.2,
            label=f"Spot ${spot:.2f}",
        )
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_xlabel("Strike price ($)")
        ax.set_ylabel(style["y_label_bar"])
        if show_title:
            ax.set_title(
                f"{style['title_bar']}\nSpot ${spot:.2f}   ·   {as_of}"
            )
        ax.yaxis.set_major_formatter(style["y_fmt_abs"])

    ax.legend(loc="upper left")
    _apply_strike_ticks(ax, strikes)

    return {
        "empty": False,
        "strikes_count": len(strikes),
        "total_call": total_call,
        "total_put": total_put,
        "crossing_x": crossing_x,
        "crossing_y": crossing_y,
        "fmt_total": fmt_total,
        "fmt_point": fmt_point,
    }


def _render_oi_diff_panel(
    ax,
    *,
    df_old,
    df_new,
    ticker,
    expiration,
    spot_old,
    spot_new,
    old_label,
    new_label,
    metric,
    cumulative,
    range_pct,
    scope,
    show_title=True,
):
    """Overlay two snapshots into one panel: old (faded) + new (bold).

    Cumulative panels overlay call/put curves (dashed = old, solid = new).
    Bar panels use grouped bars — old on the left of each strike center,
    new on the right — so per-strike deltas are directly visible.

    Returns a dict summarizing both snapshots plus the crossing shift.
    """
    # For per-strike alignment we compute both series against the *newer*
    # spot's window (so both panels show the same strike set). This keeps
    # the x-axis consistent across old/new even if spot moved a lot.
    old_series = _prepare_oi_series(df_old, spot_new, metric, range_pct)
    new_series = _prepare_oi_series(df_new, spot_new, metric, range_pct)
    if old_series is None or new_series is None:
        return {"empty": True}

    style = _oi_metric_style(metric, ticker, expiration, scope)
    fmt_total = style["fmt_total"]
    fmt_point = style["fmt_point"]

    strikes_union = sorted(
        set(old_series["strikes"]) | set(new_series["strikes"])
    )
    old_calls = old_series["calls_by_strike"]
    new_calls = new_series["calls_by_strike"]
    old_puts = old_series["puts_by_strike"]
    new_puts = new_series["puts_by_strike"]

    old_total_c = old_series["total_call"]
    old_total_p = old_series["total_put"]
    new_total_c = new_series["total_call"]
    new_total_p = new_series["total_put"]

    call_color = "#2ca02c"
    put_color = "#d62728"

    crossing_old = crossing_new = (None, None)

    if cumulative:
        call_cum_old = np.cumsum([old_calls.get(k, 0.0) for k in strikes_union])
        call_cum_new = np.cumsum([new_calls.get(k, 0.0) for k in strikes_union])
        put_arr_old = np.array(
            [old_puts.get(k, 0.0) for k in strikes_union]
        )
        put_arr_new = np.array(
            [new_puts.get(k, 0.0) for k in strikes_union]
        )
        put_cum_old = np.cumsum(put_arr_old[::-1])[::-1]
        put_cum_new = np.cumsum(put_arr_new[::-1])[::-1]

        crossing_old = _find_curve_crossing(
            strikes_union, call_cum_old, put_cum_old
        )
        crossing_new = _find_curve_crossing(
            strikes_union, call_cum_new, put_cum_new
        )

        ax.step(
            strikes_union, call_cum_old, where="post", color=call_color,
            linewidth=1.6, linestyle="--", alpha=0.55,
            label=(
                f"Call cum · {old_label}  ({fmt_total(old_total_c)})"
            ),
        )
        ax.step(
            strikes_union, put_cum_old, where="post", color=put_color,
            linewidth=1.6, linestyle="--", alpha=0.55,
            label=(
                f"Put  cum · {old_label}  ({fmt_total(old_total_p)})"
            ),
        )
        ax.step(
            strikes_union, call_cum_new, where="post", color=call_color,
            linewidth=2.4,
            label=(
                f"Call cum · {new_label}  ({fmt_total(new_total_c)})"
            ),
        )
        ax.step(
            strikes_union, put_cum_new, where="post", color=put_color,
            linewidth=2.4,
            label=(
                f"Put  cum · {new_label}  ({fmt_total(new_total_p)})"
            ),
        )
        ax.axvline(
            spot_old, color="gray", linestyle=":", linewidth=1.2,
            alpha=0.7,
            label=f"Spot (old) ${spot_old:.2f}",
        )
        ax.axvline(
            spot_new, color="black", linestyle="--", linewidth=1.4,
            label=f"Spot (new) ${spot_new:.2f}",
        )
        if crossing_new[0] is not None:
            cx, cy = crossing_new
            ax.plot(
                [cx], [cy], marker="o", markersize=8,
                markerfacecolor="#1f77b4", markeredgecolor="white",
                markeredgewidth=1.5, zorder=5,
            )
            ax.annotate(
                f"${cx:.2f}",
                xy=(cx, cy),
                xytext=(8, 10), textcoords="offset points",
                fontsize=10, fontweight="bold", color="#1f77b4",
            )
        ax.set_xlabel("Strike price ($)")
        ax.set_ylabel(style["y_label_cum"])
        if show_title:
            ax.set_title(
                f"{style['title_cum']}\n"
                f"{old_label} → {new_label}"
            )
        ax.yaxis.set_major_formatter(style["y_fmt_signed"])
    else:
        # Grouped bars: shift old left, new right, each half-width.
        bar_width, step = _bar_width(strikes_union)
        half = bar_width / 2.0
        x = np.array(strikes_union, dtype=float)

        old_call_vals = np.array([old_calls.get(k, 0.0) for k in strikes_union])
        new_call_vals = np.array([new_calls.get(k, 0.0) for k in strikes_union])
        old_put_vals = -np.array([old_puts.get(k, 0.0) for k in strikes_union])
        new_put_vals = -np.array([new_puts.get(k, 0.0) for k in strikes_union])

        ax.bar(
            x - half / 2, old_call_vals,
            width=half, color=call_color, alpha=0.35,
            label=f"Call · {old_label}  ({fmt_total(old_total_c)})",
        )
        ax.bar(
            x + half / 2, new_call_vals,
            width=half, color=call_color, alpha=0.9,
            label=f"Call · {new_label}  ({fmt_total(new_total_c)})",
        )
        ax.bar(
            x - half / 2, old_put_vals,
            width=half, color=put_color, alpha=0.35,
            label=f"Put · {old_label}  ({fmt_total(old_total_p)})",
        )
        ax.bar(
            x + half / 2, new_put_vals,
            width=half, color=put_color, alpha=0.9,
            label=f"Put · {new_label}  ({fmt_total(new_total_p)})",
        )
        ax.axvline(
            spot_old, color="gray", linestyle=":", linewidth=1.2,
            alpha=0.7,
            label=f"Spot (old) ${spot_old:.2f}",
        )
        ax.axvline(
            spot_new, color="black", linestyle="--", linewidth=1.4,
            label=f"Spot (new) ${spot_new:.2f}",
        )
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_xlabel("Strike price ($)")
        ax.set_ylabel(style["y_label_bar"])
        if show_title:
            ax.set_title(
                f"{style['title_bar']}\n"
                f"{old_label} → {new_label}"
            )
        ax.yaxis.set_major_formatter(style["y_fmt_abs"])

    ax.legend(loc="upper left", fontsize=8)
    _apply_strike_ticks(ax, strikes_union)

    return {
        "empty": False,
        "strikes_count": len(strikes_union),
        "old_total_call": old_total_c,
        "old_total_put": old_total_p,
        "new_total_call": new_total_c,
        "new_total_put": new_total_p,
        "delta_total_call": new_total_c - old_total_c,
        "delta_total_put": new_total_p - old_total_p,
        "crossing_old": crossing_old,
        "crossing_new": crossing_new,
        "fmt_total": fmt_total,
        "fmt_point": fmt_point,
    }


@cli.command("oi-dashboard")
@click.option(
    "--ticker", default="SPY", help="Ticker symbol (default: SPY)"
)
@click.option(
    "--expiration", required=True, help="Expiration date YYYY-MM-DD (required)"
)
@click.option(
    "--rate", default=0.02, help="Risk-free rate (default: 0.02)"
)
@click.option(
    "--range", "range_pct", default=5.0, type=float,
    help="Only include strikes within +/- N percent of spot (default: 5). "
    "Pass 0 to include the whole chain.",
)
@click.option(
    "--out", default=None,
    help="Output PNG path (default: plots/oi-dashboard-{ticker}-{expiration}.png)",
)
@click.option(
    "--no-open", is_flag=True, help="Don't auto-open the PNG when done",
)
@click.option("--no-cache", is_flag=True, help="Disable options-chain cache")
@click.option(
    "--refresh-cache", is_flag=True,
    help="Force refresh options-chain cache",
)
@click.option(
    "--no-snapshot", is_flag=True,
    help="Skip persisting this run to the snapshot DB (default: auto-snapshot).",
)
@ensure_logged_in
def oi_dashboard(
    ticker, expiration, rate, range_pct, out, no_open, no_cache, refresh_cache,
    no_snapshot,
):
    """Three-panel OI dashboard for a single expiration, stacked vertically.

    Panels (top → bottom):
      1. Cumulative Dollar OI  (with crossing strike annotated)
      2. Open Interest per strike (contracts)  — raw positioning walls
      3. Dollar OI per strike  (market_price × OI × 100)

    Same underlying data as three separate `lia oi-value` calls, but
    rendered into one PNG so you can eyeball dollars vs contracts vs
    cumulative pin lines side-by-side.

    Each run is automatically persisted to the snapshot DB so you can
    later diff two runs with `lia oi-dashboard-diff`. Pass `--no-snapshot`
    to skip.
    """
    try:
        datetime.strptime(expiration, "%Y-%m-%d")
    except ValueError:
        click.echo(f"❌ Invalid --expiration '{expiration}'. Use YYYY-MM-DD.")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = calculate_options_greeks(
        ticker, expiration, rate,
        use_cache=not no_cache, refresh_cache=refresh_cache,
    )
    if df.empty:
        click.echo("No options data found.")
        return

    spot = float(df.attrs.get("stock_price") or 0.0)
    try:
        fresh_spot_list = rh.stocks.get_latest_price(
            ticker, priceType=None, includeExtendedHours=True
        )
        if fresh_spot_list and fresh_spot_list[0]:
            spot = float(fresh_spot_list[0])
    except Exception as e:
        click.echo(f"⚠️  Could not refresh spot ({e}); using cached value.")
    if spot <= 0:
        click.echo("Could not determine spot price.")
        return

    # Propagate the fresh spot back into df.attrs so the snapshot writer
    # (which reads df.attrs["stock_price"]) records the right value.
    df.attrs["stock_price"] = spot

    cached_at_iso = df.attrs.get("cached_calculated_at")
    cache_age_min = None
    data_as_of_local = None
    if cached_at_iso:
        try:
            cached_dt = datetime.fromisoformat(cached_at_iso)
            cache_age_min = (datetime.now() - cached_dt).total_seconds() / 60
            data_as_of_local = cached_dt.strftime("%Y-%m-%d %H:%M")
        except ValueError:
            pass

    scope = f"±{range_pct:.0f}% of spot" if range_pct > 0 else "full chain"
    as_of = (
        f"OI as of {data_as_of_local}"
        if data_as_of_local
        else f"OI as of {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )

    panels = [
        {"metric": "dollars", "cumulative": True},
        {"metric": "contracts", "cumulative": False},
        {"metric": "dollars", "cumulative": False},
    ]

    fig, axes = plt.subplots(3, 1, figsize=(14, 20), sharex=True)
    results = []
    any_empty = False
    for ax, panel in zip(axes, panels):
        result = _render_oi_panel(
            ax, df=df, ticker=ticker, expiration=expiration, spot=spot,
            metric=panel["metric"], cumulative=panel["cumulative"],
            range_pct=range_pct, scope=scope, as_of=as_of,
            show_title=True,
        )
        results.append(result)
        if result["empty"]:
            any_empty = True

    if any_empty:
        click.echo(
            f"No strikes within ±{range_pct}% of ${spot:.2f}. "
            "Try a larger --range."
        )
        plt.close(fig)
        return

    fig.suptitle(
        f"{ticker} {expiration} — OI Dashboard   ·   "
        f"Spot ${spot:.2f}   ·   {as_of}",
        fontsize=14, fontweight="bold", y=0.995,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.985))

    if not out:
        os.makedirs("plots", exist_ok=True)
        out = f"plots/oi-dashboard-{ticker}-{expiration}.png"
    plt.savefig(out, dpi=120)
    plt.close(fig)

    click.echo(f"📊 Saved: {out}")
    labels = ["Dollar cum", "Contracts bar", "Dollar bar"]
    for label, result in zip(labels, results):
        click.echo(
            f"    {label:<14} | Call {result['fmt_total'](result['total_call'])} "
            f"| Put {result['fmt_total'](result['total_put'])}"
        )
    cum_result = results[0]
    if cum_result["crossing_x"] is not None:
        rel = (cum_result["crossing_x"] - spot) / spot * 100 if spot else 0.0
        click.echo(
            f"    Crossing strike (dollar cum): ${cum_result['crossing_x']:.2f}  "
            f"({cum_result['fmt_point'](cum_result['crossing_y'])} each, "
            f"{rel:+.2f}% vs spot)"
        )
    if cache_age_min is not None:
        stale_hint = (
            "  ⚠️  pass --refresh-cache for live premium/OI"
            if cache_age_min >= 15
            else ""
        )
        click.echo(
            f"    OI/premium data age: {cache_age_min:,.0f} min "
            f"(cached at {data_as_of_local}){stale_hint}"
        )

    if not no_snapshot:
        try:
            sid = _record_greeks_snapshot(df, ticker, expiration, rate)
            click.echo(f"    📸 Snapshot recorded: {sid}")
        except Exception as e:
            click.echo(f"    ⚠️  Snapshot failed: {e}")

    if not no_open:
        try:
            import subprocess
            subprocess.run(["open", out], check=False)
        except Exception:
            pass


@cli.command("oi-dashboard-diff")
@click.option(
    "--ticker", default="SPY", help="Ticker symbol (default: SPY)"
)
@click.option(
    "--expiration", required=True,
    help="Expiration date YYYY-MM-DD (required)",
)
@click.option(
    "--from", "from_ref", default=None,
    help="Older snapshot ref: ULID, 'first', 'latest', or N-th-most-recent "
    "(e.g. 2 = second-most-recent). Defaults to the second-most-recent.",
)
@click.option(
    "--to", "to_ref", default=None,
    help="Newer snapshot ref (same syntax as --from). Defaults to 'latest'.",
)
@click.option(
    "--range", "range_pct", default=5.0, type=float,
    help="Only include strikes within +/- N percent of the newer spot "
    "(default: 5). Pass 0 to include the whole chain.",
)
@click.option(
    "--out", default=None,
    help="Output PNG path "
    "(default: plots/oi-dashboard-diff-{ticker}-{expiration}.png)",
)
@click.option(
    "--no-open", is_flag=True, help="Don't auto-open the PNG when done",
)
def oi_dashboard_diff(ticker, expiration, from_ref, to_ref, range_pct, out, no_open):
    """Overlay two `oi-dashboard` snapshots to see what shifted between them.

    Same three panels as `oi-dashboard`, but each panel overlays two
    point-in-time snapshots:

      • Older = faded (dashed lines / translucent bars)
      • Newer = bold (solid lines / opaque bars)

    Defaults to comparing the two most recent snapshots for the given
    (ticker, expiration). Use `--from` and `--to` to pin a specific pair
    (accepts ULIDs, 'first', 'latest', or N-th-most-recent integers).

    Snapshots come from the DB written by `lia oi-dashboard` and
    `lia greeks --show-gex`. Run `lia greeks-history` to inspect them.
    """
    try:
        datetime.strptime(expiration, "%Y-%m-%d")
    except ValueError:
        click.echo(f"❌ Invalid --expiration '{expiration}'. Use YYYY-MM-DD.")
        return

    if from_ref is None and to_ref is None:
        older, newer = gex_store.latest_two(
            ticker=ticker, expiration_date=expiration
        )
        if older is None or newer is None:
            click.echo(
                f"❌ Need at least two snapshots for {ticker} {expiration}. "
                f"Run `lia oi-dashboard` twice first."
            )
            return
    else:
        newer = _resolve_snapshot_ref(
            to_ref or "latest", ticker=ticker, expiration=expiration
        )
        older = _resolve_snapshot_ref(
            from_ref or "2", ticker=ticker, expiration=expiration
        )
        if newer is None:
            click.echo(f"❌ Could not resolve --to '{to_ref or 'latest'}'.")
            return
        if older is None:
            click.echo(f"❌ Could not resolve --from '{from_ref or '2'}'.")
            return

    if older["id"] == newer["id"]:
        click.echo("❌ --from and --to resolve to the same snapshot.")
        return

    # Order: older = earlier captured_at, newer = later.
    if older["captured_at"] > newer["captured_at"]:
        older, newer = newer, older

    df_old = _load_oi_snapshot_df(older)
    df_new = _load_oi_snapshot_df(newer)
    if df_old is None or df_new is None:
        click.echo("❌ One of the snapshots has no strike rows.")
        return

    spot_old = float(older["spot_price"])
    spot_new = float(newer["spot_price"])
    old_label = f"as of {_local_ts(older['captured_at'])}"
    new_label = f"as of {_local_ts(newer['captured_at'])}"

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scope = f"±{range_pct:.0f}% of spot" if range_pct > 0 else "full chain"

    panels = [
        {"metric": "dollars", "cumulative": True},
        {"metric": "contracts", "cumulative": False},
        {"metric": "dollars", "cumulative": False},
    ]

    fig, axes = plt.subplots(3, 1, figsize=(14, 20), sharex=True)
    results = []
    any_empty = False
    for ax, panel in zip(axes, panels):
        result = _render_oi_diff_panel(
            ax, df_old=df_old, df_new=df_new,
            ticker=ticker, expiration=expiration,
            spot_old=spot_old, spot_new=spot_new,
            old_label=old_label, new_label=new_label,
            metric=panel["metric"], cumulative=panel["cumulative"],
            range_pct=range_pct, scope=scope, show_title=True,
        )
        results.append(result)
        if result.get("empty"):
            any_empty = True

    if any_empty:
        click.echo(
            f"No strikes within ±{range_pct}% of ${spot_new:.2f}. "
            "Try a larger --range."
        )
        plt.close(fig)
        return

    minutes_between = _iso_diff_minutes(
        older["captured_at"], newer["captured_at"]
    )
    span = ""
    if minutes_between is not None:
        if minutes_between >= 60 * 24:
            span = f" ({minutes_between / (60 * 24):,.1f} days apart)"
        elif minutes_between >= 60:
            span = f" ({minutes_between / 60:,.1f} hours apart)"
        else:
            span = f" ({minutes_between:,.0f} min apart)"

    fig.suptitle(
        f"{ticker} {expiration} — OI Dashboard Diff   ·   "
        f"{_local_ts(older['captured_at'])} → "
        f"{_local_ts(newer['captured_at'])}{span}",
        fontsize=14, fontweight="bold", y=0.995,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.985))

    if not out:
        os.makedirs("plots", exist_ok=True)
        out = f"plots/oi-dashboard-diff-{ticker}-{expiration}.png"
    plt.savefig(out, dpi=120)
    plt.close(fig)

    click.echo(f"📊 Saved: {out}")
    click.echo(
        f"    Old: {older['id']}  ({_local_ts(older['captured_at'])})  "
        f"spot ${spot_old:.2f}"
    )
    click.echo(
        f"    New: {newer['id']}  ({_local_ts(newer['captured_at'])})  "
        f"spot ${spot_new:.2f}   Δspot {spot_new - spot_old:+.2f}"
    )
    labels = ["Dollar cum", "Contracts bar", "Dollar bar"]
    for label, result in zip(labels, results):
        fmt = result["fmt_total"]
        click.echo(
            f"    {label:<14} | "
            f"Call {fmt(result['old_total_call'])} → "
            f"{fmt(result['new_total_call'])} "
            f"(Δ {fmt(result['delta_total_call'])})   |   "
            f"Put  {fmt(result['old_total_put'])} → "
            f"{fmt(result['new_total_put'])} "
            f"(Δ {fmt(result['delta_total_put'])})"
        )

    cum_res = results[0]
    old_cx, old_cy = cum_res["crossing_old"]
    new_cx, new_cy = cum_res["crossing_new"]
    if old_cx is not None and new_cx is not None:
        click.echo(
            f"    Crossing strike (dollar cum): "
            f"${old_cx:.2f} → ${new_cx:.2f}   "
            f"(Δ {new_cx - old_cx:+.2f})"
        )

    if not no_open:
        try:
            import subprocess
            subprocess.run(["open", out], check=False)
        except Exception:
            pass


@cli.command()
@click.option(
    "--ticker", default="SPY", help="Ticker symbol of the stock (default: SPY)"
)
@click.option(
    "--expiration",
    default=None,
    help="Expiration date YYYY-MM-DD (overrides --days if provided)",
)
@click.option(
    "--days",
    default=1,
    type=int,
    help="Business days from now to get options data (default: 1). "
    "Ignored if --expiration is given.",
)
@click.option(
    "--rate", default=0.02, help="Risk-free interest rate (default: 0.02 or 2%)"
)
@click.option(
    "--no-cache",
    is_flag=True,
    help="Disable caching and always fetch fresh data",
)
@click.option(
    "--refresh-cache",
    is_flag=True,
    help="Force refresh cached data with fresh API call",
)
@ensure_logged_in
def opt(ticker, expiration, days, rate, no_cache, refresh_cache):
    """
    Print options data (put and call) for the strike closest to current underlying price.

    Pass either --expiration YYYY-MM-DD directly, or --days N to compute
    the target date as N business days from now.

    The command fetches all options data for the chosen expiration, finds the
    strike price closest to the current underlying price, and displays both
    put and call option data for that strike (prices, Greeks, GEX).
    """
    try:
        if expiration:
            try:
                datetime.strptime(expiration, "%Y-%m-%d")
            except ValueError:
                click.echo(
                    f"❌ Invalid --expiration '{expiration}'. Use YYYY-MM-DD."
                )
                return
            expiration_date = expiration
            click.echo(f"Target expiration date: {expiration_date}")
        else:
            expiration_date = business_days_from_today(days)
            click.echo(
                f"Target expiration date: {expiration_date} ({days} business days from now)"
            )

        # Get options data using the existing Greeks function
        click.echo(f"Fetching options data for {ticker}...")
        df = calculate_options_greeks(
            ticker,
            expiration_date,
            rate,
            use_cache=not no_cache,
            refresh_cache=refresh_cache,
        )

        if df.empty:
            click.echo("No options data found or error occurred.")
            return

        # Get current stock price from the DataFrame attributes
        if hasattr(df, "attrs") and "stock_price" in df.attrs:
            current_price = df.attrs["stock_price"]
        else:
            # Fallback: get fresh stock price
            stock_price_list = rh.stocks.get_latest_price(
                ticker, priceType=None, includeExtendedHours=True
            )
            if not stock_price_list:
                click.echo(f"Could not retrieve current price for {ticker}")
                return
            current_price = float(stock_price_list[0])

        click.echo(f"Current {ticker} price: ${current_price:.2f}")

        # Find the strike closest to current price
        unique_strikes = df["strike_price"].unique()
        closest_strike = min(
            unique_strikes, key=lambda x: abs(x - current_price)
        )

        click.echo(f"Closest strike to current price: ${closest_strike:.2f}")
        click.echo(
            f"Distance from current price: ${abs(closest_strike - current_price):.2f} ({((closest_strike - current_price) / current_price * 100):+.2f}%)"
        )

        # Filter data for the closest strike (both call and put)
        closest_options = df[df["strike_price"] == closest_strike].copy()

        if closest_options.empty:
            click.echo(f"No options found for strike ${closest_strike:.2f}")
            return

        # Separate calls and puts
        calls = closest_options[closest_options["option_type"] == "CALL"]
        puts = closest_options[closest_options["option_type"] == "PUT"]

        # Display header
        click.echo(f"\n{'='*80}")
        click.echo(f"OPTIONS DATA FOR CLOSEST STRIKE: ${closest_strike:.2f}")
        click.echo(
            f"Expiration: {expiration_date} | Current Price: ${current_price:.2f}"
        )
        click.echo(f"{'='*80}")

        # Define columns to display
        display_columns = [
            "option_type",
            "market_price",
            "theoretical_price",
            "bid_price",
            "ask_price",
            "implied_volatility",
            "delta",
            "gamma",
            "vega",
            "theta",
            "gex_per_contract",
            "volume",
            "open_interest",
        ]

        # Format and display the data
        for option_type, data in [("CALL", calls), ("PUT", puts)]:
            if not data.empty:
                click.echo(f"\n{option_type} Option:")
                click.echo("-" * 50)

                # Get the single row
                option_data = data.iloc[0]

                # Display key information in a formatted way
                click.echo(
                    f"  Market Price:     ${option_data['market_price']:.2f}"
                )
                click.echo(
                    f"  Theoretical:      ${option_data['theoretical_price']:.2f}"
                )
                click.echo(
                    f"  Bid/Ask:          ${option_data['bid_price']:.2f} / ${option_data['ask_price']:.2f}"
                )
                click.echo(
                    f"  Implied Vol:      {option_data['implied_volatility']:.2%}"
                )
                click.echo(f"  Delta:            {option_data['delta']:.4f}")
                click.echo(f"  Gamma:            {option_data['gamma']:.6f}")
                click.echo(f"  Vega:             {option_data['vega']:.4f}")
                click.echo(f"  Theta:            {option_data['theta']:.4f}")
                click.echo(
                    f"  GEX/Contract:     ${option_data['gex_per_contract']:,.0f}"
                )
                click.echo(f"  Volume:           {option_data['volume'] or 0}")
                click.echo(
                    f"  Open Interest:    {option_data['open_interest']}"
                )
            else:
                click.echo(f"\n{option_type} Option: No data available")

        # Display summary comparison
        if not calls.empty and not puts.empty:
            call_data = calls.iloc[0]
            put_data = puts.iloc[0]

            click.echo(f"\n{'='*50}")
            click.echo("COMPARISON SUMMARY")
            click.echo(f"{'='*50}")
            click.echo(
                f"Call vs Put Prices:   ${call_data['market_price']:.2f} vs ${put_data['market_price']:.2f}"
            )
            click.echo(
                f"Call vs Put Deltas:   {call_data['delta']:.4f} vs {put_data['delta']:.4f}"
            )
            click.echo(
                f"Call vs Put Volume:   {call_data['volume'] or 0} vs {put_data['volume'] or 0}"
            )
            click.echo(
                f"Call vs Put OI:       {call_data['open_interest']} vs {put_data['open_interest']}"
            )

            # Calculate Put-Call parity check
            # C - P = S - K*e^(-r*T)
            time_to_expiry = calculate_time_to_expiry(expiration_date)
            theoretical_diff = current_price - closest_strike * np.exp(
                -rate * time_to_expiry
            )
            actual_diff = call_data["market_price"] - put_data["market_price"]
            parity_deviation = actual_diff - theoretical_diff

            click.echo(f"\nPut-Call Parity Check:")
            click.echo(f"  Theoretical C-P:    ${theoretical_diff:.2f}")
            click.echo(f"  Actual C-P:         ${actual_diff:.2f}")
            click.echo(f"  Deviation:          ${parity_deviation:.2f}")

    except Exception as e:
        click.echo(f"Error: {e}")


@cli.command()
@click.option("--stats", is_flag=True, help="Show cache statistics")
@click.option("--list", "list_cache", is_flag=True, help="List all cached data")
@click.option("--clear-all", is_flag=True, help="Clear all cached data")
@click.option("--clear-expired", is_flag=True, help="Clear expired cache files")
@click.option(
    "--clear-ticker", help="Clear cache for specific ticker (e.g., SPY)"
)
@ensure_logged_in
def cache(stats, list_cache, clear_all, clear_expired, clear_ticker):
    """
    Manage options data cache.

    This command provides utilities to view cache statistics, list cached data,
    and perform cache cleanup operations.
    """
    cache_instance = get_cache_instance()

    if stats:
        # Show cache statistics
        stats_data = cache_instance.get_stats()

        click.echo(f"\n📊 Cache Statistics")
        click.echo(f"{'='*50}")
        click.echo(f"Total files: {stats_data['total_files']}")
        click.echo(
            f"Total size: {stats_data['total_size_mb']} MB ({stats_data['total_size_bytes']:,} bytes)"
        )
        click.echo(f"Cache hits: {stats_data['cache_hits']}")
        click.echo(f"Cache misses: {stats_data['cache_misses']}")
        click.echo(f"Hit rate: {stats_data['hit_rate_percent']}%")
        click.echo(f"API requests saved: {stats_data['requests_saved']}")

        # Show breakdown by validity
        valid_files = [f for f in stats_data["files"] if f["valid"]]
        expired_files = [f for f in stats_data["files"] if not f["valid"]]

        click.echo(f"\nCache Health:")
        click.echo(f"  Valid files: {len(valid_files)}")
        click.echo(f"  Expired files: {len(expired_files)}")

        if expired_files:
            click.echo(
                f"  💡 Run 'python main.py cache --clear-expired' to clean up"
            )

    elif list_cache:
        # List all cached data
        cached_data = cache_instance.list_cached_data()

        if not cached_data:
            click.echo("📭 No cached data found")
            return

        click.echo(f"\n📋 Cached Data ({len(cached_data)} files)")
        click.echo(f"{'='*80}")

        for i, item in enumerate(cached_data, 1):
            status = "✅ Valid" if item["valid"] else "⏰ Expired"
            size_kb = item["file_size"] / 1024

            click.echo(f"{i:2d}. {item['ticker']} - {item['expiration_date']}")
            click.echo(
                f"    Cached: {item['cached_at'][:19]} | Size: {size_kb:.1f} KB | {status}"
            )

            if i >= 20:  # Limit to first 20 for readability
                remaining = len(cached_data) - 20
                if remaining > 0:
                    click.echo(f"    ... and {remaining} more files")
                break

    elif clear_all:
        # Clear all cache
        if click.confirm("🗑️  Are you sure you want to clear ALL cached data?"):
            count = cache_instance.clear_all()
            click.echo(f"✅ Cleared {count} cache files")
        else:
            click.echo("❌ Operation cancelled")

    elif clear_expired:
        # Clear expired cache files
        count = cache_instance.clear_expired()
        click.echo(f"✅ Cleared {count} expired cache files")

    elif clear_ticker:
        # Clear cache for specific ticker
        if click.confirm(
            f"🗑️  Clear all cached data for {clear_ticker.upper()}?"
        ):
            count = cache_instance.clear_ticker(clear_ticker)
            click.echo(
                f"✅ Cleared {count} cache files for {clear_ticker.upper()}"
            )
        else:
            click.echo("❌ Operation cancelled")

    else:
        # Show default cache info
        stats_data = cache_instance.get_stats()
        click.echo(f"📦 Options Data Cache")
        click.echo(
            f"Files: {stats_data['total_files']} | Size: {stats_data['total_size_mb']} MB | Hit Rate: {stats_data['hit_rate_percent']}%"
        )
        click.echo(f"\nAvailable commands:")
        click.echo(f"  --stats          Show detailed statistics")
        click.echo(f"  --list           List all cached data")
        click.echo(f"  --clear-expired  Remove expired cache files")
        click.echo(f"  --clear-ticker   Remove cache for specific ticker")
        click.echo(f"  --clear-all      Remove all cached data")


def main():
    # Example usage:
    stock_price = 590
    implied_volatility = 0.1956  # 30% IV
    days = [1, 3, 5, 10, 21, 30, 60]

    # 1 sig = 68% confidence, 2 sig = 95% confidence
    # 3 sig = 99.7% confidence
    confidence_levels = [0.6827, 0.9545, 0.997]

    df = implied_move(stock_price, implied_volatility, days, confidence_levels)
    print(df)

    # Example usage:
    S = 100  # Current stock price
    K = 105  # Strike price
    T = 30 / 365  # Time to expiry in years
    r = 0.01  # Risk-free rate
    market_price = 2.50  # Observed option price in the market

    iv = implied_volatility_call(S, K, T, r, market_price)
    print(f"Implied Volatility: {iv:.2%}")

    # optionData = robin_stocks.find_options_for_list_of_stocks_by_expiration_date(['fb','aapl','tsla','nflx'],
    #         expirationDate='2018-11-16',optionType='call')

    priceType = "bid_price"  # or 'ask_price'
    stock_price = rh.stocks.get_latest_price(
        "SPY", priceType=None, includeExtendedHours=True
    )
    print(f"Current SPY price: {stock_price}")


if __name__ == "__main__":
    cli()
