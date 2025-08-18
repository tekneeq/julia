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
from .options import OptionPricer
from .options_cache import get_cache_instance
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
                click.echo("❌ Error: RH_USERNAME and RH_PASSWORD must be set in environment variables or .env file")
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
@ensure_logged_in
def emove(ticker, days, confidence):

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

    # TODO: optionType 'call' or 'put'
    # YYYY-MM-DD
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
    days = [int(d) for d in days.split(",")]

    df = implied_move(stock_price, iv, days, confidence_levels)
    print(df)


@cli.command()
@click.option(
    "--ticker", default="SPY", help="Ticker symbol of the stock (default: SPY)"
)
@ensure_logged_in
def range(ticker):

    def todays_high_low_intraday(symbol: str, bounds: str = "regular") -> tuple[float | None, float | None]:
        candles = rh.stocks.get_stock_historicals(
            symbol,
            interval="5minute",
            span="day",
            bounds=bounds
        )
        if not candles:
            return None, None
        highs = [float(candle['high_price']) for candle in candles if candle.get('high_price')]
        lows = [float(candle['low_price']) for candle in candles if candle.get('low_price')]
        return max(highs) if highs else None, min(lows) if lows else None
    
    def todays_high_low(symbol: str) -> tuple[float | None, float | None]:
        hi, lo = todays_high_low_intraday(symbol)
        if hi is None or lo is None:
            hi, lo = todays_high_low_intraday(symbol, bounds="trading")

        return hi, lo

    hi, lo = todays_high_low(ticker)
    print(f"{ticker} Today's High: {hi}, Low: {lo}, Range: {hi - lo if hi and lo else 'N/A'}")



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

    # TODO: optionType 'call' or 'put'
    # YYYY-MM-DD
    expirationDate = business_days_from_today(1)
    strikePrice = int(stock_price)
    options_data_list = rh.options.find_options_by_expiration_and_strike(
        ticker, expirationDate, strikePrice, optionType="call", info=None
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
@ensure_logged_in
def greeks(
    ticker,
    expiration,
    rate,
    output,
    min_volume,
    show_all,
    show_gex,
    no_cache,
    refresh_cache,
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


@cli.command()
@click.option(
    "--ticker", default="SPY", help="Ticker symbol of the stock (default: SPY)"
)
@click.option(
    "--days",
    default=1,
    type=int,
    help="Business days from now to get options data (default: 1)",
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
def opt(ticker, days, rate, no_cache, refresh_cache):
    """
    Print options data (put and call) for the strike closest to current underlying price.

    This command calculates the target expiration date based on business days from now,
    fetches all options data for that date, finds the strike price closest to the current
    underlying price, and displays both put and call option data for that strike.

    The command reuses the Greeks calculation functionality to gather comprehensive
    options data including prices, Greeks, and GEX values.
    """
    try:
        # Calculate target expiration date
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
