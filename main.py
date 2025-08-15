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
from options import OptionPricer

# looks for a .env file in the current directory
# RH_USERNAME and RH_PASSWORD should be set in the .env file
load_dotenv()

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
            results.append({
                'Days': days,
                'Confidence Level': f"{int(conf * 100)}%",
                'Implied Move ($)': round(move, 2),
                'Price Range': f"{round(stock_price - move, 2)} - {round(stock_price + move, 2)}"
            })
    
    return pd.DataFrame(results)



# Black-Scholes Call Price
def black_scholes_call_price(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# Function to solve: difference between market price and BS price
def implied_volatility_call(S, K, T, r, market_price):
    objective = lambda sigma: black_scholes_call_price(S, K, T, r, sigma) - market_price
    return brentq(objective, 1e-5, 3)  # search in a reasonable range for sigma (0.001% - 300%)

def business_days_from_today(n):
    current_date = datetime.today()
    direction = 1 if n>= 0 else -1
    days_remaining = abs(n)

    while days_remaining > 0:
        current_date += timedelta(days=direction)
        if current_date.weekday() < 5: # Weekdays are 0-4 (Mon-Fri)
            days_remaining -= 1
    return current_date.strftime('%Y-%m-%d')

def calculate_time_to_expiry(expiration_date):
    """
    Calculate time to expiry in years from current date to expiration date.
    
    Parameters:
    - expiration_date: str, expiration date in 'YYYY-MM-DD' format
    
    Returns:
    - float, time to expiry in years
    """
    current_date = datetime.now()
    exp_date = datetime.strptime(expiration_date, '%Y-%m-%d')
    days_diff = (exp_date - current_date).days
    return max(days_diff / 365.0, 1/365)  # Minimum 1 day to avoid division by zero

def calculate_options_greeks(ticker, expiration_date, risk_free_rate=0.02):
    """
    Calculate delta and gamma for all options of a given ticker on a specific expiration date.
    
    Parameters:
    - ticker: str, stock ticker symbol
    - expiration_date: str, expiration date in 'YYYY-MM-DD' format
    - risk_free_rate: float, risk-free interest rate (default 2%)
    
    Returns:
    - pandas.DataFrame with options data and calculated Greeks
    """
    try:
        # Get current stock price
        stock_price_list = rh.stocks.get_latest_price(ticker, priceType=None, includeExtendedHours=True)
        if not stock_price_list:
            raise ValueError(f"Could not retrieve price for {ticker}")
        
        stock_price = float(stock_price_list[0])
        print(f"Current {ticker} price: ${stock_price:.2f}")
        
        # Calculate time to expiry
        time_to_expiry = calculate_time_to_expiry(expiration_date)
        print(f"Time to expiry: {time_to_expiry:.4f} years ({time_to_expiry*365:.1f} days)")
        
        # Get all available options for the expiration date
        options_data = rh.options.find_options_by_expiration(ticker, expiration_date, info=None)
        
        if not options_data:
            raise ValueError(f"No options found for {ticker} on {expiration_date}")
        
        print(f"Found {len(options_data)} options contracts")
        
        results = []
        
        for option in options_data:
            try:
                # Extract option details
                strike_price = float(option.get('strike_price', 0))
                option_type = option.get('type', '').lower()  # 'call' or 'put'
                
                # Get market price (use mark price if available, otherwise midpoint of bid/ask)
                mark_price = option.get('mark_price')
                bid_price = option.get('bid_price') 
                ask_price = option.get('ask_price')
                
                if mark_price and float(mark_price) > 0:
                    market_price = float(mark_price)
                elif bid_price and ask_price and float(bid_price) > 0 and float(ask_price) > 0:
                    market_price = (float(bid_price) + float(ask_price)) / 2
                else:
                    continue  # Skip options without valid pricing
                
                # Get implied volatility
                implied_vol = option.get('implied_volatility')
                if not implied_vol or float(implied_vol) <= 0:
                    continue  # Skip options without IV
                
                implied_vol = float(implied_vol)
                
                # Create OptionPricer instance
                pricer = OptionPricer(
                    S=stock_price,
                    K=strike_price, 
                    T=time_to_expiry,
                    r=risk_free_rate,
                    market_price=market_price
                )
                
                # Calculate Greeks
                delta = pricer.delta(implied_vol, option_type)
                gamma = pricer.gamma(implied_vol)
                vega = pricer.vega(implied_vol)
                theta = pricer.theta(implied_vol, option_type)
                
                # Calculate GEX (Gamma Exposure)
                open_interest_val = option.get('open_interest')
                open_interest_val = int(open_interest_val) if open_interest_val else 0
                
                gex_per_contract = pricer.gex_per_contract(implied_vol, open_interest_val, option_type)
                gex_notional = pricer.gex_notional(implied_vol, open_interest_val, option_type)
                
                # Debug: Print first few options to help diagnose $0 GEX issues
                if len(results) < 5:  # Show first 5 options for better debugging
                    print(f"DEBUG Option {len(results)+1}: {option_type.upper()} ${strike_price}")
                    print(f"  Raw option type: '{option.get('type', 'MISSING')}'")
                    print(f"  Processed option_type: '{option_type}'")
                    print(f"  Is 'call'? {option_type.lower() == 'call'}")
                    print(f"  Is 'put'? {option_type.lower() == 'put'}")
                    print(f"  Open Interest: {open_interest_val}")
                    print(f"  Gamma: {gamma:.6f}")
                    print(f"  GEX per contract: ${gex_per_contract:,.0f}")
                    print(f"  GEX notional: ${gex_notional:,.0f}")
                    print(f"  Stock price: ${stock_price}")
                    print("---")
                
                # Calculate theoretical price using Black-Scholes
                if option_type == 'call':
                    theoretical_price = pricer.black_scholes_call(implied_vol)
                else:
                    theoretical_price = pricer.black_scholes_put(implied_vol)
                
                # Store results
                results.append({
                    'ticker': ticker,
                    'expiration_date': expiration_date,
                    'option_type': option_type.upper(),
                    'strike_price': strike_price,
                    'market_price': market_price,
                    'theoretical_price': theoretical_price,
                    'bid_price': float(bid_price) if bid_price else None,
                    'ask_price': float(ask_price) if ask_price else None,
                    'implied_volatility': implied_vol,
                    'delta': delta,
                    'gamma': gamma,
                    'vega': vega,
                    'theta': theta,
                    'gex_per_contract': gex_per_contract,
                    'gex_notional': gex_notional,
                    'volume': option.get('volume'),
                    'open_interest': open_interest_val,
                    'time_to_expiry_days': time_to_expiry * 365
                })
                
            except (ValueError, TypeError, KeyError) as e:
                print(f"Error processing option {option.get('strike_price', 'unknown')}: {e}")
                continue
        
        if not results:
            raise ValueError(f"No valid options data found for {ticker} on {expiration_date}")
        
        # Create DataFrame and sort by strike price and option type
        df = pd.DataFrame(results)
        df = df.sort_values(['option_type', 'strike_price'])
        
        # Calculate portfolio-level GEX analysis
        if not df.empty:
            # Portfolio GEX analysis
            all_gex = df['gex_per_contract'].tolist()
            portfolio_gex = OptionPricer.calculate_portfolio_gex(all_gex)
            
            # GEX by strike analysis
            gex_by_strike = df.groupby('strike_price')['gex_per_contract'].sum().to_dict()
            gamma_levels = OptionPricer.get_gamma_levels(stock_price, gex_by_strike)
            
            # Add portfolio analysis to DataFrame as metadata
            df.attrs['portfolio_gex'] = portfolio_gex
            df.attrs['gamma_levels'] = gamma_levels
            df.attrs['stock_price'] = stock_price
        
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
@click.option('--ticker', default='SPY', help='Ticker symbol of the stock (default: SPY)')
@click.option('--days', default='1,3,5', help='Comma-separated list of days for implied move calculation (default: 1,3,5,10,21,30,60)')
@click.option('--confidence', default='0.6827,0.9545,0.997', help='Comma-separated list of confidence levels (default: 0.6827,0.9545,0.997)')
def emove(ticker, days, confidence):

    stock_price = rh.stocks.get_latest_price(ticker, priceType=None, includeExtendedHours=True)
    if stock_price is None:
        click.echo(f"Could not retrieve price for {ticker}. Please check the ticker symbol.")
        return
    
    stock_price = float(stock_price[0])
    click.echo(f"Current {ticker} price: {stock_price}")


    # TODO: optionType 'call' or 'put'
    # YYYY-MM-DD
    expirationDate = business_days_from_today(1)
    strikePrice = int(stock_price)
    options_data_list = rh.options.find_options_by_expiration_and_strike(ticker, expirationDate, strikePrice, optionType='call', info=None)
    options_data = options_data_list[0]
    
    #print(json.dumps(options_data, indent=4))
    iv = options_data.get('implied_volatility', None)
    if iv is None:
        click.echo(f"No implied volatility data available for {ticker} on {expirationDate} with strike {strikePrice}.")
        click.echo("Please check the options data or try a different ticker.")
        return
    
    iv = float(iv) if iv else 0.1956  # Default to 19.56% if not available
    click.echo(f"Implied Volatility for {ticker} on {expirationDate} with strike {strikePrice}: {iv:.4%}")


    #implied_volatility = 0.1956  # 30% IV
    #days = [1, 3, 5, 10, 21, 30, 60]

    # 1 sig = 68% confidence, 
    # 2 sig = 95% confidence
    # 3 sig = 99.7% confidence
    #confidence_levels = [0.6827, 0.9545, 0.997]
    confidence_levels = [float(c) for c in confidence.split(',')]
    days = [int(d) for d in days.split(',')]

    df = implied_move(stock_price, iv, days, confidence_levels)
    print(df)

@cli.command()
@click.option('--ticker', default='SPY', help='Ticker symbol of the stock (default: SPY)')
@click.option('--expiration', help='Expiration date in YYYY-MM-DD format (default: next business day)')
@click.option('--rate', default=0.02, help='Risk-free interest rate (default: 0.02 or 2%)')
@click.option('--output', help='Output file path to save results as CSV (optional)')
@click.option('--min-volume', default=0, help='Minimum volume filter (default: 0)')
@click.option('--show-all', is_flag=True, help='Show all Greeks (delta, gamma, vega, theta) instead of just delta and gamma')
@click.option('--show-gex', is_flag=True, help='Show Gamma Exposure (GEX) analysis and positioning')
def greeks(ticker, expiration, rate, output, min_volume, show_all, show_gex):
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
        
        click.echo(f"Calculating Greeks for {ticker} options expiring on {expiration}")
        click.echo(f"Using risk-free rate: {rate:.2%}")
        
        # Calculate options Greeks
        df = calculate_options_greeks(ticker, expiration, rate)
        
        if df.empty:
            click.echo("No options data found or error occurred.")
            return
        
        # Apply volume filter
        if min_volume > 0:
            df = df[df['volume'].fillna(0) >= min_volume]
            click.echo(f"Filtered to options with volume >= {min_volume}")
        
        # Select columns to display
        if show_all:
            display_columns = [
                'option_type', 'strike_price', 'market_price', 'theoretical_price',
                'bid_price', 'ask_price', 'implied_volatility', 'delta', 'gamma', 
                'vega', 'theta', 'gex_per_contract', 'gex_notional', 'volume', 'open_interest'
            ]
        else:
            display_columns = [
                'option_type', 'strike_price', 'market_price', 'implied_volatility',
                'delta', 'gamma', 'gex_per_contract', 'gex_notional', 'volume', 'open_interest'
            ]
        
        # Remove GEX columns if not requested for display
        if not show_gex:
            display_columns = [col for col in display_columns if not col.startswith('gex_')]
        
        # Format and display results
        display_df = df[display_columns].copy()
        
        # Round numerical columns for better display
        numeric_columns = ['market_price', 'theoretical_price', 'bid_price', 'ask_price', 
                          'implied_volatility', 'delta', 'gamma', 'vega', 'theta', 
                          'gex_per_contract', 'gex_notional']
        for col in numeric_columns:
            if col in display_df.columns:
                if col == 'implied_volatility':
                    display_df[col] = display_df[col].round(4)
                elif col in ['delta', 'gamma', 'vega', 'theta']:
                    display_df[col] = display_df[col].round(6)
                elif col in ['gex_per_contract', 'gex_notional']:
                    display_df[col] = display_df[col].round(0).astype(int)
                else:
                    display_df[col] = display_df[col].round(2)
        
        click.echo(f"\nFound {len(display_df)} options contracts:")
        click.echo(f"Calls: {len(display_df[display_df['option_type'] == 'CALL'])}")
        click.echo(f"Puts: {len(display_df[display_df['option_type'] == 'PUT'])}")
        
        # Display summary statistics
        click.echo(f"\nDelta range: {df['delta'].min():.4f} to {df['delta'].max():.4f}")
        click.echo(f"Gamma range: {df['gamma'].min():.6f} to {df['gamma'].max():.6f}")
        
        # GEX debugging summary
        if 'gex_per_contract' in df.columns:
            total_options = len(df)
            options_with_gex = len(df[df['gex_per_contract'] != 0])
            options_with_oi = len(df[df['open_interest'] > 0])
            
            # Analyze call vs put breakdown
            calls = df[df['option_type'] == 'CALL']
            puts = df[df['option_type'] == 'PUT']
            call_gex_positive = len(calls[calls['gex_per_contract'] > 0])
            put_gex_positive = len(puts[puts['gex_per_contract'] > 0])
            call_gex_negative = len(calls[calls['gex_per_contract'] < 0])
            put_gex_negative = len(puts[puts['gex_per_contract'] < 0])
            
            click.echo(f"\nGEX Debug Summary:")
            click.echo(f"  Total options: {total_options}")
            click.echo(f"  Calls: {len(calls)} | Puts: {len(puts)}")
            click.echo(f"  Options with open interest > 0: {options_with_oi}")
            click.echo(f"  Options with non-zero GEX: {options_with_gex}")
            click.echo(f"  GEX range: ${df['gex_per_contract'].min():,.0f} to ${df['gex_per_contract'].max():,.0f}")
            
            click.echo(f"\nGEX Sign Analysis:")
            click.echo(f"  Calls with negative GEX: {call_gex_negative} ✅ (expected)")
            click.echo(f"  Calls with positive GEX: {call_gex_positive} ❌ (unexpected)")
            click.echo(f"  Puts with positive GEX: {put_gex_positive} ✅ (expected)")
            click.echo(f"  Puts with negative GEX: {put_gex_negative} ❌ (unexpected)")
            
            if options_with_gex == 0:
                click.echo(f"  ⚠️  All options have $0 GEX - likely due to zero open interest")
            elif put_gex_negative > 0:
                click.echo(f"  🚨 PROBLEM: {put_gex_negative} puts have negative GEX (should be positive)")
            elif len(puts) == 0:
                click.echo(f"  🚨 PROBLEM: No PUT options found - check API data")
            
        # Display GEX Analysis
        if hasattr(df, 'attrs') and 'portfolio_gex' in df.attrs and show_gex:
            portfolio_gex = df.attrs['portfolio_gex']
            gamma_levels = df.attrs['gamma_levels']
            stock_price = df.attrs['stock_price']
            
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
            if gamma_levels and 'top_3_gamma_strikes' in gamma_levels:
                click.echo(f"\nKey Gamma Levels (Top 3 by absolute GEX):")
                for i, (strike, gex) in enumerate(gamma_levels['top_3_gamma_strikes'], 1):
                    distance = ((strike - stock_price) / stock_price) * 100
                    click.echo(f"  {i}. ${strike:.0f} - GEX: ${gex:,.0f} ({distance:+.1f}% from spot)")
                
                # Support/Resistance levels
                if gamma_levels.get('nearest_support'):
                    support_strike, support_gex = gamma_levels['nearest_support']
                    click.echo(f"\nNearest Support: ${support_strike:.0f} (GEX: ${support_gex:,.0f})")
                
                if gamma_levels.get('nearest_resistance'):
                    resistance_strike, resistance_gex = gamma_levels['nearest_resistance']
                    click.echo(f"Nearest Resistance: ${resistance_strike:.0f} (GEX: ${resistance_gex:,.0f})")
            
            # GEX interpretation
            total_gex_millions = portfolio_gex['total_gex'] / 1000000
            click.echo(f"\nGEX Interpretation:")
            if abs(total_gex_millions) < 1:
                click.echo(f"  Low GEX environment (${abs(total_gex_millions):.1f}M) - Expect normal volatility")
            elif abs(total_gex_millions) < 10:
                click.echo(f"  Moderate GEX environment (${abs(total_gex_millions):.1f}M) - Some dealer flow impact")
            else:
                click.echo(f"  High GEX environment (${abs(total_gex_millions):.1f}M) - Significant dealer flow expected")
        elif show_gex and hasattr(df, 'attrs') and 'portfolio_gex' in df.attrs:
            # Show basic GEX summary even if detailed analysis failed
            portfolio_gex = df.attrs['portfolio_gex']
            click.echo(f"\nGEX Summary: {portfolio_gex['gamma_position']} - Net GEX: ${portfolio_gex['net_gex']:,.0f}")
        
        # Display the data
        click.echo("\nOptions Greeks Data:")
        click.echo(display_df.to_string(index=False))
        
        # Save to CSV if output file specified
        if output:
            df.to_csv(output, index=False)
            click.echo(f"\nFull results saved to: {output}")
            
    except Exception as e:
        click.echo(f"Error: {e}")




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
    S = 100         # Current stock price
    K = 105         # Strike price
    T = 30/365      # Time to expiry in years
    r = 0.01        # Risk-free rate
    market_price = 2.50  # Observed option price in the market

    iv = implied_volatility_call(S, K, T, r, market_price)
    print(f"Implied Volatility: {iv:.2%}")

    


    # optionData = robin_stocks.find_options_for_list_of_stocks_by_expiration_date(['fb','aapl','tsla','nflx'],
    #         expirationDate='2018-11-16',optionType='call')

    priceType = 'bid_price'  # or 'ask_price'
    stock_price = rh.stocks.get_latest_price("SPY", priceType=None, includeExtendedHours=True)
    print(f"Current SPY price: {stock_price}")

if __name__ == "__main__":
    login_robinhood(os.getenv("RH_USERNAME"), os.getenv("RH_PASSWORD"))  # Replace with your credentials
    cli()
