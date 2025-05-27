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
