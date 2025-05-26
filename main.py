import numpy as np
from scipy.stats import norm
import pandas as pd
import robin_stocks.robinhood as rh
from dotenv import load_dotenv
import os

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

def main():
    # Example usage:
    stock_price = 100
    implied_volatility = 0.30  # 30% IV
    days = [1, 5, 10, 21, 30, 60]
    confidence_levels = [0.68, 0.95]

    df = implied_move(stock_price, implied_volatility, days, confidence_levels)
    print(df)

    login_robinhood(os.getenv("RH_USERNAME"), os.getenv("RH_PASSWORD"))  # Replace with your credentials


    # optionData = robin_stocks.find_options_for_list_of_stocks_by_expiration_date(['fb','aapl','tsla','nflx'],
    #         expirationDate='2018-11-16',optionType='call')

    priceType = 'bid_price'  # or 'ask_price'
    stock_price = rh.stocks.get_latest_price("SPY", priceType=None, includeExtendedHours=True)
    print(f"Current SPY price: {stock_price}")
if __name__ == "__main__":
    main()
