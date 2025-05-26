import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

class OptionPricer:
    def __init__(self, S, K, T, r, market_price, steps=100, q=0.0):
        self.S = S  # Underlying price
        self.K = K  # Strike price
        self.T = T  # Time to maturity (in years)
        self.r = r  # Risk-free interest rate
        self.q = q  # Dividend yield
        self.market_price = market_price
        self.steps = steps  # Number of steps for binomial tree

    # --- Black-Scholes Formulas ---
    def black_scholes_call(self, sigma):
        d1 = (np.log(self.S / self.K) + (self.r - self.q + sigma**2 / 2) * self.T) / (sigma * np.sqrt(self.T))
        d2 = d1 - sigma * np.sqrt(self.T)
        return self.S * np.exp(-self.q * self.T) * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)

    def black_scholes_put(self, sigma):
        d1 = (np.log(self.S / self.K) + (self.r - self.q + sigma**2 / 2) * self.T) / (sigma * np.sqrt(self.T))
        d2 = d1 - sigma * np.sqrt(self.T)
        return self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * np.exp(-self.q * self.T) * norm.cdf(-d1)

    def implied_volatility_bs(self, option_type='call'):
        if option_type == 'call':
            objective = lambda sigma: self.black_scholes_call(sigma) - self.market_price
        else:
            objective = lambda sigma: self.black_scholes_put(sigma) - self.market_price

        return brentq(objective, 1e-5, 3)

    # --- Unified Binomial Tree for American Options ---
    def binomial_tree_american_option(self, sigma, option_type='put'):
        dt = self.T / self.steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp((self.r - self.q) * dt) - d) / (u - d)

        prices = np.array([self.S * (u ** j) * (d ** (self.steps - j)) for j in range(self.steps + 1)])
        if option_type == 'call':
            option = np.maximum(prices - self.K, 0)
        else:
            option = np.maximum(self.K - prices, 0)

        for i in reversed(range(self.steps)):
            prices = prices[:-1] / u
            if option_type == 'call':
                intrinsic = prices - self.K
            else:
                intrinsic = self.K - prices
            option = np.maximum(
                intrinsic,
                np.exp(-self.r * dt) * (p * option[1:] + (1 - p) * option[:-1])
            )

        return option[0]

    def implied_volatility_american(self, option_type='put'):
        def objective(sigma):
            return self.binomial_tree_american_option(sigma, option_type) - self.market_price

        return brentq(objective, 1e-5, 3)


# --- Example Usage ---
if __name__ == "__main__":
    pricer_eur = OptionPricer(S=100, K=95, T=60/365, r=0.01, market_price=4.20)
    iv_eur_put = pricer_eur.implied_volatility_bs(option_type='put')
    print(f"European Put IV: {iv_eur_put:.2%}")

    pricer_amer_put = OptionPricer(S=100, K=105, T=90/365, r=0.015, market_price=7.30)
    iv_amer_put = pricer_amer_put.implied_volatility_american(option_type='put')
    print(f"American Put IV: {iv_amer_put:.2%}")

    pricer_amer_call = OptionPricer(S=100, K=95, T=90/365, r=0.01, market_price=8.50, q=0.02)
    iv_amer_call = pricer_amer_call.implied_volatility_american(option_type='call')
    print(f"American Call IV (with dividends): {iv_amer_call:.2%}")