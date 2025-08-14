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

    # --- Greeks Calculations ---
    def delta(self, sigma, option_type='call'):
        """
        Calculate delta (price sensitivity to underlying price changes).
        
        Parameters:
        - sigma: float, volatility
        - option_type: str, 'call' or 'put'
        
        Returns:
        - float, delta value
        """
        if self.T <= 0:
            # At expiration
            if option_type == 'call':
                return 1.0 if self.S > self.K else 0.0
            else:
                return -1.0 if self.S < self.K else 0.0
        
        d1 = (np.log(self.S / self.K) + (self.r - self.q + sigma**2 / 2) * self.T) / (sigma * np.sqrt(self.T))
        
        if option_type == 'call':
            return np.exp(-self.q * self.T) * norm.cdf(d1)
        else:  # put
            return -np.exp(-self.q * self.T) * norm.cdf(-d1)

    def gamma(self, sigma):
        """
        Calculate gamma (rate of change of delta with respect to underlying price).
        Gamma is the same for both calls and puts.
        
        Parameters:
        - sigma: float, volatility
        
        Returns:
        - float, gamma value
        """
        if self.T <= 0:
            return 0.0  # At expiration, gamma is 0
        
        d1 = (np.log(self.S / self.K) + (self.r - self.q + sigma**2 / 2) * self.T) / (sigma * np.sqrt(self.T))
        return (np.exp(-self.q * self.T) * norm.pdf(d1)) / (self.S * sigma * np.sqrt(self.T))

    def vega(self, sigma):
        """
        Calculate vega (price sensitivity to volatility changes).
        Vega is the same for both calls and puts.
        
        Parameters:
        - sigma: float, volatility
        
        Returns:
        - float, vega value
        """
        if self.T <= 0:
            return 0.0  # At expiration, vega is 0
        
        d1 = (np.log(self.S / self.K) + (self.r - self.q + sigma**2 / 2) * self.T) / (sigma * np.sqrt(self.T))
        return self.S * np.exp(-self.q * self.T) * norm.pdf(d1) * np.sqrt(self.T) / 100  # Divided by 100 for 1% change

    def theta(self, sigma, option_type='call'):
        """
        Calculate theta (time decay).
        
        Parameters:
        - sigma: float, volatility
        - option_type: str, 'call' or 'put'
        
        Returns:
        - float, theta value (usually negative)
        """
        if self.T <= 0:
            return 0.0  # At expiration, theta is 0
        
        d1 = (np.log(self.S / self.K) + (self.r - self.q + sigma**2 / 2) * self.T) / (sigma * np.sqrt(self.T))
        d2 = d1 - sigma * np.sqrt(self.T)
        
        common_term = -(self.S * np.exp(-self.q * self.T) * norm.pdf(d1) * sigma) / (2 * np.sqrt(self.T))
        
        if option_type == 'call':
            theta = common_term - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2) + self.q * self.S * np.exp(-self.q * self.T) * norm.cdf(d1)
        else:  # put
            theta = common_term + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.q * self.S * np.exp(-self.q * self.T) * norm.cdf(-d1)
        
        return theta / 365  # Convert to daily theta

    def gex_per_contract(self, sigma, open_interest, option_type='call'):
        """
        Calculate Gamma Exposure (GEX) for a single options contract.
        
        GEX represents the dollar amount of gamma exposure for each 1% move in the underlying.
        For market makers who are short options, positive GEX means they need to buy stock as price goes up
        and sell as price goes down (stabilizing effect). Negative GEX means the opposite (destabilizing).
        
        Parameters:
        - sigma: float, volatility
        - open_interest: int, number of contracts outstanding
        - option_type: str, 'call' or 'put'
        
        Returns:
        - float, GEX value in dollars
        """
        if not open_interest or open_interest <= 0:
            return 0.0
            
        gamma_value = self.gamma(sigma)
        
        # GEX = Gamma × Open Interest × 100 shares/contract × Spot Price × 0.01 (for 1% move)
        # Market makers are typically short options, so we flip the sign
        gex = -gamma_value * open_interest * 100 * self.S * 0.01
        
        return gex
    
    def gex_notional(self, sigma, open_interest, option_type='call'):
        """
        Calculate notional GEX (total gamma exposure in dollar terms).
        
        Parameters:
        - sigma: float, volatility
        - open_interest: int, number of contracts outstanding
        - option_type: str, 'call' or 'put'
        
        Returns:
        - float, notional GEX value in dollars
        """
        if not open_interest or open_interest <= 0:
            return 0.0
            
        gamma_value = self.gamma(sigma)
        
        # Notional GEX = Gamma × Open Interest × 100 shares/contract × (Spot Price)^2
        # This represents the total dollar gamma exposure
        notional_gex = -gamma_value * open_interest * 100 * (self.S ** 2)
        
        return notional_gex

    @staticmethod
    def calculate_portfolio_gex(gex_values):
        """
        Calculate total portfolio GEX and determine gamma positioning.
        
        Parameters:
        - gex_values: list of floats, individual contract GEX values
        
        Returns:
        - dict with portfolio GEX metrics and positioning analysis
        """
        total_gex = sum(gex_values)
        call_gex = sum([gex for gex in gex_values if gex > 0])
        put_gex = sum([gex for gex in gex_values if gex < 0])
        
        # Determine gamma positioning
        if abs(total_gex) < 1000000:  # Less than $1M threshold
            gamma_position = "GAMMA NEUTRAL"
            position_description = "Market is approximately gamma neutral. Limited systematic flow expected from gamma hedging."
        elif total_gex > 0:
            gamma_position = "LONG GAMMA"
            position_description = "Market makers are net short gamma. They will buy on dips and sell on rallies (stabilizing)."
        else:
            gamma_position = "SHORT GAMMA"
            position_description = "Market makers are net long gamma. They will sell on dips and buy on rallies (destabilizing)."
        
        return {
            'total_gex': total_gex,
            'call_gex': call_gex,
            'put_gex': put_gex,
            'net_gex': total_gex,
            'gamma_position': gamma_position,
            'position_description': position_description,
            'gex_per_1pct_move': total_gex,
            'abs_gex': abs(total_gex)
        }

    @staticmethod
    def get_gamma_levels(spot_price, gex_by_strike):
        """
        Identify key gamma levels (strikes with highest absolute GEX).
        
        Parameters:
        - spot_price: float, current underlying price
        - gex_by_strike: dict, mapping of strike prices to GEX values
        
        Returns:
        - dict with key gamma levels and analysis
        """
        if not gex_by_strike:
            return {}
            
        # Sort strikes by absolute GEX (highest first)
        sorted_strikes = sorted(gex_by_strike.items(), 
                              key=lambda x: abs(x[1]), reverse=True)
        
        # Find largest positive and negative GEX levels
        max_positive_gex = max(gex_by_strike.items(), key=lambda x: x[1])
        max_negative_gex = min(gex_by_strike.items(), key=lambda x: x[1])
        
        # Find closest strikes to current price
        strikes_above = {k: v for k, v in gex_by_strike.items() if k > spot_price}
        strikes_below = {k: v for k, v in gex_by_strike.items() if k < spot_price}
        
        resistance_level = None
        support_level = None
        
        if strikes_above:
            resistance_level = min(strikes_above.items(), key=lambda x: x[0])
        if strikes_below:
            support_level = max(strikes_below.items(), key=lambda x: x[0])
        
        return {
            'top_3_gamma_strikes': sorted_strikes[:3],
            'max_positive_gex_strike': max_positive_gex,
            'max_negative_gex_strike': max_negative_gex,
            'nearest_resistance': resistance_level,
            'nearest_support': support_level,
            'current_spot': spot_price
        }

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
    # Test European options pricing and Greeks
    pricer_eur = OptionPricer(S=100, K=95, T=60/365, r=0.01, market_price=4.20)
    iv_eur_put = pricer_eur.implied_volatility_bs(option_type='put')
    print(f"European Put IV: {iv_eur_put:.2%}")
    
    # Test Greeks calculations
    print(f"European Put Delta: {pricer_eur.delta(iv_eur_put, 'put'):.4f}")
    print(f"European Put Gamma: {pricer_eur.gamma(iv_eur_put):.6f}")
    print(f"European Put Vega: {pricer_eur.vega(iv_eur_put):.4f}")
    print(f"European Put Theta: {pricer_eur.theta(iv_eur_put, 'put'):.4f}")

    pricer_amer_put = OptionPricer(S=100, K=105, T=90/365, r=0.015, market_price=7.30)
    iv_amer_put = pricer_amer_put.implied_volatility_american(option_type='put')
    print(f"\nAmerican Put IV: {iv_amer_put:.2%}")
    print(f"American Put Delta: {pricer_amer_put.delta(iv_amer_put, 'put'):.4f}")
    print(f"American Put Gamma: {pricer_amer_put.gamma(iv_amer_put):.6f}")

    pricer_amer_call = OptionPricer(S=100, K=95, T=90/365, r=0.01, market_price=8.50, q=0.02)
    iv_amer_call = pricer_amer_call.implied_volatility_american(option_type='call')
    print(f"\nAmerican Call IV (with dividends): {iv_amer_call:.2%}")
    print(f"American Call Delta: {pricer_amer_call.delta(iv_amer_call, 'call'):.4f}")
    print(f"American Call Gamma: {pricer_amer_call.gamma(iv_amer_call):.6f}")
    print(f"American Call Vega: {pricer_amer_call.vega(iv_amer_call):.4f}")
    print(f"American Call Theta: {pricer_amer_call.theta(iv_amer_call, 'call'):.4f}")