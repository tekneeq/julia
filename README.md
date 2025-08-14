# Options Greeks Calculator

A Python tool for calculating options Greeks (Delta, Gamma, Vega, Theta) using Black-Scholes pricing models and real-time data from Robinhood API.

## Features

### Options Pricing & Greeks
- **Delta**: Price sensitivity to underlying asset price changes
- **Gamma**: Rate of change of delta with respect to underlying price
- **Vega**: Price sensitivity to volatility changes
- **Theta**: Time decay (daily)
- **Implied Volatility**: Both European (Black-Scholes) and American (Binomial Tree)

### Gamma Exposure (GEX) Analysis
- **GEX Calculation**: Calculate Gamma Exposure for individual contracts and portfolio-level analysis
- **Gamma Positioning**: Determine if market is in Long Gamma, Short Gamma, or Gamma Neutral environment
- **Key Gamma Levels**: Identify strikes with highest gamma exposure that act as support/resistance
- **Market Impact Analysis**: Understand expected dealer hedging flows and volatility patterns

### CLI Commands

#### Calculate Greeks for All Options on a Given Day
```bash
python main.py greeks --ticker SPY --expiration 2024-01-19
```

**Options:**
- `--ticker`: Stock ticker symbol (default: SPY)
- `--expiration`: Expiration date in YYYY-MM-DD format (default: next business day)
- `--rate`: Risk-free interest rate (default: 0.02 or 2%)
- `--output`: Save results to CSV file (optional)
- `--min-volume`: Filter options by minimum volume (default: 0)
- `--show-all`: Show all Greeks instead of just delta and gamma
- `--show-gex`: Show Gamma Exposure (GEX) analysis and positioning

**Examples:**
```bash
# Basic usage - calculate Greeks for SPY options expiring tomorrow
python main.py greeks

# Calculate Greeks for AAPL options expiring on a specific date
python main.py greeks --ticker AAPL --expiration 2024-02-16

# Show all Greeks with volume filter and save to CSV
python main.py greeks --ticker TSLA --expiration 2024-01-26 --show-all --min-volume 10 --output tesla_greeks.csv

# Show GEX analysis to determine gamma positioning
python main.py greeks --ticker SPY --show-gex

# Complete analysis with all Greeks and GEX
python main.py greeks --ticker AAPL --expiration 2024-02-16 --show-all --show-gex
```

#### Calculate Implied Move
```bash
python main.py emove --ticker SPY --days 1,3,5 --confidence 0.68,0.95
```

## Installation

1. Install dependencies:
```bash
pip install numpy scipy pandas robin-stocks python-dotenv click
```

2. Set up Robinhood credentials in `.env` file:
```
RH_USERNAME=your_username
RH_PASSWORD=your_password
```

## Usage

### Programmatic Usage

```python
from options import OptionPricer

# Create option pricer
pricer = OptionPricer(S=100, K=105, T=30/365, r=0.02, market_price=2.50)

# Calculate implied volatility
iv = pricer.implied_volatility_bs(option_type='put')

# Calculate Greeks
delta = pricer.delta(iv, 'put')
gamma = pricer.gamma(iv)
vega = pricer.vega(iv)
theta = pricer.theta(iv, 'put')

# Calculate GEX (Gamma Exposure)
open_interest = 1000  # Example open interest
gex_per_contract = pricer.gex_per_contract(iv, open_interest, 'put')
gex_notional = pricer.gex_notional(iv, open_interest, 'put')

print(f"Delta: {delta:.4f}")
print(f"Gamma: {gamma:.6f}")
print(f"Vega: {vega:.4f}")
print(f"Theta: {theta:.4f}")
print(f"GEX per Contract: ${gex_per_contract:,.0f}")
print(f"Notional GEX: ${gex_notional:,.0f}")

# Portfolio-level GEX analysis
gex_values = [gex_per_contract, -50000, 25000]  # Example GEX values
portfolio_analysis = OptionPricer.calculate_portfolio_gex(gex_values)
print(f"Gamma Position: {portfolio_analysis['gamma_position']}")
print(f"Net GEX: ${portfolio_analysis['net_gex']:,.0f}")
```

### Bulk Greeks Calculation

```python
from main import calculate_options_greeks

# Calculate Greeks for all SPY options expiring on 2024-01-19
df = calculate_options_greeks('SPY', '2024-01-19', risk_free_rate=0.02)

# Access portfolio-level GEX analysis
if hasattr(df, 'attrs') and 'portfolio_gex' in df.attrs:
    gex_analysis = df.attrs['portfolio_gex']
    print(f"Portfolio Gamma Position: {gex_analysis['gamma_position']}")
    print(f"Total GEX: ${gex_analysis['total_gex']:,.0f}")

print(df)
```

## Output Format

The Greeks calculation returns a pandas DataFrame with the following columns:

- `ticker`: Stock ticker symbol
- `expiration_date`: Option expiration date
- `option_type`: 'CALL' or 'PUT'
- `strike_price`: Strike price of the option
- `market_price`: Current market price
- `theoretical_price`: Black-Scholes theoretical price
- `bid_price` / `ask_price`: Bid and ask prices
- `implied_volatility`: Implied volatility
- `delta`: Delta value
- `gamma`: Gamma value
- `vega`: Vega value (when --show-all is used)
- `theta`: Theta value (when --show-all is used)
- `gex_per_contract`: Gamma Exposure per contract in dollars (when --show-gex is used)
- `gex_notional`: Notional Gamma Exposure in dollars (when --show-gex is used)
- `volume`: Trading volume
- `open_interest`: Open interest
- `time_to_expiry_days`: Days until expiration

### GEX Analysis Output

When using `--show-gex`, additional analysis is provided:

- **Portfolio GEX Summary**: Total, Call, and Put GEX values
- **Gamma Position**: LONG GAMMA, SHORT GAMMA, or GAMMA NEUTRAL
- **Key Gamma Levels**: Top strikes by absolute GEX
- **Support/Resistance**: Nearest GEX levels above/below current price
- **Market Impact**: Expected dealer hedging behavior

## Notes

- Greeks are calculated using Black-Scholes formulas for European-style options
- GEX calculations assume market makers are net short options (standard assumption)
- Requires valid Robinhood credentials for fetching real-time options data
- Risk-free rate defaults to 2% but can be adjusted
- Options without valid pricing or implied volatility are filtered out
- GEX analysis helps predict market volatility and dealer flow patterns