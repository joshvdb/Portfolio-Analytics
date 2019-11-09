# Portfolio Analytics
Python-based project to deliver CAPM portfolio analytics and high-frequency trading analytics using based on equity time-series data (price/returns).

This project uses the Capital Asset Pricing Model and high-frequency trading analytics to gain insight into an equity portfolio. It allows a user to select a set of equities (by ticker), and plots a range of standard portfolio metrics (volatility, VaR, Expected Shortfall, Sharpe Ratio, etc), and high-frequency trading analytics (SMA, EMA, VWAP, TWAP).

# Limitations

1. The project uses the Alpha Advantage package to obtain price data. Due to the limited number of allowed server requests, we currently save the data into Excel files, which are then read in by the main program, Portfolio-Analytics.py for processing. The Alpha Advantage model only allows us to obtain the prices and volume for each equity for the last 100 days

# Outstanding Enhancements

1. Portfolio breakdown by sector
2. Portfolio breakdown by country
3. Portfolio breakdown by region
4. Returns attribution by equity
5. Additional high-frequency trading analytics
6. Portfolio Optimizer
7. Seaborn plots
8. Expansion to other asset classes
