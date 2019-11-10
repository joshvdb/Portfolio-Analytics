# Portfolio Analytics
Python-based project to deliver CAPM portfolio analytics and high-frequency trading analytics using based on equity time-series data (price/returns).

This project uses the Capital Asset Pricing Model and high-frequency trading analytics to gain insight into an equity portfolio. It allows a user to select a set of equities (by ticker), and plots a range of standard portfolio metrics (volatility, VaR, Expected Shortfall, Sharpe Ratio, etc), and high-frequency trading analytics (SMA, EMA, VWAP, TWAP).

# Current Analytics

1. Portfolio R-Squared
2. Portfolio Beta
3. Portfolio Volatility
4. Portfolio Alpha (calculated from the returns regression)
Portfolio Alpha (based on risk-free rate)
Portfolio Sharpe Ratio
Portfolio Tracking Error
Analytical VaR (Normal Distribution)
Analytical VaR (t-Distribution)
Expected Shortfall (Normal Distribution)
Expected Shortfall (t-Distribution)
Historical VaR

# Expected Outputs

Portfolio R-Squared = 0.5713749319708816
Portfolio Beta = 1.6823252286102592
Portfolio Volatility = 0.001774262341931516
Portfolio Alpha (from the regression) = 0.043032257203536886
Portfolio Alpha (based on risk-free rate) = 0.0052482979658219735
Portfolio Sharpe Ratio = 3.354557106775981
Portfolio Tracking Error = 0.030298260899050793
Analytical VaR (Normal) = 7.390523472420238% at -1.0% of daily returns
Analytical VaR (t-distribution) = 49.62040090212909% at -1.0% of daily returns
Expected Shortfall (Normal) at 5.0% level = 3.592204573450777
Expected Shortfall (t-distribution) at 5.0% level = 7.077446307947621
Historical VaR = 4.0% at -1.0% of daily returns

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
8. Inclusion of other asset classes
9. Plot Expected Shortfall as an overlay on VaR plots
