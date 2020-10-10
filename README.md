# Portfolio Analytics

## Overview
Python-based program that delivers CAPM portfolio analytics and algorithmic trading analytics using time-series data (price/returns) for equities.

This project uses the Capital Asset Pricing Model and algorithmic trading analytics to gain insight into an equity portfolio. It allows a user to select a set of equities (by ticker), and plots a range of standard portfolio analytics (volatility, VaR, Expected Shortfall, Sharpe Ratio, etc), and algorithmic trading analytics (SMA, EMA, VWAP, TWAP). The GUI for this program is written in HTML5, using the Flask package to interface with Python.

## Installation

The requirements file can be installed using the below commands in a terminal:

**Conda:** conda install --file requirements.txt

**Pip:** pip install -r requirements.txt

## Current Analytics

1. Portfolio R-Squared
2. Portfolio Beta
3. Portfolio Volatility
4. Portfolio Alpha (calculated from the returns regression)
5. Portfolio Alpha (based on risk-free rate)
6. Portfolio Sharpe Ratio
7. Portfolio Treynor Ratio
8. Portfolio Tracking Error
9. Analytical VaR (Normal Distribution)
10. Analytical VaR (t-Distribution)
11. Expected Shortfall (Normal Distribution)
12. Expected Shortfall (t-Distribution)
13. Historical VaR

## Example Outputs

Using the equities: AAPL, GOOGL, BLK, and IBM (with respective portfolio wieghts of: 0.15, 0.6, 0.2, and 0.05), and the S&P 500 (^GSPC) as the benchmark, in the date range: 2019-06-20 to 2019-11-08, we obtain the below portfolio analytics.

1. Portfolio R-Squared = 0.5713749319708816
2. Portfolio Beta = 1.6823252286102592
3. Portfolio Volatility = 0.001774262341931516
4. Portfolio Alpha (calculated from the returns regression) = 0.043032257203536886
5. Portfolio Alpha (based on risk-free rate) = 0.0052482979658219735
6. Portfolio Sharpe Ratio = 3.354557106775981
7. Portfolio Tracking Error = 0.030298260899050793
8. Analytical VaR (Normal Distribution) = 7.390523472420238% at -1.0% of daily returns
9. Analytical VaR (t-Distribution) = 49.62040090212909% at -1.0% of daily returns
10. Expected Shortfall (Normal Distribution) at 5.0% level = 3.592204573450777
11. Expected Shortfall (t-Distribution) at 5.0% level = 7.077446307947621
12. Historical VaR = 4.0% at -1.0% of daily returns

**Daily Return vs Time (Portfolio and Benchmark)**
![alt text](https://github.com/jjvdb/Portfolio-Analytics/blob/master/Diagrams/Daily%20Return%20vs%20Time.png)

**Returns Regression**
![alt text](https://github.com/jjvdb/Portfolio-Analytics/blob/master/Diagrams/Returns%20Regression.png)

**Analytical VaR (Normal Distribution)**
![alt text](https://github.com/jjvdb/Portfolio-Analytics/blob/master/Diagrams/Analytical%20VaR%20(Normal%20Distribution).png)

**Analytical VaR (t Distribution)**
![alt text](https://github.com/jjvdb/Portfolio-Analytics/blob/master/Diagrams/Analytical%20VaR%20(t%20Distribution).png)

**Historical VaR**
![alt text](https://github.com/jjvdb/Portfolio-Analytics/blob/master/Diagrams/Historical%20VaR.png)

**Trading Analytics (Apple)**
![alt text](https://github.com/jjvdb/Portfolio-Analytics/blob/master/Diagrams/Trading%20Analytics%20(Apple).png)

**Price vs Time (Apple)**
![alt text](https://github.com/jjvdb/Portfolio-Analytics/blob/master/Diagrams/Price%20vs%20Time%20(Apple).png)

**Return vs Time (Apple)**
![alt text](https://github.com/jjvdb/Portfolio-Analytics/blob/master/Diagrams/Return%20vs%20Time%20(Apple).png)

## Graphical User Interface Guide

A sample of the Graphical User Interface is shown in the file: Portfolio Analytics Sample Page.png.

**Get Prices Tab**

This tab contains two buttons that are used to get the most recent 5 months of prices for a set of equities and the benchmark.

![alt text](https://github.com/jjvdb/Portfolio-Analytics/blob/master/Diagrams/Get%20Prices%20Tab.png)

**Variables Tab**

This tab lists the variables that are used to generate the portfolio analytics (together with the price data). The explanation for each variable is given below:

1. Moving-average Interval: the number of days to use in the calculation of the Simple Moving Average.
2. Moving-average Alpha Value: represents the degree of weighting decrease, a constant smoothing factor between 0 and 1. A higher alpha discounts older observations faster.
3. Equity Ticker: The ticker of the equity to look at for algorithmic trading analytics.
4. VaR Threshold (% Daily Returns): the % daily returns threshold to use in the calculation of VaR (Historical and Analytical). For example, entering -0.01 means that we will calculate the probability of losing more than 1% of the value of our portfolio (the Expected Shortfall will be calculated based on this value).
5. VaR Threshold (Confidence Interval): the Confidence Interval to use in the calculation of the VaR (this is another way of specifying the VaR). For example, entering 0.05 means that we will calculate the % daily returns that we can expect to lose at 5% VaR (that is, the amount that we expect to lose/gain with a probability of 5%).
6. Risk-free Rate: this is the risk-free rate that is used in the calculation of certain portfolio analytics (e.g. a 0.02 = 2%). This is often set as the interest rate on a 3 month U.S. Treasury Bill.

![Intro.pdf](https://github.com/jjvdb/Portfolio-Analytics/blob/master/Diagrams/Variables%20Tab.png)

**Example Output**

This image shows an example of the text ouput of the program - all portfolio analytics are listed here. These can be used, together with the graphs produced by the program, to gain an understanding of an equity portfolio.

![alt text](https://github.com/jjvdb/Portfolio-Analytics/blob/master/Diagrams/Example%20Output.png)

## Limitations

1. The project uses the Alpha Advantage package to obtain price data. Due to the limited number of server requests this API allows, we currently save the data into Excel files, which are then read in by the main program, Portfolio-Analytics.py for processing. The Alpha Advantage model only allows us to obtain the prices and volume for each equity for the last 100 days

## Outstanding Enhancements

1. Portfolio breakdown by sector
2. Portfolio breakdown by country
3. Portfolio breakdown by region
4. Returns attribution by equity
5. Additional high-frequency trading analytics
6. Portfolio Optimizer
7. Seaborn plots
8. Inclusion of other asset classes
9. Plot Expected Shortfall as an overlay on VaR plots
10. Monte Carlo (MC) VaR and other MC-based analytics
11. stress-testing scenarios
12. Returns Attribution (Brinson, Frongello, risk-factor attribution)
