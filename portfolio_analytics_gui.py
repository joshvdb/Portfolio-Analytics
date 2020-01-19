import os
import io
import base64
import numpy as np
import pandas as pd
from numba import jit
import statsmodels.api as sm
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import integrate, stats
from flask import Flask, send_file


@jit(nopython=True, cache=False)
def sma(matrix, interval):
    """
    Function to implement a Simple Moving Average (SMA), optimized with Numba.
    :param matrix: np.array([float])
    :param interval: int
    :return: np.array([float])
    """

    # declare empty SMA numpy array
    s = np.zeros((matrix.shape[0] - interval))

    # calculate the value of each point in the Simple Moving Average array
    for t in range(0, s.shape[0]):
        s[t] = np.sum(matrix[t:t + interval])/interval

    return s


@jit(nopython=True, cache=False)
def ema(matrix, alpha):
    """
    Function to implement an Exponential Moving Average (EMA), optimized with Numba. The variable alpha represents the
    degree of weighting decrease, a constant smoothing factor between 0 and 1. A higher alpha discounts older
    observations faster.
    :param matrix: np.array([float])
    :param alpha: float
    :return: np.array([float])
    """

    # declare empty EMA numpy array
    e = np.zeros(matrix.shape[0])

    # set the value of the first element in the EMA array
    e[0] = matrix[0]

    # use the EMA formula to calculate the value of each point in the EMA array
    for t in range(1, matrix.shape[0]):
        e[t] = alpha*matrix[t] + (1 - alpha)*e[t - 1]

    return e


@jit(nopython=True, cache=False)
def twap(high, low, open, close, interval):
    """
    Function to implement a Time-Weighted Average Price (TWAP), optimized with Numba.
    :param high: np.array([float])
    :param low: np.array([float])
    :param open: np.array([float])
    :param close: np.array([float])
    :param interval: int
    :return: np.array([float])
    """

    # calculate prices data for each day
    prices = (high + low + open + close) / 4

    # declare empty TWAP numpy array
    p = np.zeros((prices.shape[0] - interval))

    # calculate the value of each point in the TWAP array
    for t in range(0, p.shape[0]):
        p[t] = np.sum(prices[t:t + interval]) / interval

    return p


@jit(nopython=True, cache=False)
def vwap(high, low, close, volumes, interval):
    """
    Function to implement a Volume-Weighted Average Price (VWAP), optimized with Numba.
    :param high: np.array([float])
    :param low: np.array([float])
    :param close: np.array([float])
    :param volumes: np.array([float])
    :param interval: int
    :return: np.array([float])
    """

    # calculate prices data for each day
    prices = (high + low + close) / 3

    # declare empty VWAP numpy array
    p = np.zeros((prices.shape[0] - interval))

    # calculate the value of each point in the VWAP array
    for t in range(0, p.shape[0]):
        p[t] = np.sum(prices[t:t + interval]*volumes[t:t + interval]) / np.sum(volumes[t:t + interval])

    return p


def portfolio_returns(returns, weights):
    """
    Calculate the total portfolio returns time-series array by multiplying the weights matrix (dimensions = 1*N) and the
    portfolio matrix (dimensions = N*t), for N securities in the portfolio and t returns per security. This function
    returns a 1*t matrix where each element in the matrix is the portfolio return for a given time.
    :param returns: np.array([float])
    :param weights: np.array([float])
    :return: np.array([float])
    """

    # the portfolio returns are given by the dot product of the weights matrix and the portfolio matrix
    port_returns = np.dot(weights, returns)

    return port_returns


def alpha_rf(port_returns, risk_free_rate, market_returns, b):
    """
    Calculate the Alpha of the portfolio.
    :param port_returns: np.array([float])
    :param risk_free_rate: np.array([float])
    :param market_returns: np.array([float])
    :param b: float
    :return: float
    """

    # the portfolio Alpha is given by the below equation, as stated by the Capital Asset Pricing Model
    alpha = np.mean(port_returns) - risk_free_rate + b*(np.mean(market_returns) - risk_free_rate)

    return alpha


def portfolio_analytics(port_returns, market_returns):
    """
    Perform a regression on the portfolio returns and benchmark returns to calculate the Alpha, beta, and R-squared of
    the portfolio. This function will also return a numpy array containing the regression prediction values.
    :param port_returns: np.array([float])
    :param market_returns: np.array([float])
    :return: float, float, float, np.array([float])
    """

    # add the intercept to the model
    x2 = sm.add_constant(market_returns)

    # train the model
    estimator = sm.OLS(port_returns, x2)
    model = estimator.fit()

    # get portfolio analytics
    alpha, beta = model.params
    r_squared = model.rsquared
    regression = model.predict()

    return alpha, beta, r_squared, regression


def portfolio_volatility(returns, weights):
    """
    Calculate the total portfolio volatility (the variance of the historical returns of the portfolio) using a
    covariance matrix.
    :param returns: np.array([float])
    :param weights: np.array([float])
    :return: float
    """

    # generate the transform of the 1D numpy weights array
    w_T = np.array([[x] for x in weights])

    # calculate the covariance matrix of the asset returns
    covariance = np.cov(returns)

    # calculate the portfolio volatility
    port_volatility = np.dot(np.dot(weights, covariance), w_T)[0]

    return port_volatility


def sharpe_ratio(port_returns, risk_free_rate, asset_returns, weights):
    """
    Calculate the Sharpe ratio of the portfolio, based on the latest portfolio return value.
    :param port_returns: np.array([float])
    :param risk_free_rate: float
    :param asset_returns: np.array([float])
    :param weights: np.array([float])
    :return: float
    """

    # calculate the standard deviation of the returns of the portfolio
    portfolio_standard_deviation = np.sqrt(portfolio_volatility(asset_returns, weights))

    # calculate the Sharpe ratio of the portfolio
    sr = (port_returns[-1] - risk_free_rate)/portfolio_standard_deviation

    return sr


def treynor_ratio(port_returns, risk_free_rate, beta):
    """
    Calculate the Treynor ratio of the portfolio, based on the latest portfolio return value.
    :param port_returns: np.array([float])
    :param risk_free_rate: float
    :param beta: float
    :return: float
    """

    # calculate the Treynor ratio of the portfolio
    return (port_returns[-1] - risk_free_rate)/beta


def tracking_error(port_returns, market_returns):
    """
    Calculate the Tracking Error of the portfolio relative to the benchmark.
    :param port_returns: np.array([float])
    :param market_returns: np.array([float])
    :return: float
    """

    return np.std(port_returns - market_returns)


def risk_metrics(returns, weights, portfolio_vol, var_p, d):
    """
    Calculate the Analytical VaR and Expected Shortfall for a portfolio, using both the Normal distribution and the
    T-distribution.
    :param returns: np.array([float]) - the historical portfolio returns
    :param weights: np.array([float]) - the weights of the assets in the portfolio
    :param portfolio_vol: float - the volatility of the portfolio
    :param var_p: float - the value of the daily returns at which to take the Analytical VaR
    :param d: int - the number of degrees of freedom to use
    :return: float, float, float, float, float, float
    """

    # calculate the standard deviation of the portfolio
    sigma = np.sqrt(portfolio_vol)

    # calculate the mean return of the portfolio
    mu = np.sum(portfolio_returns(returns, weights))/returns.shape[1]

    # integrate the Probability Density Function to find the Analytical Value at Risk for both Normal and t distributions
    a_var = stats.norm(mu, sigma).cdf(var_p)
    t_dist_a_var = stats.t(d).cdf(var_p)

    # calculate the expected shortfall for each distribution - this is the expected loss (in % daily returns) for the
    # portfolio in the worst a_var% of cases - it is effectively the mean of the values along the x-axis from
    # -infinity% to a_var%
    es = (stats.norm.pdf(stats.norm.ppf((1 - a_var))) * sigma)/(1 - a_var) - mu
    percent_point_function = stats.t.ppf((1 - a_var), d)
    t_dist_es = -1/(1 - a_var) * (1-d)**(-1) * (d-2 + percent_point_function**2) * stats.t.pdf(percent_point_function, d)*sigma - mu
    
    return sigma, mu, a_var, t_dist_a_var, es, t_dist_es


def ci_specified_risk_metrics(returns, weights, portfolio_vol, ci, d):
    """
    Calculate the Analytical VaR and Expected Shortfall for a portfolio, using both the Normal distribution and the
    T-distribution.
    :param returns: np.array([float]) - the historical portfolio returns
    :param weights: np.array([float]) - the weights of the assets in the portfolio
    :param portfolio_vol: float - the volatility of the portfolio
    :param ci: float - the confidence interval at which to take the Analytical VaR
    :param d: int - the number of degrees of freedom to use
    :return: float, float, float, float, float, float
    """

    # calculate the standard deviation of the portfolio
    sigma = np.sqrt(portfolio_vol)

    # calculate the mean return of the portfolio
    mu = np.sum(portfolio_returns(returns, weights))/returns.shape[1]

    # integrate the Probability Density Function to find the Analytical Value at Risk for both Normal and t distributions
    var_level = stats.norm.ppf(ci, mu, sigma)
    t_dist_var_level = stats.t(d).ppf(ci)

    # calculate the expected shortfall for each distribution - this is the expected loss (in % daily returns) for the
    # portfolio in the worst a_var% of cases - it is effectively the mean of the values along the x-axis from
    # -infinity% to a_var%
    es = (stats.norm.pdf(stats.norm.ppf((1 - ci))) * sigma)/(1 - ci) - mu
    percent_point_function = stats.t.ppf((1 - ci), d)
    t_dist_es = -1 / (1 - ci) * (1 - d) ** (-1) * (d - 2 + percent_point_function ** 2) * stats.t.pdf(percent_point_function, d) * sigma - mu
    
    return sigma, mu, var_level, t_dist_var_level, es, t_dist_es


def historical_var(port_returns, var_p):
    """
    Calculate the Historical VaR for a portfolio.
    :param port_returns: np.array([float])
    :param var_p: float
    :return: float
    """

    # calculate the Historical VaR of the portfolio - check if the daily return value is less than a% - if it is, then
    # it is counted in the Historical VaR calculation
    relevant_returns = 0
    port_returns = np.sort(port_returns)
    for i in range(0, port_returns.shape[0]):
        if port_returns[i] < var_p:
            relevant_returns += 1

    h_var = relevant_returns/port_returns.shape[0]

    return h_var


def ci_historical_var(port_returns, ci):
    """
    Calculate the Historical VaR for a portfolio.
    :param port_returns: np.array([float])
    :param ci: float
    :return: float
    """

    # calculate the Historical VaR of the portfolio based on the confidence interval
    return np.quantile(np.sort(port_returns), ci)


def plot_analytical_var(var_p, sigma, mu, n, z, plot='Normal'):
    """
    Plot the normal or t distribution of portfolio returns, showing the area under the curve that corresponds to the
    Analytical VaR of the portfolio.
    :param var_p: float - the value of the daily returns at which to take the Analytical VaR
    :param sigma: float - the standard deviation
    :param mu: float - the mean
    :param n: int - the number of points to use in the plot
    :param z: float - the number of standard deviations from the mean to use as the plot range
    :param plot: str - (Normal or T-dist) - used to select the function to plot
    :return:
    """

    # set the plot range at z standard deviations from the mean in both the left and right directions
    plot_range = z * sigma

    # set the bottom value on the x-axis (of % daily returns)
    bottom = mu - plot_range

    # set the top value on the x-axis (of % daily returns)
    top = mu + plot_range

    # declare the numpy array of the range of x values for the normal distribution
    x = np.linspace(bottom, top, n)

    # calculate the index of the nearest daily return in x corresponding to a%
    risk_range = (np.abs(x - var_p)).argmin()

    # calculate the normal distribution pdf for plotting purposes
    if plot == 'Normal':
        pdf = stats.norm(mu, sigma).pdf(x)
    else:
        pdf = stats.t(sigma).pdf(x)
        plot = 't-dist'

    figure = plt.figure()
    axis = figure.add_subplot(111)

    axis.plot(x, pdf, linewidth=2, color='r', label='Distribution of Returns')
    axis.fill_between(x[0:risk_range], pdf[0:risk_range], facecolor='blue', label='Analytical VaR')
    axis.legend(loc='upper left')
    axis.set_xlabel('Daily Returns')
    axis.set_ylabel('Frequency')
    axis.set_title('Frequency vs Daily Returns (' + plot + ')')

    return figure


def plot_historical_var(port_returns, var_p, num_plot_points):
    """
    Plot a histogram showing the distribution of portfolio returns - mark the cutoff point that corresponds to the
    Historical VaR of the portfolio. The variable x is the historical distribution of returns of the portfolio, a is
    the cutoff value, and the bins are the bins in which to stratify the historical returns.
    :param port_returns: np.array([float])
    :param var_p: float
    :param num_plot_points: int
    :return:
    """

    # create a numpy array of the bins to use for plotting the Historical VaR, based on the maximum and minimum values
    # of the portfolio returns, and the number of plot points to include
    bins = np.linspace(np.sort(port_returns)[0], np.sort(port_returns)[-1], num_plot_points)

    figure = plt.figure()
    axis = figure.add_subplot(111)

    axis.hist(np.sort(port_returns), bins, label='Distribution of Returns')
    axis.axvline(x=var_p, ymin=0, color='r', label='Historical VaR cutoff value')
    axis.legend(loc='upper left')
    axis.set_xlabel('Daily Returns')
    axis.set_ylabel('Frequency')
    axis.set_title('Frequency vs Daily Returns')

    return figure


def plot_historical_returns(port_returns, market_returns):
    """
    Function to plot the historical portfolio returns.
    :param port_returns: np.array([float])
    :param market_returns: np.array([float])
    :return:
    """

    # define x-axis data points
    x = np.linspace(0, port_returns.shape[0], port_returns.shape[0])

    figure = plt.figure()
    axis = figure.add_subplot(111)

    axis.plot(x, port_returns, linewidth=1, color='b', label='Portfolio Returns')
    axis.plot(x, market_returns, linewidth=1, color='r', label='Benchmark Returns')
    axis.legend(loc='upper left')
    axis.set_xlabel('Time (days)')
    axis.set_ylabel('Daily Return')
    axis.set_title('Daily Return vs Time')

    return figure


def plot_returns_regression(port_returns, market_returns, regression):
    """
    Function to plot the Returns Regression.
    :param port_returns: np.array([float])
    :param market_returns: np.array([float])
    :param regression: np.array([float])
    :return:
    """

    figure = plt.figure()
    axis = figure.add_subplot(111)

    axis.scatter(market_returns, port_returns, marker='.', linewidth=1, color='b', label='Actual Returns')
    axis.plot(market_returns, regression, linewidth=1, color='k', label='Returns Regression')
    axis.legend(loc='upper left')
    axis.set_xlabel('Benchmark Daily Return')
    axis.set_ylabel('Portfolio Daily Return')
    axis.set_title('Returns Regression')

    return figure


def plot_equity_prices(ticker, prices):
    """
    Function to plot the prices of a single equity.
    :param ticker: str
    :param prices: np.array([float])
    :return:
    """

    # define x-axis data points
    x = np.linspace(0, prices.shape[0], prices.shape[0])

    figure = plt.figure()
    axis = figure.add_subplot(111)

    axis.plot(x, prices[ticker], linewidth=1, color='b', label=ticker)
    axis.legend(loc='upper left')
    axis.set_xlabel('Time (days)')
    axis.set_ylabel('Price')
    axis.set_title('Price vs Time: ' + ticker)

    return figure


def equity_price_analytics(ticker, high_prices, low_prices, open_prices, close_prices, volume, alpha, interval):
    """
    Function to calculate the EMA, SMA, TWAP and VWAP of a single equity.
    :param ticker: str
    :param high_prices: pd.DataFrame([float])
    :param low_prices: pd.DataFrame([float])
    :param open_prices: pd.DataFrame([float])
    :param close_prices: pd.DataFrame([float])
    :param volume: pd.DataFrame([float])
    :param alpha: float
    :param interval: int
    :return: np.array([float]), np.array([float]), np.array([float]), np.array([float]), np.array([float])
    """

    # get price and volume data in numpy array form
    high = np.array(high_prices[ticker])
    low = np.array(low_prices[ticker])
    open = np.array(open_prices[ticker])
    close = np.array(close_prices[ticker])
    volume = np.array(volume[ticker])

    # calculate price analytics
    e_ma = ema(close, alpha)
    s_ma = sma(close, interval)
    t_wap = twap(high, low, open, close, interval)
    v_wap = vwap(high, low, close, volume, interval)

    return e_ma, s_ma, t_wap, v_wap, close


def plot_equity_price_analytics(ticker, e_ma, s_ma, t_wap, v_wap, close_prices, interval):
    """
    Function to plot the mean price, EMA, SMA, TWAP and VWAP of a single equity.
    :param ticker: str
    :param e_ma: np.array([float])
    :param s_ma: np.array([float])
    :param t_wap: np.array([float])
    :param v_wap: np.array([float])
    :param close_prices: np.array([float])
    :param interval: int
    :return:
    """

    # define x-axis data points
    x = np.linspace(0, close_prices.shape[0], close_prices.shape[0])

    figure = plt.figure()
    axis = figure.add_subplot(111)

    axis.plot(x, close_prices, linewidth=1, color='k', label='Close Price')
    axis.plot(x, e_ma, linewidth=1, color='r', label='EMA Price')
    axis.plot(x[interval:], s_ma, linewidth=1, color='b', label='SMA Price')
    axis.plot(x[interval:], t_wap, linewidth=1, color='g', label='TWAP Price')
    axis.plot(x[interval:], v_wap, linewidth=1, color='m', label='VWAP Price')
    axis.legend(loc='upper left')
    axis.set_xlabel('Time (days)')
    axis.set_ylabel('Price')
    axis.set_title('Price vs Time: ' + ticker)

    return figure


def plot_equity_returns(ticker, returns):
    """
    Function to plot the returns of a single equity.
    :param ticker: str
    :param returns: np.array([float])
    :return:
    """

    # define x-axis data points
    x = np.linspace(0, returns.shape[0], returns.shape[0])

    figure = plt.figure()
    axis = figure.add_subplot(111)

    axis.plot(x, returns[ticker], linewidth=1, color='b', label=ticker)
    axis.legend(loc='upper left')
    axis.set_xlabel('Time (days)')
    axis.set_ylabel('Daily Return')
    axis.set_title('Return vs Time for: ' + ticker)

    return figure


def get_data(data):
    """
    Function to convert pandas DataFrames into numpy array's of shape (n*m), where n = the number of equities and m =
    the number of days for which we have price or return data.
    :param data: pd.DataFrame([float])
    :return: np.array([float])
    """

    np_data = np.array(data)
    array = []

    for i in range(0, np_data.shape[1]):
        array.append(np_data[:, i])

    return np.array(array)
