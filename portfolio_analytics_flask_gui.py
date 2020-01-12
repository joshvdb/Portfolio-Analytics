import io
import pickle
import pandas as pd
from portfolio_analytics_gui import *
from alpha_vantage.timeseries import TimeSeries
from flask import Flask, request, render_template, Response, make_response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('portfolio_analytics.html',
                           equity_list=pickle.load(open('equities/user_input_equities.p', 'rb')),
                           benchmark_ticker=pickle.load(open('equities/user_input_benchmark.p', 'rb')),
                           ma_interval_box=pickle.load(open('equities/ma_interval.p', 'rb')),
                           ma_alpha_box=pickle.load(open('equities/ma_alpha.p', 'rb')),
                           ma_ticker_box=pickle.load(open('equities/ma_ticker.p', 'rb')),
                           var_p_box=pickle.load(open('equities/var_p.p', 'rb')),
                           ci_box=pickle.load(open('equities/ci.p', 'rb')),
                           rf_box=pickle.load(open('equities/rf.p', 'rb')),
                           portfolio_analytics_values=pickle.load(open('equities/portfolio_analytics_values.p', 'rb')))


@app.route('/get_equity_prices', methods=['GET', 'POST'])
def get_equity_prices():

    equity_input = str(request.form['equity_input'])

    # save user entry for later default display
    pickle.dump(equity_input.strip(), open('equities/user_input_equities.p', 'wb'))

    equities = [s[1:s.index(':')] for s in equity_input.strip().split(', ')]
    quantities = list(int(s[s.index(':') + 2:-1]) for s in equity_input.strip().split(', '))

    key = 'yourkeyhere'

    high_prices = pd.DataFrame({'A': []})
    low_prices = pd.DataFrame({'A': []})
    open_prices = pd.DataFrame({'A': []})
    close_prices = pd.DataFrame({'A': []})
    volumes = pd.DataFrame({'A': []})

    for equity in equities:
        ts = TimeSeries(key, output_format='pandas')
        ticker, meta = ts.get_daily(symbol=equity)

        high_prices = pd.concat([high_prices, ticker['1. open']], axis=1)
        low_prices = pd.concat([low_prices, ticker['2. high']], axis=1)
        open_prices = pd.concat([open_prices, ticker['3. low']], axis=1)
        close_prices = pd.concat([close_prices, ticker['4. close']], axis=1)
        volumes = pd.concat([volumes, ticker['5. volume']], axis=1)

    pickle.dump(quantities, open('equities/quantities.p', 'wb'))

    high_prices = high_prices.dropna(axis=1)
    high_prices.columns = equities
    low_prices = low_prices.dropna(axis=1)
    low_prices.columns = equities
    open_prices = open_prices.dropna(axis=1)
    open_prices.columns = equities
    close_prices = close_prices.dropna(axis=1)
    close_prices.columns = equities
    volumes = volumes.dropna(axis=1)
    volumes.columns = equities

    high_prices.to_csv('equities/High Prices.csv')
    low_prices.to_csv('equities/Low Prices.csv')
    open_prices.to_csv('equities/Open Prices.csv')
    close_prices.to_csv('equities/Close Prices.csv')
    volumes.to_csv('equities/Volumes.csv')

    return render_template('portfolio_analytics.html',
                           equity_list=pickle.load(open('equities/user_input_equities.p', 'rb')),
                           benchmark_ticker=pickle.load(open('equities/user_input_benchmark.p', 'rb')),
                           ma_interval_box=pickle.load(open('equities/ma_interval.p', 'rb')),
                           ma_alpha_box=pickle.load(open('equities/ma_alpha.p', 'rb')),
                           ma_ticker_box=pickle.load(open('equities/ma_ticker.p', 'rb')),
                           var_p_box=pickle.load(open('equities/var_p.p', 'rb')),
                           ci_box=pickle.load(open('equities/ci.p', 'rb')),
                           rf_box=pickle.load(open('equities/rf.p', 'rb')),
                           portfolio_analytics_values=pickle.load(open('equities/portfolio_analytics_values.p', 'rb')))


@app.route('/get_benchmark_prices', methods=['GET', 'POST'])
def get_benchmark_prices():

    benchmark_input = str(request.form['benchmark_input'])

    # save user entry for later default display
    pickle.dump(benchmark_input.strip(), open('equities/user_input_benchmark.p', 'wb'))

    equity = benchmark_input.strip()

    key = 'yourkeyhere'

    close_prices = pd.DataFrame({'A': []})

    ts = TimeSeries(key, output_format='pandas')
    ticker, meta = ts.get_daily(symbol=equity)
    close_prices = pd.concat([close_prices, ticker['4. close']], axis=1)
    close_prices = close_prices.dropna(axis=1)
    close_prices.columns = [equity]
    close_prices.to_csv('equities/Benchmark Prices.csv')

    return render_template('portfolio_analytics.html',
                           equity_list=pickle.load(open('equities/user_input_equities.p', 'rb')),
                           benchmark_ticker=pickle.load(open('equities/user_input_benchmark.p', 'rb')),
                           ma_interval_box=pickle.load(open('equities/ma_interval.p', 'rb')),
                           ma_alpha_box=pickle.load(open('equities/ma_alpha.p', 'rb')),
                           ma_ticker_box=pickle.load(open('equities/ma_ticker.p', 'rb')),
                           var_p_box=pickle.load(open('equities/var_p.p', 'rb')),
                           ci_box=pickle.load(open('equities/ci.p', 'rb')),
                           rf_box=pickle.load(open('equities/rf.p', 'rb')),
                           portfolio_analytics_values=pickle.load(open('equities/portfolio_analytics_values.p', 'rb')))


def get_returns():

    # get equity price data
    close_prices = pd.read_csv('equities/Close Prices.csv')

    quantities = pickle.load(open('equities/quantities.p', 'rb'))

    # calculate the weight of each asset in the portfolio - each element in the row corresponds to the weight of the Nth asset
    w = np.array(list(q / sum(quantities) for q in quantities))

    close_prices = close_prices.drop(['date'], axis=1)

    # get the S&P500 benchmark price data
    benchmark = pd.read_csv('equities/Benchmark Prices.csv')
    benchmark = benchmark.drop(['date'], axis=1)

    # calculate the daily returns of the equities and the S&P500 benchmark
    equity_returns = close_prices.divide(close_prices.iloc[0])
    benchmark_returns = benchmark.divide(benchmark.iloc[0])

    # get historical return data for all equities
    eq_returns = get_data(equity_returns) - 1

    # calculate the historical returns of the portfolio
    port_returns = portfolio_returns(eq_returns, w)

    # declare the historical returns of the benchmark index
    market_returns = np.array(benchmark_returns)[:, 0] - 1

    return w, equity_returns, eq_returns, port_returns, market_returns


@app.route('/returns_plot.png')
def plot_returns():

    # calculate returns for the portfolio and the benchmark
    _, _, _, port_returns, market_returns = get_returns()

    figure = plot_historical_returns(port_returns, market_returns)

    output = io.BytesIO()
    FigureCanvas(figure).print_png(output)

    return Response(output.getvalue(), mimetype='image/png')


@app.route('/returns_regression_plot.png')
def gui_plot_returns_regression():

    # calculate returns for the portfolio and the benchmark
    _, _, _, port_returns, market_returns = get_returns()

    # calculate portfolio analytics
    _, _, _, regression = portfolio_analytics(port_returns, market_returns)

    figure = plot_returns_regression(port_returns, market_returns, regression)

    output = io.BytesIO()
    FigureCanvas(figure).print_png(output)

    return Response(output.getvalue(), mimetype='image/png')


@app.route('/var_normal_plot.png')
def var_normal_plot():

    # calculate returns for the portfolio and the benchmark
    w, _, eq_returns, port_returns, market_returns = get_returns()

    # calculate portfolio analytics
    alpha, beta, r_squared, _ = portfolio_analytics(port_returns, market_returns)

    # set the VaR percentage
    var_p = pickle.load(open('equities/var_p.p', 'rb'))

    # calculate the portfolio volatility
    portfolio_vol = portfolio_volatility(eq_returns, w)

    # calculate the Analytical VaR and the associated values
    sigma, mu, a_var, t_dist_a_var, es, t_dist_es = risk_metrics(eq_returns, w, portfolio_vol, var_p, 5)

    # plot the Analytical VaR
    figure = plot_analytical_var(var_p, sigma, mu, 10000, 10, 'Normal')

    output = io.BytesIO()
    FigureCanvas(figure).print_png(output)

    return Response(output.getvalue(), mimetype='image/png')


@app.route('/var_t_dist_plot.png')
def var_t_dist_plot():

    # calculate returns for the portfolio and the benchmark
    w, _, eq_returns, port_returns, market_returns = get_returns()

    # calculate portfolio analytics
    alpha, beta, r_squared, _ = portfolio_analytics(port_returns, market_returns)

    # set the VaR percentage
    var_p = pickle.load(open('equities/var_p.p', 'rb'))

    # calculate the portfolio volatility
    portfolio_vol = portfolio_volatility(eq_returns, w)

    # calculate the Analytical VaR and the associated values
    sigma, mu, a_var, t_dist_a_var, es, t_dist_es = risk_metrics(eq_returns, w, portfolio_vol, var_p, 5)

    # plot the Analytical VaR
    figure = plot_analytical_var(var_p, sigma, mu, 10000, 100, 't-dist')

    output = io.BytesIO()
    FigureCanvas(figure).print_png(output)

    return Response(output.getvalue(), mimetype='image/png')


@app.route('/h_var_plot.png')
def h_var_plot():

    # calculate returns for the portfolio and the benchmark
    w, _, eq_returns, port_returns, market_returns = get_returns()

    # set the VaR percentage
    var_p = pickle.load(open('equities/var_p.p', 'rb'))

    # calculate  the Historical VaR
    h_var = historical_var(port_returns, var_p)

    # plot the Historical VaR
    figure = plot_historical_var(port_returns, var_p, 100)

    output = io.BytesIO()
    FigureCanvas(figure).print_png(output)

    return Response(output.getvalue(), mimetype='image/png')


@app.route('/equity_prices_plot.png')
def equity_prices_plot():

    close_prices = pd.read_csv('equities/Close Prices.csv')
    close_prices = close_prices.drop(['date'], axis=1)

    # define the ticker of the stock to analyze
    ticker = pickle.load(open('equities/ma_ticker.p', 'rb'))

    # plot the prices of a single equity
    figure = plot_equity_prices(ticker, close_prices)

    output = io.BytesIO()
    FigureCanvas(figure).print_png(output)

    return Response(output.getvalue(), mimetype='image/png')


@app.route('/equity_returns_plot.png')
def equity_returns_plot():

    close_prices = pd.read_csv('equities/Close Prices.csv')
    close_prices = close_prices.drop(['date'], axis=1)

    equity_returns = close_prices.divide(close_prices.iloc[0])

    # define the ticker of the stock to analyze
    ticker = pickle.load(open('equities/ma_ticker.p', 'rb'))

    # plot the returns of a single equity
    figure = plot_equity_returns(ticker, (equity_returns - 1))

    output = io.BytesIO()
    FigureCanvas(figure).print_png(output)

    return Response(output.getvalue(), mimetype='image/png')


@app.route('/equity_price_analytics_plot.png')
def gui_plot_equity_price_analytics():
    # get equity price data
    high_prices = pd.read_csv('equities/High Prices.csv')
    low_prices = pd.read_csv('equities/Low Prices.csv')
    open_prices = pd.read_csv('equities/Open Prices.csv')
    close_prices = pd.read_csv('equities/Close Prices.csv')
    volumes = pd.read_csv('equities/Volumes.csv')

    high_prices = high_prices.drop(['date'], axis=1)
    low_prices = low_prices.drop(['date'], axis=1)
    open_prices = open_prices.drop(['date'], axis=1)
    close_prices = close_prices.drop(['date'], axis=1)
    volumes = volumes.drop(['date'], axis=1)

    # declare SMA and EMA variables
    interval = pickle.load(open('equities/ma_interval.p', 'rb'))
    ma_alpha = pickle.load(open('equities/ma_alpha.p', 'rb'))

    # define the ticker of the stock to look at
    ticker = pickle.load(open('equities/ma_ticker.p', 'rb'))

    # calculate price analytics
    e_ma, s_ma, t_wap, v_wap, close = equity_price_analytics(ticker, high_prices, low_prices, open_prices, close_prices,
                                                             volumes, ma_alpha, interval)

    figure = plot_equity_price_analytics(ticker, e_ma, s_ma, t_wap, v_wap, close, interval)

    output = io.BytesIO()
    FigureCanvas(figure).print_png(output)

    return Response(output.getvalue(), mimetype='image/png')


@app.route('/get_analytics', methods=['GET', 'POST'])
def get_analytics():

    # get equity price data
    high_prices = pd.read_csv('equities/High Prices.csv')
    low_prices = pd.read_csv('equities/Low Prices.csv')
    open_prices = pd.read_csv('equities/Open Prices.csv')
    close_prices = pd.read_csv('equities/Close Prices.csv')
    volumes = pd.read_csv('equities/Volumes.csv')

    high_prices = high_prices.drop(['date'], axis=1)
    low_prices = low_prices.drop(['date'], axis=1)
    open_prices = open_prices.drop(['date'], axis=1)
    close_prices = close_prices.drop(['date'], axis=1)
    volumes = volumes.drop(['date'], axis=1)

    # calculate returns for the portfolio and the benchmark
    w, _, eq_returns, port_returns, market_returns = get_returns()

    # calculate portfolio analytics
    alpha, beta, r_squared, _ = portfolio_analytics(port_returns, market_returns)

    # set the VaR percentage
    var_p = float(request.form['var_p'])
    pickle.dump(var_p, open('equities/var_p.p', 'wb'))

    # set the risk-free rate
    rf = float(request.form['rf'])
    pickle.dump(rf, open('equities/rf.p', 'wb'))

    # calculate the portfolio alpha based on the risk-free rate
    alpha_rf_rate = alpha_rf(port_returns, rf, market_returns, beta)

    # calculate the portfolio volatility
    portfolio_vol = portfolio_volatility(eq_returns, w)

    # calculate the portfolio Sharpe ratio
    sr = sharpe_ratio(port_returns, rf, eq_returns, w)

    # calculate the portfolio Treynor ratio
    tr = treynor_ratio(port_returns, rf, beta)

    # calculate the portfolio tracking error
    te = tracking_error(port_returns, market_returns)

    # calculate the Analytical VaR and the associated values
    sigma, mu, a_var, t_dist_a_var, es, t_dist_es = risk_metrics(eq_returns, w, portfolio_vol, var_p, 5)

    # define the confidence interval at which to take the Analytical VaR
    ci = float(request.form['ci'])
    pickle.dump(ci, open('equities/ci.p', 'wb'))

    _, _, var_level, t_dist_var_level, ci_es, ci_t_dist_es = ci_specified_risk_metrics(eq_returns, w, portfolio_vol, ci, 5)

    # calculate the current portfolio return
    current_portfolio_return = port_returns[-1]

    # calculate  the Historical VaR
    h_var = historical_var(port_returns, var_p)
    ci_h_var = ci_historical_var(port_returns, ci)

    # declare SMA and EMA variables
    interval = int(request.form['ma_interval'])
    pickle.dump(interval, open('equities/ma_interval.p', 'wb'))
    ma_alpha = float(request.form['ma_alpha'])
    pickle.dump(ma_alpha, open('equities/ma_alpha.p', 'wb'))

    # define the ticker of the stock to analyze
    ticker = str(request.form['ma_ticker'])
    pickle.dump(ticker, open('equities/ma_ticker.p', 'wb'))

    # calculate price analytics
    e_ma, s_ma, t_wap, v_wap, close = equity_price_analytics(ticker, high_prices, low_prices, open_prices, close_prices,
                                                             volumes, ma_alpha, interval)

    # get the most recent values of each price measure
    e_ma_recent = e_ma[-1]
    s_ma_recent = s_ma[-1]
    t_wap_recent = t_wap[-1]
    v_wap_recent = v_wap[-1]
    close_recent = close[-1]

    portfolio_analytics_values = 'Portfolio Alpha (calculated from the returns regression) = ' + str(alpha * 100) + \
                                 '\n' + 'Portfolio Alpha (calculated from the risk-free rate) = ' + \
                                 str(alpha_rf_rate * 100) + '\n' + 'Portfolio Beta = ' + str(beta) + '\n' + \
                                 'Portfolio Return = ' + str(current_portfolio_return * 100) + '%' + '\n' + \
                                 'Portfolio R-Squared = ' + str(r_squared) + '\n' + 'Portfolio Sharpe Ratio = ' + \
                                 str(sr) + '\n' + 'Portfolio Treynor Ratio = ' + str(tr) \
                                 + '\n' + 'Portfolio Tracking Error = ' + str(te) \
                                 + '\n' \
                                 + '\n' + 'Historical VaR = ' + str(h_var * 100) + '% at ' + str(var_p * 100) + \
                                 '% daily return' \
                                 + '\n' + 'Analytical VaR (Normal Distribution) = ' + str(a_var * 100) + '% at ' + \
                                 str(var_p * 100) + '% daily return' \
                                 + '\n' + 'Analytical VaR (t-Distribution) = ' \
                                 + str(t_dist_a_var * 100) + '% at ' + str(var_p * 100) + '% daily return' + '\n' \
                                 + 'Expected Shortfall (Normal Distribution) at ' + str(a_var * 100) + \
                                 '% level = ' + str(es * 100) + '%' \
                                 + '\n' + 'Expected Shortfall (t-Distribution) at ' \
                                 + str(a_var * 100) + '% level = ' + str(t_dist_es * 100) + '%' \
                                 + '\n' \
                                 + '\n' + 'Historical VaR at Confidence Interval: ' + str(ci * 100) + '% = ' + str(ci_h_var * 100) + \
                                 '% daily return' \
                                 + '\n' + 'Analytical VaR (Normal Distribution) at Confidence Interval: ' + str(ci * 100) + '% = ' + str(var_level * 100) + '% daily return' \
                                 + '\n' + 'Analytical VaR (t-Distribution) at Confidence Interval: ' + str( ci * 100) + '% = ' + str(t_dist_var_level * 100) + '% daily return' \
                                 + '\n' + 'Expected Shortfall (Normal Distribution) at Confidence Interval: ' + str(ci * 100) + \
                                 '% = ' + str(ci_es * 100) + '% daily return' \
                                 + '\n' + 'Expected Shortfall (t-Distribution) at Confidence Interval: ' + str(ci * 100) + \
                                 '% = ' + str(ci_t_dist_es * 100) + '% daily return' \
                                 + '\n' \
                                 + '\n' + 'Exponential Moving Average Price (most recent) = ' + str(e_ma_recent) + \
                                 ' USD' + '\n' + 'Simple Moving Average Price (most recent) = ' + str(s_ma_recent) + \
                                 ' USD' + '\n' + 'Time-Weighted Average Price (most recent) = ' + str(t_wap_recent) + \
                                 ' USD' + '\n' + 'Volume-Weighted Average Price (most recent) = ' + str(v_wap_recent) \
                                 + ' USD' + '\n' + 'Close Price (most recent) = ' + str(close_recent) + ' USD'

    pickle.dump(portfolio_analytics_values, open('equities/portfolio_analytics_values.p', 'wb'))

    return render_template('portfolio_analytics.html',
                           equity_list=pickle.load(open('equities/user_input_equities.p', 'rb')),
                           benchmark_ticker=pickle.load(open('equities/user_input_benchmark.p', 'rb')),
                           ma_interval_box=pickle.load(open('equities/ma_interval.p', 'rb')),
                           ma_alpha_box=pickle.load(open('equities/ma_alpha.p', 'rb')),
                           ma_ticker_box=pickle.load(open('equities/ma_ticker.p', 'rb')),
                           var_p_box=pickle.load(open('equities/var_p.p', 'rb')),
                           ci_box=pickle.load(open('equities/ci.p', 'rb')),
                           rf_box=pickle.load(open('equities/rf.p', 'rb')),
                           portfolio_analytics_values=pickle.load(open('equities/portfolio_analytics_values.p', 'rb')))


if __name__ == '__main__':
    app.run()
