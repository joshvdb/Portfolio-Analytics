import pandas as pd
from alpha_vantage.timeseries import TimeSeries

equities = ['^GSPC']

key = 'yourkeyhere'

close_prices = pd.DataFrame({'A' : []})

for equity in equities:
    ts = TimeSeries(key, output_format='pandas')
    ticker, meta = ts.get_daily(symbol=equity)
    close_prices = pd.concat([close_prices, ticker['4. close']], axis=1)

close_prices = close_prices.dropna(axis=1)

close_prices.to_csv('Benchmark Prices.csv')
