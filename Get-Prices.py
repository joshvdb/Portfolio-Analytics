import pandas as pd
from alpha_vantage.timeseries import TimeSeries

equities = ['AAPL', 'GOOGL', 'BLK', 'IBM']

key = 'yourkeyhere'

high_prices = pd.DataFrame({'A' : []})
low_prices = pd.DataFrame({'A' : []})
open_prices = pd.DataFrame({'A' : []})
close_prices = pd.DataFrame({'A' : []})
volumes = pd.DataFrame({'A' : []})

for equity in equities:
    ts = TimeSeries(key, output_format='pandas')
    ticker, meta = ts.get_daily(symbol=equity)

    high_prices = pd.concat([high_prices, ticker['1. open']], axis=1)
    low_prices = pd.concat([low_prices, ticker['2. high']], axis=1)
    open_prices = pd.concat([open_prices, ticker['3. low']], axis=1)
    close_prices = pd.concat([close_prices, ticker['4. close']], axis=1)
    volumes = pd.concat([volumes, ticker['5. volume']], axis=1)

high_prices = high_prices.dropna(axis=1)
low_prices = low_prices.dropna(axis=1)
open_prices = open_prices.dropna(axis=1)
close_prices = close_prices.dropna(axis=1)
volumes = volumes.dropna(axis=1)

high_prices.to_csv('High Prices.csv')
low_prices.to_csv('Low Prices.csv')
open_prices.to_csv('Open Prices.csv')
close_prices.to_csv('Close Prices.csv')
volumes.to_csv('Volumes.csv')
