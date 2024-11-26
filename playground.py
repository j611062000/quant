from repository.ticker import Ticker


ticker: Ticker = Ticker('2312.TW')
print(ticker.historical_data(start='2024-01-01', end='2024-11-26'))