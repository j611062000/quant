from finlab import backtest
from finlab import data

close = data.get('price:收盤價')

# 創三百個交易日新高
position = close >= close.rolling(300).max()

report = backtest.sim(position, resample='M')
report.display()