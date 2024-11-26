import json
import pickle
from pandas import DataFrame
import yfinance as yf

class Ticker:
    def __init__(self, symbol: str):
        self.stock = yf.Ticker(symbol)
        self.symbol: str = symbol
        
    def __getstate__(self):
        # Only pickle the symbol, not the yfinance Ticker object
        return {'symbol': self.symbol}
    
    def __setstate__(self, state):
        # Recreate the yfinance Ticker object when unpickling
        self.symbol = state['symbol']
        self.stock = yf.Ticker(self.symbol)
    
    def pe_ratio(self):
        return self.stock.info['trailingPE']
    
    def historical_data(self, start: str, end: str) -> DataFrame:
        return self.stock.history(start=start, end=end)
    
def dump_ticker(ticker: Ticker):
    with open(f'tickers/{ticker.symbol}.pkl', 'wb') as f:
        pickle.dump(ticker, f)

def load_ticker(symbol: str) -> Ticker:
    with open(f'tickers/{symbol}.pkl', 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    ticker = Ticker('2312.TW')
    dump_ticker(ticker)
    loaded_ticker = load_ticker('2312.TW')
    print(loaded_ticker.symbol)