from turtle import st
from pandas import DataFrame
import yfinance as yf

class Ticker:
    def __init__(self, symbol: str):
        self.stock = yf.Ticker(symbol)
        self.symbol: str = symbol
    
    def pe_ratio(self):
        return self.stock.info['trailingPE']
    
    def historical_data(self, start: str, end: str) -> DataFrame: 
        return self.stock.history(start=start, end=end)