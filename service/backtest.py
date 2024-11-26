from abc import ABC, abstractmethod
from typing import Dict, Any, Protocol, List
import numpy as np
from repository.model import load_model
from repository.ticker import Ticker
from pandas import DataFrame

class Strategy(Protocol):
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Make predictions on the input data"""
        ...

class Metric(ABC):
    @abstractmethod
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, confidence: np.ndarray) -> float:
        """Calculate metric value"""
        pass

class MSE(Metric):
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, confidence: np.ndarray) -> float:
        return float(np.mean((y_true - y_pred) ** 2))

class DirectionalAccuracy(Metric):
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, confidence: np.ndarray) -> float:
        y_true_dir = np.diff(y_true) > 0
        y_pred_dir = np.diff(y_pred) > 0
        return float(np.mean(y_true_dir == y_pred_dir))

class WinRate(Metric):
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, confidence: np.ndarray) -> float:
        price_changes = np.diff(y_true)
        pred_changes = np.diff(y_pred)
        correct = (price_changes > 0) == (pred_changes > 0)
        profitable = correct & (np.abs(price_changes) > 0)
        return float(np.mean(profitable))

class TotalReturn(Metric):
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, confidence: np.ndarray) -> float:
        price_changes = np.diff(y_true)
        pred_changes = np.diff(y_pred)
        
        # Only take positions when confidence > 90%
        high_conf = confidence[:-1] > 0.7
        
        # Calculate returns based on trading strategy
        position = np.zeros_like(price_changes)
        position[high_conf & (pred_changes > 0)] = 1  # Buy signals
        position[high_conf & (pred_changes < 0)] = -1  # Sell signals
        
        # Calculate returns for each trade
        trade_returns = position * price_changes / y_true[:-1]
        
        # Calculate total return
        total_return = np.sum(trade_returns) * 100
        return float(total_return)

class Backtester:
    def __init__(self, strategy: Strategy, metrics: List[Metric], sequence_length: int = 10):
        self.strategy = strategy
        self.metrics = metrics
        self.sequence_length = sequence_length

    def prepare_data(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length])
        return np.array(X), np.array(y)

    def backtest(self, ticker: Ticker, start: str, end: str) -> Dict[str, float]:
        # Get historical data
        historical_data = ticker.historical_data(start=start, end=end)
        values = historical_data['Close'].values

        # Prepare sequences
        X, y_true = self.prepare_data(values)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        # Get predictions and confidence scores
        predictions = self.strategy.predict(X)
        y_pred = predictions[:, 0]  # Direction prediction
        confidence = predictions[:, 1]  # Confidence scores
        
        if len(y_pred.shape) > 1:
            y_pred = y_pred.flatten()
            confidence = confidence.flatten()

        # Calculate all metrics
        results = {}
        for metric in self.metrics:
            metric_name = metric.__class__.__name__.lower()
            results[metric_name] = metric.calculate(y_true, y_pred, confidence)

        return results

if __name__ == '__main__':
    # Example usage with LSTM model
    ticker = Ticker('2312.TW')
    model = load_model('models/2312.TW_model.keras')
    
    metrics = [MSE(), DirectionalAccuracy(), WinRate(), TotalReturn()]
    backtester = Backtester(strategy=model, metrics=metrics)
    
    results = backtester.backtest(
        ticker=ticker,
        start='2024-11-01',
        end='2024-11-26'
    )
    print(results)