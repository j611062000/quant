from tensorflow.keras.models import Sequential
from repository.model import load_model
from repository.ticker import Ticker
from pandas import DataFrame
import numpy as np
from typing import Dict
def backtest(model: Sequential, ticker: Ticker) -> Dict[str, float]:
    """
    Perform backtesting of the LSTM model on historical data.
    Returns metrics including MSE and directional accuracy.
    """
    # Get historical data
    historical_data = ticker.historical_data(start='2024-11-01', end='2024-11-26')
    
    # Prepare test data
    values = historical_data['Close'].values
    sequence_length = 10
    
    # Create sequences for prediction
    X, y_true = [], []
    for i in range(len(values) - sequence_length):
        X.append(values[i:(i + sequence_length)])
        y_true.append(values[i + sequence_length])
    
    X = np.array(X)
    y_true = np.array(y_true)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate metrics
    mse = np.mean((y_true - y_pred.flatten()) ** 2)
    
    # Calculate directional accuracy
    y_true_direction = np.diff(y_true) > 0
    y_pred_direction = np.diff(y_pred.flatten()) > 0
    directional_accuracy = np.mean(y_true_direction == y_pred_direction)
    
    return {
        'mse': float(mse),
        'directional_accuracy': float(directional_accuracy)
    }
    
if __name__ == '__main__':
    ticker = Ticker('2312.TW')
    model = load_model('models/2312.TW_model.keras')
    print(backtest(model, ticker))