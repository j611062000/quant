import numpy as np
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import LSTM, Dense
from repository.model import dump_model
from repository.ticker import Ticker
from pandas import DataFrame
import os
from typing import Tuple, List, Any, Union
from numpy import ndarray

def prepare_stock_data(df: DataFrame, train_split: float = 0.8) -> Tuple[ndarray, ndarray]:
    """Prepare stock price data by splitting into train/test sets"""
    values = df['Close'].values
    train_size = int(len(values) * train_split)
    return values[:train_size], values[train_size:]

def create_sequence_dataset(data: ndarray, sequence_length: int = 10) -> Tuple[ndarray, ndarray]:
    """Create sequences of data for LSTM training"""
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

def build_lstm_model(sequence_length: int, units: int = 50) -> Sequential:
    """Build LSTM model architecture"""
    model = Sequential([
        LSTM(units, input_shape=(sequence_length, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def create_model(ticker: Ticker, start: str = '2024-01-01', end: str = '2024-10-26') -> Sequential:
    """Create and train LSTM model for stock price prediction"""
    # Get and prepare data
    historical_data = ticker.historical_data(start=start, end=end)
    train_data, test_data = prepare_stock_data(historical_data)
    
    # Create sequences for LSTM
    sequence_length = 10
    X_train, y_train = create_sequence_dataset(train_data, sequence_length)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    
    # Build and train model
    model = build_lstm_model(sequence_length)
    model.fit(X_train, y_train, epochs=20, batch_size=32)
    
    return model

if __name__ == '__main__':
    # Create and save model
    ticker = Ticker('2312.TW')
    model = create_model(ticker)
    
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    
    # Save trained model
    model_path = f'models/{ticker.symbol}_model.keras'
    dump_model(model, model_path)