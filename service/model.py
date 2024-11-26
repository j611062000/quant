import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from repository.model import dump_model
from repository.ticker import Ticker
from pandas import DataFrame
import os
from typing import Tuple, List, Any, Union
from numpy import ndarray
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.losses import Loss
from keras.saving import register_keras_serializable

def prepare_stock_data(df: DataFrame, train_split: float = 0.8) -> Tuple[ndarray, ndarray, MinMaxScaler]:
    """Prepare stock price data by splitting into train/test sets"""
    scaler = MinMaxScaler()
    values = scaler.fit_transform(df['Close'].values.reshape(-1, 1)).flatten()
    
    train_size = int(len(values) * train_split)
    return values[:train_size], values[train_size:], scaler

def create_sequence_dataset(data: ndarray, sequence_length: int = 10) -> Tuple[ndarray, ndarray]:
    """Create sequences of data for LSTM training"""
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append([data[i + sequence_length], 1.0])
    return np.array(X), np.array(y)

def build_lstm_model(sequence_length: int, units: int = 50) -> Sequential:
    """Build LSTM model architecture with prediction and confidence output"""
    model = Sequential([
        LSTM(units, input_shape=(sequence_length, 1), return_sequences=True),
        Dropout(0.2),
        LSTM(units//2),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(2, activation='sigmoid')  # Output: [prediction, confidence]
    ])
    
    # Use MSE loss directly
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    return model

def create_model(ticker: Ticker, start: str = '2024-01-01', end: str = '2024-10-26') -> Sequential:
    """Create and train LSTM model for stock price prediction"""
    # Get and prepare data
    historical_data = ticker.historical_data(start=start, end=end)
    train_data, test_data, scaler = prepare_stock_data(historical_data)
    
    # Create sequences for LSTM
    sequence_length = 20
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