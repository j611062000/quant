import numpy as np
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import LSTM, Dense
from repository.model import dump_model
from repository.ticker import Ticker
from pandas import DataFrame
import os

def create_model(ticker: Ticker) -> Sequential:
    # Get historical data
    df: DataFrame = ticker.historical_data(start='2024-01-01', end='2024-11-26')
    values = df['Close'].values
    
    # Prepare training data
    train_size = int(len(values) * 0.8)
    train, test = values[:train_size], values[train_size:]

    # Reshape for LSTM (samples, timesteps, features)
    def create_dataset(data, look_back=1):
        X, y = [], []
        for i in range(len(data) - look_back):
            X.append(data[i:i + look_back])
            y.append(data[i + look_back])
        return np.array(X), np.array(y)

    look_back = 10
    X_train, y_train = create_dataset(train, look_back)
    
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

    # Build and train LSTM model
    model = Sequential([
        LSTM(50, input_shape=(look_back, 1)),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=20, batch_size=32)

    return model

if __name__ == '__main__':
    ticker = Ticker('AAPL')
    model = create_model(ticker)
    
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Save the model
    model_path = f'models/{ticker.symbol}_model.keras'
    dump_model(model, model_path)