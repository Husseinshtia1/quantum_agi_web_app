import yfinance as yf
import pandas as pd
import numpy as np

class StockDataCollector:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date

    def fetch_data(self):
        # Fetch historical stock data
        stock_data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        return stock_data

    def preprocess_data(self, stock_data):
        # Preprocess the data (e.g., normalize, handle missing values)
        stock_data = stock_data.dropna()
        stock_data['Normalized Close'] = stock_data['Close'] / stock_data['Close'].max()
        return stock_data

    def get_features_and_labels(self, stock_data, window_size=5):
        # Create features and labels for training
        features = []
        labels = []
        for i in range(len(stock_data) - window_size):
            features.append(stock_data['Normalized Close'].iloc[i:i+window_size].values)
            labels.append(stock_data['Normalized Close'].iloc[i+window_size])
        return np.array(features), np.array(labels)

# Example usage
if __name__ == "__main__":
    collector = StockDataCollector(ticker='AAPL', start_date='2020-01-01', end_date='2021-01-01')
    stock_data = collector.fetch_data()
    processed_data = collector.preprocess_data(stock_data)
    X_train, y_train = collector.get_features_and_labels(processed_data)

    print(f"Features shape: {X_train.shape}")
    print(f"Labels shape: {y_train.shape}")
