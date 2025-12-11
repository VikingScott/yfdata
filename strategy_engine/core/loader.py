"""
Data loader for processed data and features.
Interfaces with data_pipeline outputs.
"""

import pandas as pd
from pathlib import Path


class DataLoader:
    """Load processed prices and computed features."""
    
    def __init__(self, base_dir="data_pipeline"):
        self.base_dir = Path(base_dir)
        self.processed_dir = self.base_dir / "processed"
        self.features_dir = self.base_dir / "features"
    
    def load_price_panel(self):
        """Load aligned price panel."""
        price_path = self.processed_dir / "price_panel.csv"
        return pd.read_csv(price_path, index_col=0, parse_dates=True)
    
    def load_features(self):
        """Load computed feature matrix."""
        feature_path = self.features_dir / "feature_matrix.csv"
        return pd.read_csv(feature_path, index_col=0, parse_dates=True)
    
    def load_data_for_backtest(self, start_date=None, end_date=None):
        """
        Load price and feature data for backtest period.
        
        Args:
            start_date: str, 'YYYY-MM-DD' format
            end_date: str, 'YYYY-MM-DD' format
        
        Returns:
            Tuple of (price_panel, features)
        """
        prices = self.load_price_panel()
        features = self.load_features()
        
        if start_date and end_date:
            prices = prices[start_date:end_date]
            features = features[start_date:end_date]
        
        return prices, features


if __name__ == "__main__":
    loader = DataLoader()
    prices = loader.load_price_panel()
    print(prices.head())
