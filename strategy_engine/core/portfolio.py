"""
Portfolio management module.
Handles position sizing, rebalancing, and weight calculations.
"""

import pandas as pd
import numpy as np


class Portfolio:
    """Manage portfolio positions and weights."""
    
    def __init__(self, initial_capital, assets, max_leverage=1.0):
        """
        Initialize portfolio.
        
        Args:
            initial_capital: Initial capital in dollars
            assets: List of asset tickers
            max_leverage: Maximum leverage ratio
        """
        self.initial_capital = initial_capital
        self.assets = assets
        self.max_leverage = max_leverage
        self.current_values = {asset: 0 for asset in assets}
        self.weights = {asset: 1.0 / len(assets) for asset in assets}
    
    def calculate_weights(self, signals, method='signal_based'):
        """
        Calculate portfolio weights based on signals.
        
        Args:
            signals: DataFrame or dict of signals
            method: 'equal', 'signal_based', 'momentum_based'
        
        Returns:
            Dictionary of weights
        """
        if method == 'equal':
            return {asset: 1.0 / len(self.assets) for asset in self.assets}
        
        elif method == 'signal_based':
            # Normalize signals to weights
            if isinstance(signals, pd.DataFrame):
                latest_signals = signals.iloc[-1]
            else:
                latest_signals = signals
            
            raw_weights = {}
            for asset in self.assets:
                if asset in latest_signals.index:
                    raw_weights[asset] = latest_signals[asset]
                else:
                    raw_weights[asset] = 0
            
            # Normalize to sum to 1
            total = sum(abs(v) for v in raw_weights.values())
            if total > 0:
                weights = {k: v / total for k, v in raw_weights.items()}
            else:
                weights = {asset: 1.0 / len(self.assets) for asset in self.assets}
            
            return weights
        
        else:
            return {asset: 1.0 / len(self.assets) for asset in self.assets}
    
    def apply_leverage(self, weights):
        """Apply leverage constraints to weights."""
        total_exposure = sum(abs(w) for w in weights.values())
        
        if total_exposure > self.max_leverage:
            scale = self.max_leverage / total_exposure
            weights = {k: v * scale for k, v in weights.items()}
        
        return weights


if __name__ == "__main__":
    portfolio = Portfolio(100000, ['SPY', 'QQQ', 'IWM'])
