"""
Signal generation module.
Includes trend, volatility, and regime detection signals.
"""

import pandas as pd
import numpy as np


class SignalGenerator:
    """Generate trading signals from price and feature data."""
    
    @staticmethod
    def trend_signal(prices, fast_ma=20, slow_ma=63):
        """
        Simple moving average crossover signal.
        
        Returns:
            1 if fast_ma > slow_ma, -1 otherwise
        """
        fast = prices.rolling(fast_ma).mean()
        slow = prices.rolling(slow_ma).mean()
        signal = pd.Series(
            np.where(fast > slow, 1, -1),
            index=prices.index
        )
        return signal
    
    @staticmethod
    def volatility_signal(returns, vol_window=20, vol_threshold=0.15):
        """
        Volatility regime signal.
        
        Returns:
            1 if vol < threshold (low vol), -1 if vol > threshold (high vol)
        """
        vol = returns.rolling(vol_window).std()
        signal = pd.Series(
            np.where(vol < vol_threshold, 1, -1),
            index=returns.index
        )
        return signal
    
    @staticmethod
    def momentum_signal(prices, window=20):
        """
        Momentum signal based on price momentum.
        
        Returns:
            1 if positive momentum, -1 if negative momentum
        """
        momentum = prices.pct_change(window)
        signal = pd.Series(
            np.where(momentum > 0, 1, -1),
            index=prices.index
        )
        return signal
    
    @staticmethod
    def combine_signals(signal_dict, weights=None):
        """
        Combine multiple signals with optional weights.
        
        Args:
            signal_dict: Dictionary of signals
            weights: Dictionary of weights for each signal
        
        Returns:
            Combined signal
        """
        if weights is None:
            weights = {k: 1.0 / len(signal_dict) for k in signal_dict.keys()}
        
        combined = None
        for name, signal in signal_dict.items():
            w = weights.get(name, 0)
            if combined is None:
                combined = signal * w
            else:
                combined += signal * w
        
        return combined


if __name__ == "__main__":
    # Example usage
    gen = SignalGenerator()
