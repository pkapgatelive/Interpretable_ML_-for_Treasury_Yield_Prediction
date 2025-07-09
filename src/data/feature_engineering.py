"""
Feature engineering module for yield curve forecasting.

This module contains utilities for creating features from yield curve data,
including technical indicators, lag features, and yield curve characteristics.
"""

import pandas as pd
import numpy as np
from typing import Union, List, Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class YieldCurveFeatureEngineer:
    """
    Main class for creating features from yield curve data.
    
    Generates various features including:
    - Yield curve shape characteristics (slope, curvature, level)
    - Technical indicators
    - Lag features
    - Rolling statistics
    """
    
    def __init__(self, 
                 lag_periods: List[int] = [1, 5, 10, 22],
                 rolling_windows: List[int] = [5, 10, 22, 66]):
        """
        Initialize the feature engineer.
        
        Parameters
        ----------
        lag_periods : List[int]
            List of lag periods to create
        rolling_windows : List[int]
            List of rolling window sizes for statistics
        """
        self.lag_periods = lag_periods
        self.rolling_windows = rolling_windows
        
    def create_features(self, 
                       yield_data: pd.DataFrame,
                       macro_data: Optional[pd.DataFrame] = None,
                       config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Create all features from input data.
        
        Parameters
        ----------
        yield_data : pd.DataFrame
            Yield curve data with tenors as columns
        macro_data : pd.DataFrame, optional
            Macroeconomic data
        config : Dict[str, Any], optional
            Feature engineering configuration
            
        Returns
        -------
        pd.DataFrame
            DataFrame with all engineered features
        """
        logger.info("Starting feature engineering...")
        
        features = pd.DataFrame(index=yield_data.index)
        
        # Yield curve shape features
        yield_features = calculate_yield_curve_features(yield_data)
        features = pd.concat([features, yield_features], axis=1)
        
        # Technical indicators
        tech_indicators = create_technical_indicators(yield_data)
        features = pd.concat([features, tech_indicators], axis=1)
        
        # Lag features
        lag_features = create_lag_features(yield_data, self.lag_periods)
        features = pd.concat([features, lag_features], axis=1)
        
        # Rolling statistics
        rolling_features = self._create_rolling_features(yield_data)
        features = pd.concat([features, rolling_features], axis=1)
        
        # Macro features if provided
        if macro_data is not None:
            macro_features = self._process_macro_features(macro_data)
            # Align dates
            common_dates = features.index.intersection(macro_features.index)
            features = features.loc[common_dates]
            macro_features = macro_features.loc[common_dates]
            features = pd.concat([features, macro_features], axis=1)
        
        # Drop any rows with all NaN values
        features = features.dropna(how='all')
        
        logger.info(f"Feature engineering completed. Shape: {features.shape}")
        return features
    
    def _create_rolling_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create rolling statistical features."""
        rolling_features = pd.DataFrame(index=data.index)
        
        for window in self.rolling_windows:
            for col in data.columns:
                # Rolling mean
                rolling_features[f"{col}_ma_{window}"] = data[col].rolling(window).mean()
                # Rolling std
                rolling_features[f"{col}_std_{window}"] = data[col].rolling(window).std()
                # Rolling min/max
                rolling_features[f"{col}_min_{window}"] = data[col].rolling(window).min()
                rolling_features[f"{col}_max_{window}"] = data[col].rolling(window).max()
        
        return rolling_features
    
    def _process_macro_features(self, macro_data: pd.DataFrame) -> pd.DataFrame:
        """Process macroeconomic features."""
        macro_features = macro_data.copy()
        
        # Add changes and percentage changes
        for col in macro_data.columns:
            macro_features[f"{col}_change"] = macro_data[col].diff()
            macro_features[f"{col}_pct_change"] = macro_data[col].pct_change()
        
        return macro_features


def calculate_yield_curve_features(yield_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate yield curve shape characteristics.
    
    Parameters
    ----------
    yield_data : pd.DataFrame
        Yield curve data with tenors as columns
        
    Returns
    -------
    pd.DataFrame
        DataFrame with yield curve features
    """
    features = pd.DataFrame(index=yield_data.index)
    
    # Get tenor columns (assuming they're numeric or can be converted)
    tenor_cols = yield_data.columns.tolist()
    
    if len(tenor_cols) >= 3:
        # Level (average of all yields)
        features['level'] = yield_data.mean(axis=1)
        
        # Slope (long-term minus short-term)
        features['slope'] = yield_data.iloc[:, -1] - yield_data.iloc[:, 0]
        
        # Curvature (2*middle - short - long)
        if len(tenor_cols) >= 3:
            mid_idx = len(tenor_cols) // 2
            features['curvature'] = (2 * yield_data.iloc[:, mid_idx] - 
                                   yield_data.iloc[:, 0] - 
                                   yield_data.iloc[:, -1])
    
    # Spreads between different tenors
    for i in range(len(tenor_cols)):
        for j in range(i+1, len(tenor_cols)):
            col1, col2 = tenor_cols[i], tenor_cols[j]
            features[f'spread_{col1}_{col2}'] = yield_data[col2] - yield_data[col1]
    
    # Yield curve steepness (difference between consecutive tenors)
    for i in range(len(tenor_cols)-1):
        col1, col2 = tenor_cols[i], tenor_cols[i+1]
        features[f'steepness_{col1}_{col2}'] = yield_data[col2] - yield_data[col1]
    
    return features


def create_technical_indicators(yield_data: pd.DataFrame) -> pd.DataFrame:
    """
    Create technical indicators from yield data.
    
    Parameters
    ----------
    yield_data : pd.DataFrame
        Yield curve data
        
    Returns
    -------
    pd.DataFrame
        DataFrame with technical indicators
    """
    indicators = pd.DataFrame(index=yield_data.index)
    
    for col in yield_data.columns:
        series = yield_data[col]
        
        # Rate of change
        indicators[f'{col}_roc_1'] = series.pct_change(1)
        indicators[f'{col}_roc_5'] = series.pct_change(5)
        indicators[f'{col}_roc_22'] = series.pct_change(22)
        
        # Moving averages
        indicators[f'{col}_sma_5'] = series.rolling(5).mean()
        indicators[f'{col}_sma_22'] = series.rolling(22).mean()
        indicators[f'{col}_ema_5'] = series.ewm(span=5).mean()
        indicators[f'{col}_ema_22'] = series.ewm(span=22).mean()
        
        # Bollinger Bands
        sma_20 = series.rolling(20).mean()
        std_20 = series.rolling(20).std()
        indicators[f'{col}_bb_upper'] = sma_20 + (2 * std_20)
        indicators[f'{col}_bb_lower'] = sma_20 - (2 * std_20)
        indicators[f'{col}_bb_width'] = indicators[f'{col}_bb_upper'] - indicators[f'{col}_bb_lower']
        indicators[f'{col}_bb_position'] = (series - indicators[f'{col}_bb_lower']) / indicators[f'{col}_bb_width']
        
        # RSI
        indicators[f'{col}_rsi'] = calculate_rsi(series)
        
        # Momentum
        indicators[f'{col}_momentum_5'] = series - series.shift(5)
        indicators[f'{col}_momentum_22'] = series - series.shift(22)
    
    return indicators


def create_lag_features(data: pd.DataFrame, lag_periods: List[int]) -> pd.DataFrame:
    """
    Create lag features from the data.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data
    lag_periods : List[int]
        List of lag periods to create
        
    Returns
    -------
    pd.DataFrame
        DataFrame with lag features
    """
    lag_features = pd.DataFrame(index=data.index)
    
    for lag in lag_periods:
        for col in data.columns:
            lag_features[f'{col}_lag_{lag}'] = data[col].shift(lag)
    
    return lag_features


def calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    
    Parameters
    ----------
    series : pd.Series
        Price series
    window : int
        RSI period
        
    Returns
    -------
    pd.Series
        RSI values
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def create_volatility_features(data: pd.DataFrame, windows: List[int] = [5, 10, 22]) -> pd.DataFrame:
    """
    Create volatility-based features.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data
    windows : List[int]
        Rolling windows for volatility calculation
        
    Returns
    -------
    pd.DataFrame
        DataFrame with volatility features
    """
    vol_features = pd.DataFrame(index=data.index)
    
    for window in windows:
        for col in data.columns:
            # Rolling volatility (standard deviation)
            vol_features[f'{col}_vol_{window}'] = data[col].rolling(window).std()
            
            # Range-based volatility (high-low)
            vol_features[f'{col}_range_{window}'] = (
                data[col].rolling(window).max() - data[col].rolling(window).min()
            )
    
    return vol_features


def create_interaction_features(data: pd.DataFrame, max_interactions: int = 10) -> pd.DataFrame:
    """
    Create interaction features between different yields.
    
    Parameters
    ----------
    data : pd.DataFrame
        Yield data
    max_interactions : int
        Maximum number of interaction features to create
        
    Returns
    -------
    pd.DataFrame
        DataFrame with interaction features
    """
    interaction_features = pd.DataFrame(index=data.index)
    
    columns = data.columns.tolist()
    interaction_count = 0
    
    for i in range(len(columns)):
        for j in range(i+1, len(columns)):
            if interaction_count >= max_interactions:
                break
                
            col1, col2 = columns[i], columns[j]
            
            # Ratio
            interaction_features[f'{col1}_{col2}_ratio'] = data[col1] / (data[col2] + 1e-8)
            
            # Product
            interaction_features[f'{col1}_{col2}_product'] = data[col1] * data[col2]
            
            interaction_count += 2
            
        if interaction_count >= max_interactions:
            break
    
    return interaction_features 