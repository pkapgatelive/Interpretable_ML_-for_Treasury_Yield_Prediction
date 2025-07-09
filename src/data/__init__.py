"""
Data processing package for yield curve forecasting.

This package contains modules for:
- Loading data from various sources (FRED, Bloomberg, etc.)
- Cleaning and preprocessing data
- Feature engineering for yield curve modeling
"""

from src.data.data_loader import (
    YieldCurveDataLoader,
    MacroDataLoader,
    load_treasury_data,
    load_fred_data,
)

from src.data.data_cleaner import (
    YieldCurvePreprocessor,
    handle_missing_values,
    detect_outliers,
    remove_outliers,
)

from src.data.feature_engineering import (
    YieldCurveFeatureEngineer,
    calculate_yield_curve_features,
    create_technical_indicators,
    create_lag_features,
)

__all__ = [
    # Data loading
    "YieldCurveDataLoader",
    "MacroDataLoader", 
    "load_treasury_data",
    "load_fred_data",
    
    # Data cleaning
    "YieldCurvePreprocessor",
    "handle_missing_values",
    "detect_outliers",
    "remove_outliers",
    
    # Feature engineering
    "YieldCurveFeatureEngineer",
    "calculate_yield_curve_features",
    "create_technical_indicators",
    "create_lag_features",
] 