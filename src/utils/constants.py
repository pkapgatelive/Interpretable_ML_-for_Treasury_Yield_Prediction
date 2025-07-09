"""
Constants for yield curve forecasting project.

This module contains all project-wide constants including:
- Yield curve tenors
- Data source URLs
- Model parameters
- File paths
- API endpoints
"""

from typing import List, Dict, Any
import os

# Project Information
PROJECT_NAME = "yield-curve-forecasting"
PROJECT_VERSION = "0.1.0"

# Yield Curve Tenors (in years)
YIELD_TENORS = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]
YIELD_TENOR_NAMES = ["3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]
YIELD_TENOR_MAPPING = dict(zip(YIELD_TENORS, YIELD_TENOR_NAMES))

# FRED (Federal Reserve Economic Data) Series IDs
FRED_SERIES_IDS = {
    "DGS3MO": "3-Month Treasury",
    "DGS6MO": "6-Month Treasury", 
    "DGS1": "1-Year Treasury",
    "DGS2": "2-Year Treasury",
    "DGS3": "3-Year Treasury",
    "DGS5": "5-Year Treasury",
    "DGS7": "7-Year Treasury",
    "DGS10": "10-Year Treasury",
    "DGS20": "20-Year Treasury",
    "DGS30": "30-Year Treasury",
    "FEDFUNDS": "Federal Funds Rate",
    "CPIAUCSL": "CPI All Items",
    "UNRATE": "Unemployment Rate",
    "GDPC1": "Real GDP",
    "VIXCLS": "VIX Volatility Index",
    "DEXUSEU": "USD/EUR Exchange Rate",
}

# Macro Economic Indicators
MACRO_INDICATORS = [
    "fed_funds_rate",
    "cpi_inflation", 
    "unemployment_rate",
    "gdp_growth",
    "vix",
    "dollar_index",
    "oil_price",
    "gold_price",
]

# Technical Indicators
TECHNICAL_INDICATORS = [
    "sma_5",    # 5-day simple moving average
    "sma_20",   # 20-day simple moving average
    "sma_60",   # 60-day simple moving average
    "ema_12",   # 12-day exponential moving average
    "ema_26",   # 26-day exponential moving average
    "rsi_14",   # 14-day relative strength index
    "macd",     # MACD indicator
    "bollinger_upper",  # Bollinger Bands upper
    "bollinger_lower",  # Bollinger Bands lower
]

# Yield Curve Features
YIELD_CURVE_FEATURES = [
    "level",      # Average yield across curve
    "slope",      # 10Y - 2Y spread
    "curvature",  # 2*(5Y) - (2Y) - (10Y)
    "twist",      # Long-end slope minus short-end slope
    "butterfly",  # (2Y + 10Y) / 2 - 5Y
]

# File Extensions
SUPPORTED_DATA_FORMATS = [".csv", ".xlsx", ".parquet", ".json", ".pkl"]
SUPPORTED_MODEL_FORMATS = [".pkl", ".joblib", ".h5", ".pb"]
SUPPORTED_IMAGE_FORMATS = [".png", ".jpg", ".jpeg", ".pdf", ".svg"]

# Data Paths (relative to project root)
DATA_PATHS = {
    "raw": "data/raw",
    "processed": "data/processed", 
    "features": "data/features",
    "external": "data/external",
}

# Model Paths
MODEL_PATHS = {
    "trained": "models/trained",
    "checkpoints": "models/checkpoints",
    "artifacts": "models/artifacts",
}

# Report Paths
REPORT_PATHS = {
    "figures": "reports/figures",
    "tables": "reports/tables", 
    "presentations": "reports/presentations",
}

# API URLs
API_URLS = {
    "fred": "https://api.stlouisfed.org/fred",
    "treasury": "https://home.treasury.gov/resource-center/data-chart-center",
    "bloomberg": "https://api.bloomberg.com",
}

# Model Types
MODEL_TYPES = {
    "baseline": ["linear_regression", "arima", "var", "nelson_siegel"],
    "tree_based": ["random_forest", "gradient_boosting", "xgboost", "lightgbm"],
    "neural_networks": ["mlp", "lstm", "gru", "transformer"],
    "ensemble": ["voting", "stacking", "bagging"],
}

# Evaluation Metrics
EVALUATION_METRICS = [
    "mae",          # Mean Absolute Error
    "mse",          # Mean Squared Error  
    "rmse",         # Root Mean Squared Error
    "mape",         # Mean Absolute Percentage Error
    "r2",           # R-squared
    "explained_variance",  # Explained Variance Score
    "max_error",    # Maximum Error
    "median_ae",    # Median Absolute Error
]

# Time Series Cross-Validation Parameters
TIME_SERIES_CV = {
    "n_splits": 5,
    "test_size_days": 252,  # 1 year
    "gap": 0,
    "max_train_size": None,
}

# Preprocessing Parameters
PREPROCESSING_PARAMS = {
    "outlier_method": "iqr",  # iqr, zscore, isolation_forest
    "outlier_threshold": 3.0,
    "missing_value_method": "interpolate",  # drop, fill, interpolate
    "scaling_method": "standard",  # standard, minmax, robust
}

# Feature Engineering Parameters
FEATURE_ENGINEERING_PARAMS = {
    "lag_periods": [1, 2, 3, 5, 10, 20],
    "rolling_windows": [5, 10, 20, 60],
    "differencing_orders": [1, 2],
    "log_transform": True,
}

# Visualization Parameters  
PLOT_PARAMS = {
    "figsize": (12, 8),
    "dpi": 300,
    "style": "seaborn-v0_8",
    "color_palette": "Set2",
    "font_size": 12,
}

# Colors for different plot elements
COLORS = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e", 
    "success": "#2ca02c",
    "danger": "#d62728",
    "warning": "#ff9500",
    "info": "#17a2b8",
    "light": "#f8f9fa",
    "dark": "#343a40",
}

# Random Seeds for Reproducibility
RANDOM_SEEDS = {
    "data_split": 42,
    "model_training": 123,
    "cross_validation": 456,
    "feature_selection": 789,
}

# Logging Configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "formatter": "standard", 
            "class": "logging.StreamHandler",
        },
    },
    "loggers": {
        "": {
            "handlers": ["default"],
            "level": "INFO",
            "propagate": False
        }
    }
}

# Environment Variables
REQUIRED_ENV_VARS = [
    "FRED_API_KEY",
    "BLOOMBERG_API_KEY", 
]

OPTIONAL_ENV_VARS = [
    "WANDB_API_KEY",
    "MLFLOW_TRACKING_URI",
] 