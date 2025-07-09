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

# Model and training constants
MODEL_RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

# Data processing constants
MISSING_VALUE_THRESHOLD = 0.1  # 10% missing values threshold
MIN_OBSERVATIONS = 100  # Minimum observations required

# FRED API Series IDs for data acquisition
FRED_SERIES_IDS = {
    # Treasury Yields
    "DGS1MO": "1-Month Treasury Constant Maturity Rate",
    "DGS3MO": "3-Month Treasury Constant Maturity Rate", 
    "DGS6MO": "6-Month Treasury Constant Maturity Rate",
    "DGS1": "1-Year Treasury Constant Maturity Rate",
    "DGS2": "2-Year Treasury Constant Maturity Rate",
    "DGS3": "3-Year Treasury Constant Maturity Rate",
    "DGS5": "5-Year Treasury Constant Maturity Rate",
    "DGS7": "7-Year Treasury Constant Maturity Rate",
    "DGS10": "10-Year Treasury Constant Maturity Rate",
    "DGS20": "20-Year Treasury Constant Maturity Rate",
    "DGS30": "30-Year Treasury Constant Maturity Rate",
    
    # Monetary Policy
    "FEDFUNDS": "Federal Funds Rate",
    "DFF": "Daily Federal Funds Rate",
    "TB3MS": "3-Month Treasury Bill Rate",
    
    # Inflation
    "CPIAUCSL": "Consumer Price Index for All Urban Consumers: All Items",
    "CPILFESL": "Consumer Price Index for All Urban Consumers: All Items Less Food and Energy",
    "PCEPILFE": "Personal Consumption Expenditures Excluding Food and Energy (Chain-Type Price Index)",
    "T5YIE": "5-Year Breakeven Inflation Rate",
    "T10YIE": "10-Year Breakeven Inflation Rate",
    
    # Economic Activity
    "INDPRO": "Industrial Production Index",
    "PAYEMS": "All Employees, Nonfarm Payrolls",
    "UNRATE": "Unemployment Rate",
    "NAPM": "ISM Manufacturing: PMI Composite Index",
    "UMCSENT": "University of Michigan: Consumer Sentiment",
    
    # Financial Markets
    "DEXUSEU": "U.S. / Euro Foreign Exchange Rate",
    "DEXJPUS": "Japanese Yen to U.S. Dollar Spot Exchange Rate",
    "VIXCLS": "CBOE Volatility Index: VIX",
    "SP500": "S&P 500",
    
    # Credit
    "BAMLC0A0CM": "ICE BofA US Corporate Index Option-Adjusted Spread",
    "TEDRATE": "TED Spread",
    
    # Housing
    "HOUST": "New Privately-Owned Housing Units Started: Total Units",
    "CSUSHPISA": "S&P/Case-Shiller U.S. National Home Price Index"
}

# Data validation bounds
YIELD_BOUNDS = {
    "min": -5.0,  # Minimum reasonable yield (negative rates possible)
    "max": 50.0   # Maximum reasonable yield
}

MACRO_BOUNDS = {
    "rates": {"min": -5.0, "max": 50.0},
    "indices": {"min": 0.0, "max": 1000.0},
    "employment": {"min": 0.0, "max": 50.0}  # unemployment rate %
}

# File naming conventions
DATA_FILENAME_PATTERNS = {
    "treasury": "yieldcurve_us_{date}.{ext}",
    "macro": "macro_fred_{date}.{ext}",
    "ecb": "yieldcurve_ecb_{date}.{ext}"
}

# Logging configuration
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

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