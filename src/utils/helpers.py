"""
Helper functions for yield curve forecasting project.

This module contains utility functions that are used across the project
for common operations like logging, model persistence, data validation,
and project management.

Functions
---------
setup_logging : Configure logging for the project
save_model : Save trained models to disk
load_model : Load trained models from disk
create_directory : Create directories if they don't exist
get_project_root : Get the project root directory
validate_data : Validate data quality and structure
split_time_series : Split time series data for training/validation/testing
"""

import os
import pickle
import joblib
import logging
import logging.config
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yaml
import json

from src.utils.constants import (
    LOGGING_CONFIG,
    SUPPORTED_MODEL_FORMATS,
    SUPPORTED_DATA_FORMATS,
    RANDOM_SEEDS,
    EVALUATION_METRICS,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
)

# Setup module logger
logger = logging.getLogger(__name__)


def setup_logging(
    config_path: Optional[str] = None,
    log_level: str = "INFO",
    log_file: Optional[str] = None
) -> None:
    """
    Setup logging configuration for the project.
    
    Parameters
    ----------
    config_path : Optional[str]
        Path to logging configuration file
    log_level : str
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    log_file : Optional[str]
        Path to log file (if None, logs to console only)
    """
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.config.dictConfig(config)
    else:
        # Use default configuration
        config = LOGGING_CONFIG.copy()
        config["handlers"]["default"]["level"] = log_level
        
        if log_file:
            # Add file handler
            config["handlers"]["file"] = {
                "level": log_level,
                "formatter": "standard",
                "class": "logging.FileHandler",
                "filename": log_file,
                "mode": "a",
            }
            config["loggers"][""]["handlers"].append("file")
        
        logging.config.dictConfig(config)
    
    logger.info("Logging configuration setup complete")


def save_model(
    model: Any,
    filepath: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None,
    compression: bool = True
) -> None:
    """
    Save a trained model to disk with optional metadata.
    
    Parameters
    ----------
    model : Any
        Trained model object to save
    filepath : Union[str, Path]
        Path where to save the model
    metadata : Optional[Dict[str, Any]]
        Additional metadata to save with the model
    compression : bool
        Whether to use compression when saving
        
    Raises
    ------
    ValueError
        If file extension is not supported
    """
    filepath = Path(filepath)
    extension = filepath.suffix.lower()
    
    # Create directory if it doesn't exist
    create_directory(filepath.parent)
    
    if extension == ".pkl":
        protocol = pickle.HIGHEST_PROTOCOL if compression else pickle.DEFAULT_PROTOCOL
        with open(filepath, 'wb') as f:
            pickle.dump(model, f, protocol=protocol)
    elif extension == ".joblib":
        compress = 3 if compression else 0
        joblib.dump(model, filepath, compress=compress)
    elif extension in [".h5", ".hdf5"]:
        # For deep learning models (TensorFlow/Keras)
        if hasattr(model, 'save'):
            model.save(filepath)
        else:
            raise ValueError(f"Model does not support .h5 format")
    else:
        raise ValueError(f"Unsupported model format: {extension}")
    
    # Save metadata if provided
    if metadata:
        metadata_path = filepath.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    logger.info(f"Model saved to {filepath}")


def load_model(
    filepath: Union[str, Path],
    load_metadata: bool = True
) -> Union[Any, Tuple[Any, Dict[str, Any]]]:
    """
    Load a trained model from disk.
    
    Parameters
    ----------
    filepath : Union[str, Path]
        Path to the saved model
    load_metadata : bool
        Whether to also load metadata
        
    Returns
    -------
    Union[Any, Tuple[Any, Dict[str, Any]]]
        Loaded model, optionally with metadata
        
    Raises
    ------
    FileNotFoundError
        If model file doesn't exist
    ValueError
        If file extension is not supported
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    extension = filepath.suffix.lower()
    
    if extension == ".pkl":
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
    elif extension == ".joblib":
        model = joblib.load(filepath)
    elif extension in [".h5", ".hdf5"]:
        # For deep learning models
        try:
            from tensorflow.keras.models import load_model as tf_load_model
            model = tf_load_model(filepath)
        except ImportError:
            raise ImportError("TensorFlow required to load .h5 models")
    else:
        raise ValueError(f"Unsupported model format: {extension}")
    
    logger.info(f"Model loaded from {filepath}")
    
    if load_metadata:
        metadata_path = filepath.with_suffix('.json')
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        return model, metadata
    
    return model


def create_directory(path: Union[str, Path]) -> Path:
    """
    Create directory if it doesn't exist.
    
    Parameters
    ----------
    path : Union[str, Path]
        Directory path to create
        
    Returns
    -------
    Path
        Path object of the created directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns
    -------
    Path
        Path to the project root directory
    """
    current_file = Path(__file__)
    # Navigate up from src/utils/helpers.py to project root
    project_root = current_file.parent.parent.parent
    return project_root.resolve()


def validate_data(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    check_missing: bool = True,
    check_duplicates: bool = True,
    check_dtypes: bool = True
) -> Dict[str, Any]:
    """
    Validate data quality and structure.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate
    required_columns : Optional[List[str]]
        List of required column names
    check_missing : bool
        Whether to check for missing values
    check_duplicates : bool
        Whether to check for duplicate rows
    check_dtypes : bool
        Whether to check data types
        
    Returns
    -------
    Dict[str, Any]
        Validation results and statistics
    """
    validation_results = {
        "shape": df.shape,
        "is_valid": True,
        "issues": [],
        "statistics": {}
    }
    
    # Check required columns
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            validation_results["is_valid"] = False
            validation_results["issues"].append(f"Missing columns: {missing_cols}")
    
    # Check for missing values
    if check_missing:
        missing_counts = df.isnull().sum()
        missing_pct = (missing_counts / len(df)) * 100
        validation_results["statistics"]["missing_values"] = {
            "counts": missing_counts.to_dict(),
            "percentages": missing_pct.to_dict()
        }
        
        if missing_counts.sum() > 0:
            validation_results["issues"].append(f"Missing values found in {missing_counts.sum()} cells")
    
    # Check for duplicates
    if check_duplicates:
        duplicate_count = df.duplicated().sum()
        validation_results["statistics"]["duplicates"] = duplicate_count
        
        if duplicate_count > 0:
            validation_results["issues"].append(f"Found {duplicate_count} duplicate rows")
    
    # Check data types
    if check_dtypes:
        validation_results["statistics"]["dtypes"] = df.dtypes.to_dict()
    
    # Additional statistics
    validation_results["statistics"]["memory_usage"] = df.memory_usage(deep=True).sum()
    validation_results["statistics"]["date_range"] = {
        "start": str(df.index.min()) if hasattr(df.index, 'min') else None,
        "end": str(df.index.max()) if hasattr(df.index, 'max') else None
    }
    
    logger.info(f"Data validation completed. Valid: {validation_results['is_valid']}")
    
    return validation_results


def split_time_series(
    df: pd.DataFrame,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    shuffle: bool = False,
    random_state: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split time series data for training, validation, and testing.
    
    Parameters
    ----------
    df : pd.DataFrame
        Time series data to split
    train_ratio : float
        Proportion of data for training
    val_ratio : float
        Proportion of data for validation
    test_ratio : float
        Proportion of data for testing
    shuffle : bool
        Whether to shuffle the data (not recommended for time series)
    random_state : Optional[int]
        Random state for reproducibility
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Training, validation, and test datasets
        
    Raises
    ------
    ValueError
        If ratios don't sum to 1.0
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Train, validation, and test ratios must sum to 1.0")
    
    if shuffle and random_state:
        np.random.seed(random_state)
        df = df.sample(frac=1.0).reset_index(drop=True)
    
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    return train_df, val_df, test_df


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Calculate evaluation metrics for model predictions.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    metrics : Optional[List[str]]
        List of metrics to calculate
        
    Returns
    -------
    Dict[str, float]
        Dictionary of calculated metrics
    """
    if metrics is None:
        metrics = EVALUATION_METRICS
    
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        mean_absolute_percentage_error,
        r2_score,
        explained_variance_score,
        max_error,
        median_absolute_error
    )
    
    results = {}
    
    for metric in metrics:
        try:
            if metric == "mae":
                results[metric] = mean_absolute_error(y_true, y_pred)
            elif metric == "mse":
                results[metric] = mean_squared_error(y_true, y_pred)
            elif metric == "rmse":
                results[metric] = np.sqrt(mean_squared_error(y_true, y_pred))
            elif metric == "mape":
                results[metric] = mean_absolute_percentage_error(y_true, y_pred)
            elif metric == "r2":
                results[metric] = r2_score(y_true, y_pred)
            elif metric == "explained_variance":
                results[metric] = explained_variance_score(y_true, y_pred)
            elif metric == "max_error":
                results[metric] = max_error(y_true, y_pred)
            elif metric == "median_ae":
                results[metric] = median_absolute_error(y_true, y_pred)
        except Exception as e:
            logger.warning(f"Could not calculate {metric}: {e}")
            results[metric] = np.nan
    
    return results


def format_currency(amount: float, currency: str = "USD") -> str:
    """
    Format currency values for display.
    
    Parameters
    ----------
    amount : float
        Amount to format
    currency : str
        Currency code
        
    Returns
    -------
    str
        Formatted currency string
    """
    if currency == "USD":
        return f"${amount:,.2f}"
    else:
        return f"{amount:,.2f} {currency}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format percentage values for display.
    
    Parameters
    ----------
    value : float
        Value to format (as decimal, e.g., 0.05 for 5%)
    decimals : int
        Number of decimal places
        
    Returns
    -------
    str
        Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: str = "logs"
) -> logging.Logger:
    """
    Setup logging configuration for the application.
    
    Parameters
    ----------
    log_level : str
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    log_file : Optional[str]
        Log file name. If None, logs only to console.
    log_dir : str
        Directory to store log files
        
    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    if log_file:
        create_directory(log_dir)
        log_path = Path(log_dir) / log_file
    else:
        log_path = None
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        filename=log_path,
        filemode='a' if log_path else None
    )
    
    # Also log to console if logging to file
    if log_path:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        formatter = logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT)
        console_handler.setFormatter(formatter)
        
        logger = logging.getLogger()
        logger.addHandler(console_handler)
    
    return logging.getLogger(__name__)


def create_directory(path: Union[str, Path]) -> Path:
    """
    Create directory if it doesn't exist.
    
    Parameters
    ----------
    path : Union[str, Path]
        Path to directory to create
        
    Returns
    -------
    Path
        Path object of created directory
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def validate_date_format(date_string: str) -> bool:
    """
    Validate if date string is in YYYY-MM-DD format.
    
    Parameters
    ----------
    date_string : str
        Date string to validate
        
    Returns
    -------
    bool
        True if valid date format, False otherwise
    """
    try:
        datetime.strptime(date_string, '%Y-%m-%d')
        return True
    except ValueError:
        return False


def clean_numeric_data(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    fill_method: str = "forward"
) -> pd.DataFrame:
    """
    Clean numeric data by handling missing values and outliers.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to clean
    columns : Optional[List[str]]
        Specific columns to clean. If None, cleans all numeric columns.
    fill_method : str
        Method to fill missing values ('forward', 'backward', 'interpolate', 'drop')
        
    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame
    """
    df_clean = df.copy()
    
    if columns is None:
        columns = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in columns:
        if col not in df_clean.columns:
            continue
            
        # Handle missing values
        if fill_method == "forward":
            df_clean[col] = df_clean[col].fillna(method='ffill')
        elif fill_method == "backward":
            df_clean[col] = df_clean[col].fillna(method='bfill')
        elif fill_method == "interpolate":
            df_clean[col] = df_clean[col].interpolate()
        elif fill_method == "drop":
            df_clean = df_clean.dropna(subset=[col])
    
    return df_clean


def calculate_data_quality_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate data quality metrics for a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to analyze
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing data quality metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics["total_rows"] = len(df)
    metrics["total_columns"] = len(df.columns)
    
    # Missing data analysis
    missing_counts = df.isnull().sum()
    missing_percentages = (missing_counts / len(df) * 100).round(2)
    
    metrics["missing_data"] = {
        "total_missing": missing_counts.sum(),
        "missing_by_column": missing_counts.to_dict(),
        "missing_percentage_by_column": missing_percentages.to_dict(),
        "columns_with_missing": missing_counts[missing_counts > 0].index.tolist()
    }
    
    # Data type analysis
    metrics["data_types"] = df.dtypes.astype(str).to_dict()
    
    # Numeric columns analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        metrics["numeric_summary"] = {
            "columns": numeric_cols,
            "statistics": df[numeric_cols].describe().to_dict()
        }
    
    # Date columns analysis
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    if date_cols:
        metrics["date_summary"] = {
            "columns": date_cols,
            "date_ranges": {
                col: {
                    "min": df[col].min().strftime('%Y-%m-%d') if pd.notna(df[col].min()) else None,
                    "max": df[col].max().strftime('%Y-%m-%d') if pd.notna(df[col].max()) else None
                }
                for col in date_cols
            }
        }
    
    return metrics


def save_data_with_metadata(
    df: pd.DataFrame,
    filepath: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None,
    format: str = "csv"
) -> None:
    """
    Save DataFrame with associated metadata.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save
    filepath : Union[str, Path]
        Path to save the file
    metadata : Optional[Dict[str, Any]]
        Additional metadata to save
    format : str
        File format ('csv' or 'parquet')
    """
    filepath = Path(filepath)
    
    # Save the main data file
    if format.lower() == "parquet":
        df.to_parquet(filepath, index=False)
    else:
        df.to_csv(filepath, index=False)
    
    # Save metadata if provided
    if metadata:
        metadata_path = filepath.with_suffix('.metadata.json')
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human readable format.
    
    Parameters
    ----------
    size_bytes : int
        File size in bytes
        
    Returns
    -------
    str
        Formatted file size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"


def generate_timestamp(format: str = "%Y%m%d_%H%M%S") -> str:
    """
    Generate timestamp string.
    
    Parameters
    ----------
    format : str
        Timestamp format string
        
    Returns
    -------
    str
        Formatted timestamp string
    """
    return datetime.now().strftime(format)


# Example usage
if __name__ == "__main__":
    # Setup logging
    setup_logging(log_level="INFO")
    
    # Test data validation
    test_data = pd.DataFrame({
        'A': [1, 2, 3, None, 5],
        'B': [1.1, 2.2, 3.3, 4.4, 5.5],
        'C': ['a', 'b', 'c', 'd', 'e']
    })
    
    validation_results = validate_data(test_data)
    print("Validation results:", validation_results)
    
    # Test data splitting
    train, val, test = split_time_series(test_data, shuffle=False)
    print(f"Split sizes: {len(train)}, {len(val)}, {len(test)}")
    
    print("Helper functions test completed successfully") 