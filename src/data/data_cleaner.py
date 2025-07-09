"""
Data cleaning and preprocessing module for yield curve forecasting.

This module contains utilities for cleaning financial time series data,
handling missing values, and detecting/removing outliers.
"""

import pandas as pd
import numpy as np
from typing import Union, List, Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class YieldCurvePreprocessor:
    """
    Main class for preprocessing yield curve data.
    
    Handles missing values, outlier detection, and data normalization
    for yield curve time series.
    """
    
    def __init__(self, 
                 missing_value_method: str = "interpolate",
                 outlier_method: str = "iqr",
                 outlier_threshold: float = 3.0):
        """
        Initialize the preprocessor.
        
        Parameters
        ----------
        missing_value_method : str
            Method for handling missing values ("interpolate", "forward_fill", "drop")
        outlier_method : str
            Method for outlier detection ("iqr", "z_score", "isolation_forest")
        outlier_threshold : float
            Threshold for outlier detection
        """
        self.missing_value_method = missing_value_method
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the input data by handling missing values and outliers.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data with potential missing values and outliers
            
        Returns
        -------
        pd.DataFrame
            Cleaned data
        """
        logger.info("Starting data cleaning process...")
        
        # Handle missing values
        data_clean = handle_missing_values(data, method=self.missing_value_method)
        
        # Detect and handle outliers
        outliers = detect_outliers(data_clean, method=self.outlier_method, 
                                 threshold=self.outlier_threshold)
        data_clean = remove_outliers(data_clean, outliers)
        
        logger.info(f"Data cleaning completed. Shape: {data_clean.shape}")
        return data_clean


def handle_missing_values(data: pd.DataFrame, 
                         method: str = "interpolate") -> pd.DataFrame:
    """
    Handle missing values in time series data.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data with missing values
    method : str
        Method for handling missing values
        
    Returns
    -------
    pd.DataFrame
        Data with missing values handled
    """
    data_copy = data.copy()
    
    if method == "interpolate":
        # Linear interpolation for time series
        data_copy = data_copy.interpolate(method='linear', limit_direction='both')
    elif method == "forward_fill":
        data_copy = data_copy.fillna(method='ffill')
    elif method == "backward_fill":
        data_copy = data_copy.fillna(method='bfill')
    elif method == "drop":
        data_copy = data_copy.dropna()
    else:
        logger.warning(f"Unknown method {method}. Using interpolation.")
        data_copy = data_copy.interpolate(method='linear', limit_direction='both')
    
    missing_count = data.isnull().sum().sum()
    remaining_missing = data_copy.isnull().sum().sum()
    logger.info(f"Handled {missing_count - remaining_missing} missing values. "
                f"{remaining_missing} missing values remain.")
    
    return data_copy


def detect_outliers(data: pd.DataFrame, 
                   method: str = "iqr",
                   threshold: float = 3.0) -> pd.DataFrame:
    """
    Detect outliers in the data.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data
    method : str
        Method for outlier detection
    threshold : float
        Threshold for outlier detection
        
    Returns
    -------
    pd.DataFrame
        Boolean mask indicating outliers
    """
    outliers = pd.DataFrame(False, index=data.index, columns=data.columns)
    
    for column in data.select_dtypes(include=[np.number]).columns:
        if method == "iqr":
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers[column] = (data[column] < lower_bound) | (data[column] > upper_bound)
            
        elif method == "z_score":
            z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
            outliers[column] = z_scores > threshold
            
        elif method == "modified_z_score":
            median = data[column].median()
            mad = np.median(np.abs(data[column] - median))
            modified_z_scores = 0.6745 * (data[column] - median) / mad
            outliers[column] = np.abs(modified_z_scores) > threshold
    
    total_outliers = outliers.sum().sum()
    logger.info(f"Detected {total_outliers} outliers using {method} method.")
    
    return outliers


def remove_outliers(data: pd.DataFrame, 
                   outliers: pd.DataFrame,
                   method: str = "cap") -> pd.DataFrame:
    """
    Remove or cap outliers in the data.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data
    outliers : pd.DataFrame
        Boolean mask indicating outliers
    method : str
        Method for handling outliers ("remove", "cap", "interpolate")
        
    Returns
    -------
    pd.DataFrame
        Data with outliers handled
    """
    data_clean = data.copy()
    
    if method == "remove":
        # Set outliers to NaN
        data_clean[outliers] = np.nan
        
    elif method == "cap":
        # Cap outliers at 5th and 95th percentiles
        for column in data.select_dtypes(include=[np.number]).columns:
            if outliers[column].any():
                lower_cap = data[column].quantile(0.05)
                upper_cap = data[column].quantile(0.95)
                data_clean.loc[outliers[column], column] = np.clip(
                    data_clean.loc[outliers[column], column], 
                    lower_cap, upper_cap
                )
    
    elif method == "interpolate":
        # Set outliers to NaN and then interpolate
        data_clean[outliers] = np.nan
        data_clean = data_clean.interpolate(method='linear', limit_direction='both')
    
    return data_clean


def validate_data_quality(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate data quality and generate quality report.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data to validate
        
    Returns
    -------
    Dict[str, Any]
        Data quality report
    """
    report = {
        "shape": data.shape,
        "missing_values": data.isnull().sum().to_dict(),
        "data_types": data.dtypes.to_dict(),
        "memory_usage": data.memory_usage(deep=True).sum(),
        "duplicate_rows": data.duplicated().sum(),
    }
    
    # Add numeric statistics
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 0:
        report["numeric_summary"] = data[numeric_columns].describe().to_dict()
    
    return report 