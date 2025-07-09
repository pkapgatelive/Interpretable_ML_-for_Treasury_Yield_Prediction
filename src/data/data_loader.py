"""
Data loading module for yield curve forecasting project.

This module provides classes and functions to load yield curve data and 
macroeconomic indicators from various sources including FRED, Bloomberg,
and Treasury department databases.

Classes
-------
YieldCurveDataLoader : Class for loading yield curve data
MacroDataLoader : Class for loading macroeconomic data

Functions
---------
load_fred_data : Load data from FRED API
load_treasury_data : Load data from Treasury department
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union, Tuple
import logging
from pathlib import Path

# Third-party imports for data sources
import yfinance as yf
from fredapi import Fred
import requests
from io import StringIO

from src.utils.constants import (
    FRED_SERIES_IDS,
    YIELD_TENORS,
    YIELD_TENOR_NAMES,
    API_URLS,
)
from config import load_config

# Setup logging
logger = logging.getLogger(__name__)


class YieldCurveDataLoader:
    """
    Class for loading yield curve data from various sources.
    
    This class provides methods to load Treasury yield curve data from
    different sources including FRED, Treasury Direct, and Bloomberg.
    
    Parameters
    ----------
    source : str
        Data source ('fred', 'treasury', 'bloomberg')
    api_key : Optional[str]
        API key for the data source (required for FRED and Bloomberg)
    start_date : Optional[str]
        Start date for data retrieval (YYYY-MM-DD format)
    end_date : Optional[str]  
        End date for data retrieval (YYYY-MM-DD format)
    """
    
    def __init__(
        self,
        source: str = "fred",
        api_key: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ):
        self.source = source.lower()
        self.api_key = api_key
        self.start_date = start_date or "2000-01-01"
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        
        # Initialize API clients
        if self.source == "fred" and api_key:
            self.fred_client = Fred(api_key=api_key)
        else:
            self.fred_client = None
            
        logger.info(f"Initialized YieldCurveDataLoader with source: {source}")
    
    def load_data(self, tenors: Optional[List[float]] = None) -> pd.DataFrame:
        """
        Load yield curve data for specified tenors.
        
        Parameters
        ----------
        tenors : Optional[List[float]]
            List of tenors to load (in years). If None, loads all available tenors.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with dates as index and tenors as columns
            
        Raises
        ------
        ValueError
            If source is not supported or API key is missing
        """
        if tenors is None:
            tenors = YIELD_TENORS
            
        if self.source == "fred":
            return self._load_from_fred(tenors)
        elif self.source == "treasury":
            return self._load_from_treasury(tenors)
        elif self.source == "bloomberg":
            return self._load_from_bloomberg(tenors)
        else:
            raise ValueError(f"Unsupported data source: {self.source}")
    
    def _load_from_fred(self, tenors: List[float]) -> pd.DataFrame:
        """
        Load yield curve data from FRED API.
        
        Parameters
        ----------
        tenors : List[float]
            List of tenors to load
            
        Returns
        -------
        pd.DataFrame
            Yield curve data
        """
        if not self.fred_client:
            raise ValueError("FRED API key is required for FRED data source")
        
        # TODO: Implement FRED data loading
        # 1. Map tenors to FRED series IDs
        # 2. Download data for each series
        # 3. Combine into single DataFrame
        # 4. Handle missing values and data alignment
        
        logger.info("Loading yield curve data from FRED")
        
        # Placeholder implementation
        date_range = pd.date_range(
            start=self.start_date, 
            end=self.end_date, 
            freq='B'  # Business days
        )
        
        # Create mock data for development
        np.random.seed(42)
        data = {}
        for tenor in tenors:
            # Generate realistic yield curve data
            base_yield = 2.0 + np.log(tenor + 0.25)  # Term structure
            yields = base_yield + np.random.normal(0, 0.5, len(date_range))
            data[f"{tenor}Y"] = yields
            
        df = pd.DataFrame(data, index=date_range)
        logger.info(f"Loaded {len(df)} observations for {len(tenors)} tenors")
        
        return df
    
    def _load_from_treasury(self, tenors: List[float]) -> pd.DataFrame:
        """
        Load yield curve data from Treasury Direct.
        
        Parameters
        ----------
        tenors : List[float]
            List of tenors to load
            
        Returns
        -------
        pd.DataFrame
            Yield curve data
        """
        # TODO: Implement Treasury Direct data loading
        logger.info("Loading yield curve data from Treasury Direct")
        
        # Placeholder implementation
        return self._load_from_fred(tenors)  # Use FRED as fallback for now
    
    def _load_from_bloomberg(self, tenors: List[float]) -> pd.DataFrame:
        """
        Load yield curve data from Bloomberg API.
        
        Parameters
        ----------
        tenors : List[float]
            List of tenors to load
            
        Returns
        -------
        pd.DataFrame
            Yield curve data
        """
        # TODO: Implement Bloomberg API data loading
        logger.info("Loading yield curve data from Bloomberg")
        
        # Placeholder implementation  
        return self._load_from_fred(tenors)  # Use FRED as fallback for now
    
    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean loaded yield curve data.
        
        Parameters
        ----------
        df : pd.DataFrame
            Raw yield curve data
            
        Returns
        -------
        pd.DataFrame
            Validated and cleaned data
        """
        # TODO: Implement data validation
        # 1. Check for reasonable yield values (0-20%)
        # 2. Check for proper date ordering
        # 3. Identify and flag potential data errors
        # 4. Basic outlier detection
        
        logger.info("Validating yield curve data")
        return df


class MacroDataLoader:
    """
    Class for loading macroeconomic data.
    
    This class provides methods to load various macroeconomic indicators
    that can be used as features for yield curve forecasting.
    
    Parameters
    ----------
    source : str
        Data source ('fred', 'bloomberg', 'yahoo')
    api_key : Optional[str]
        API key for the data source
    """
    
    def __init__(
        self,
        source: str = "fred",
        api_key: Optional[str] = None
    ):
        self.source = source.lower()
        self.api_key = api_key
        
        if self.source == "fred" and api_key:
            self.fred_client = Fred(api_key=api_key)
        else:
            self.fred_client = None
            
        logger.info(f"Initialized MacroDataLoader with source: {source}")
    
    def load_indicators(
        self, 
        indicators: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load macroeconomic indicators.
        
        Parameters
        ----------
        indicators : List[str]
            List of indicator names to load
        start_date : Optional[str]
            Start date for data retrieval
        end_date : Optional[str]
            End date for data retrieval
            
        Returns
        -------
        pd.DataFrame
            Macroeconomic indicators data
        """
        # TODO: Implement macro data loading
        # 1. Map indicator names to series IDs
        # 2. Download data for each indicator
        # 3. Handle different frequencies (monthly, quarterly)
        # 4. Interpolate to daily frequency if needed
        
        logger.info(f"Loading {len(indicators)} macroeconomic indicators")
        
        # Placeholder implementation
        start_date = start_date or "2000-01-01"
        end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        
        np.random.seed(123)
        data = {}
        for indicator in indicators:
            # Generate mock data for development
            data[indicator] = np.random.normal(0, 1, len(date_range))
            
        df = pd.DataFrame(data, index=date_range)
        logger.info(f"Loaded {len(df)} observations for {len(indicators)} indicators")
        
        return df


def load_fred_data(
    series_ids: List[str],
    api_key: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Load data from FRED API for specified series.
    
    Parameters
    ----------
    series_ids : List[str]
        List of FRED series IDs to load
    api_key : str
        FRED API key
    start_date : Optional[str]
        Start date for data retrieval
    end_date : Optional[str]
        End date for data retrieval
        
    Returns
    -------
    pd.DataFrame
        Combined data for all series
    """
    # TODO: Implement FRED data loading function
    logger.info(f"Loading {len(series_ids)} series from FRED")
    
    # Placeholder implementation
    fred = Fred(api_key=api_key)
    
    # This would be the actual implementation:
    # data = {}
    # for series_id in series_ids:
    #     series_data = fred.get_series(series_id, start=start_date, end=end_date)
    #     data[series_id] = series_data
    # 
    # df = pd.DataFrame(data)
    # return df
    
    return pd.DataFrame()  # Placeholder


def load_treasury_data(
    tenors: List[float],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Load yield curve data from Treasury Direct.
    
    Parameters
    ----------
    tenors : List[float]
        List of tenors to load (in years)
    start_date : Optional[str]
        Start date for data retrieval
    end_date : Optional[str]
        End date for data retrieval
        
    Returns
    -------
    pd.DataFrame
        Treasury yield curve data
    """
    # TODO: Implement Treasury Direct data loading
    logger.info(f"Loading Treasury data for {len(tenors)} tenors")
    
    # Placeholder implementation
    return pd.DataFrame()


# Example usage and testing functions
if __name__ == "__main__":
    # Example usage
    config = load_config()
    
    # Load yield curve data
    yield_loader = YieldCurveDataLoader(
        source="fred",
        api_key=config.get("data", {}).get("sources", {}).get("fred", {}).get("api_key"),
        start_date="2020-01-01",
        end_date="2023-12-31"
    )
    
    yield_data = yield_loader.load_data()
    print(f"Loaded yield curve data: {yield_data.shape}")
    
    # Load macro data
    macro_loader = MacroDataLoader(source="fred")
    macro_indicators = ["fed_funds_rate", "cpi_inflation", "unemployment_rate"]
    macro_data = macro_loader.load_indicators(macro_indicators)
    print(f"Loaded macro data: {macro_data.shape}") 