#!/usr/bin/env python3
"""
Generate mock data for yield curve forecasting pipeline testing.

This script creates synthetic Treasury yield data and macro indicators
to test the full pipeline when API keys are not available.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Set random seed for reproducibility
np.random.seed(42)

def generate_yield_data(start_date: str = "2020-01-01", end_date: str = "2023-12-31"):
    """Generate synthetic Treasury yield curve data."""
    
    # Create date range
    dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
    
    # Treasury tenors (in years)
    tenors = ['3M', '6M', '1Y', '2Y', '3Y', '5Y', '7Y', '10Y', '20Y', '30Y']
    tenor_values = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]
    
    # Generate base yield curves using Nelson-Siegel-like model
    n_obs = len(dates)
    yields_data = {}
    
    # Base parameters for yield curve shape
    beta0 = 2.5  # Level
    beta1 = -1.0  # Slope
    beta2 = 2.0   # Curvature
    tau = 2.0     # Decay parameter
    
    for i, tenor in enumerate(tenors):
        tenor_val = tenor_values[i]
        
        # Nelson-Siegel model with random walk
        base_yield = beta0 + beta1 * (1 - np.exp(-tenor_val/tau)) / (tenor_val/tau) + \
                    beta2 * ((1 - np.exp(-tenor_val/tau)) / (tenor_val/tau) - np.exp(-tenor_val/tau))
        
        # Add time series dynamics (random walk with drift)
        drift = 0.0001  # Small drift
        volatility = 0.01  # Daily volatility
        
        # Generate random walk
        innovations = np.random.normal(0, volatility, n_obs)
        cumulative_innovations = np.cumsum(innovations) + drift * np.arange(n_obs)
        
        # Final yield series
        yields_data[tenor] = base_yield + cumulative_innovations
        
        # Ensure positive yields
        yields_data[tenor] = np.maximum(yields_data[tenor], 0.01)
    
    # Create DataFrame
    yield_df = pd.DataFrame(yields_data, index=dates)
    
    return yield_df

def generate_macro_data(start_date: str = "2020-01-01", end_date: str = "2023-12-31"):
    """Generate synthetic macroeconomic indicators."""
    
    # Create date range (monthly data)
    dates = pd.date_range(start=start_date, end=end_date, freq='MS')  # Month start
    n_obs = len(dates)
    
    # Generate macro variables
    macro_data = {}
    
    # Fed Funds Rate (random walk around 2%)
    fed_funds = 2.0 + np.cumsum(np.random.normal(0, 0.1, n_obs))
    macro_data['FEDFUNDS'] = np.maximum(fed_funds, 0.01)  # Non-negative
    
    # CPI Inflation (random walk around 2.5%)
    cpi_inflation = 2.5 + np.cumsum(np.random.normal(0, 0.2, n_obs))
    macro_data['CPIAUCSL'] = np.maximum(cpi_inflation, 0.1)
    
    # Unemployment Rate (random walk around 5%)
    unemployment = 5.0 + np.cumsum(np.random.normal(0, 0.1, n_obs))
    macro_data['UNRATE'] = np.maximum(unemployment, 0.5)
    
    # GDP Growth (quarterly, interpolated to monthly)
    gdp_growth = 2.0 + np.random.normal(0, 0.5, n_obs)
    macro_data['GDP'] = gdp_growth
    
    # Dollar Index (random walk around 100)
    dollar_index = 100 + np.cumsum(np.random.normal(0, 0.5, n_obs))
    macro_data['DTWEXBGS'] = dollar_index
    
    # VIX (random walk around 20 with higher volatility)
    vix = 20 + np.cumsum(np.random.normal(0, 1, n_obs))
    macro_data['VIXCLS'] = np.maximum(vix, 5)  # VIX can't be too low
    
    # Create DataFrame
    macro_df = pd.DataFrame(macro_data, index=dates)
    
    return macro_df

def main():
    """Generate and save mock data."""
    
    print("🔄 Generating synthetic data for pipeline testing...")
    
    # Create output directories
    os.makedirs("data/raw", exist_ok=True)
    
    # Generate yield curve data
    print("📈 Generating Treasury yield curve data...")
    yield_data = generate_yield_data()
    yield_filename = f"data/raw/treasury_yields_{datetime.now().strftime('%Y%m%d')}.csv"
    yield_data.to_csv(yield_filename)
    print(f"✅ Saved yield data: {yield_filename} ({yield_data.shape[0]} rows, {yield_data.shape[1]} columns)")
    
    # Generate macro data
    print("📊 Generating macroeconomic indicators...")
    macro_data = generate_macro_data()
    macro_filename = f"data/raw/macro_indicators_{datetime.now().strftime('%Y%m%d')}.csv"
    macro_data.to_csv(macro_filename)
    print(f"✅ Saved macro data: {macro_filename} ({macro_data.shape[0]} rows, {macro_data.shape[1]} columns)")
    
    # Display sample data
    print("\n📋 Sample Treasury Yield Data:")
    print(yield_data.head())
    print("\n📋 Sample Macro Data:")
    print(macro_data.head())
    
    print(f"\n🎉 Mock data generation completed successfully!")
    print(f"Data saved to data/raw/ directory")

if __name__ == "__main__":
    main() 